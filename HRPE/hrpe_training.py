import cntk as C
import cntkx as Cx
import cv2
import numpy as np
import os
import pandas as pd
import random

from cntk.layers import Convolution2D, Dense
from cntkx.learners import CyclicalLearningRate

img_channel = 3
img_height = 320
img_width = 480
num_keypoint = 16

epoch_size = 160
minibatch_size = 16
num_samples = 14797

step_size = num_samples // minibatch_size * 10


class HigherResolutionPoseEstimation:
    def __init__(self, map_file, is_train):
        self.sample_count = 0
        self.minibatch_count = 0
        
        with open(map_file) as f:
            self.map_list = f.readlines()
        if is_train:
            random.shuffle(self.map_list)

    def next_minibatch(self, minibatch_size):
        self.minibatch_count = minibatch_size
        
        batch_img = np.zeros((minibatch_size, img_channel, img_height, img_width), dtype="float32")
        batch_kps = np.zeros((minibatch_size, num_keypoint, img_height // 4, img_width // 4), dtype="float32")
        batch_psn = np.zeros((minibatch_size, 1), dtype="float32")

        batch_file = self.map_list[self.sample_count: self.sample_count + minibatch_size]
        for i, line in enumerate(batch_file):
            img_file, kps_file, psn = line[:-1].split("\t")

            batch_img[i] = np.ascontiguousarray(cv2.imread(img_file).transpose(2, 0, 1), dtype="float32")
            batch_kps[i] = np.load(kps_file)
            batch_psn[i] = psn

        self.sample_count += self.minibatch_count

        return batch_img, batch_kps, batch_psn

    def randomize(self):
        random.shuffle(self.map_list)


def GlobalSumPooling(name=''):

    @C.BlockFunction('GlobalSumPooling', name)
    def global_sum_pooling(x):
        return C.reduce_sum(x, axis=(1, 2))

    return global_sum_pooling


def hrpe320x480(h):
    hrnet = C.load_model("./hrnet.model")  # pretrained model
    h = C.combine([hrnet.find_by_name("higher")]).clone(method="clone", substitutions={hrnet.arguments[0]: h})

    #
    # counting person
    #
    count = Convolution2D((1, 1), 512, activation=Cx.mish, init=C.normal(0.01), pad=True, strides=1)(h)
    count = GlobalSumPooling()(count)
    count = Dense(1, init=C.normal(0.01))(count)

    #
    # confidence map
    #
    conf = Convolution2D((3, 3), 512, activation=Cx.mish, init=C.normal(0.01), pad=True, strides=1)(h)
    conf = Convolution2D((3, 3), 512, activation=Cx.mish, init=C.normal(0.01), pad=True, strides=1)(conf)
    conf = Convolution2D((3, 3), 512, activation=Cx.mish, init=C.normal(0.01), pad=True, strides=1)(conf)
    conf = Convolution2D((3, 3), 512, activation=Cx.mish, init=C.normal(0.01), pad=True, strides=1)(conf)
    conf = Convolution2D((3, 3), 512, activation=Cx.mish, init=C.normal(0.01), pad=True, strides=1)(conf)
    conf = Convolution2D((3, 3), 512, activation=Cx.mish, init=C.normal(0.01), pad=True, strides=1)(conf)
    conf = Convolution2D((1, 1), num_keypoint, init=C.normal(0.01), pad=True, strides=1)(conf)

    return conf, count


if __name__ == "__main__":
    #
    # minibatch reader
    #
    train_reader = HigherResolutionPoseEstimation("./train_hrpe320x480_map.txt", is_train=True)

    #
    # input, heatmap, person, and model
    #
    input = C.input_variable(shape=(img_channel, img_height, img_width), dtype="float32", needs_gradient=True)
    heatmap = C.input_variable(shape=(num_keypoint, img_height // 4, img_width // 4), dtype="float32")
    person = C.input_variable(shape=(1,), dtype="float32")

    conf, count = hrpe320x480(input)
    
    #
    # loss function
    #
    heatmap_loss = C.reduce_sum(C.reduce_mean(C.square(conf - heatmap), axis=(1, 2), keepdims=False))
    counting_loss = C.reduce_sum_square(count - person)
    loss = heatmap_loss + counting_loss

    #
    # optimizer and cyclical learning rate
    #
    learner = C.adam(loss.parameters, lr=C.learning_parameter_schedule_per_sample(0.1), momentum=0.9,
                     gradient_clipping_threshold_per_sample=minibatch_size, gradient_clipping_with_truncation=True)
    clr = CyclicalLearningRate(learner, base_lr=1e-8, max_lr=1e-4, ramp_up_step_size=step_size,
                               minibatch_size=minibatch_size, lr_policy="triangular2")
    progress_printer = C.logging.ProgressPrinter(tag="Training")

    trainer = C.Trainer(loss, (loss, None), [learner], [progress_printer])

    C.logging.log_number_of_parameters(loss)

    #
    # training
    #
    logging = {"epoch": [], "loss": []}
    for epoch in range(epoch_size):
        sample_count = 0
        epoch_loss = 0
        epoch_metric = 0
        while sample_count < num_samples:
            batch_img, batch_kps, batch_psn = train_reader.next_minibatch(
                min(minibatch_size, num_samples - sample_count))

            trainer.train_minibatch({input: batch_img, heatmap: batch_kps, person: batch_psn})

            clr.batch_step()

            sample_count += train_reader.minibatch_count
            epoch_loss += trainer.previous_minibatch_loss_average
            
        #
        # loss and error logging
        #
        logging["epoch"].append(epoch + 1)
        logging["loss"].append(epoch_loss / (num_samples / minibatch_size))
        
        trainer.summarize_training_progress()

        train_reader.sample_count = 0
        train_reader.randomize()

    #
    # save model and logging
    #
    model = C.combine(conf, count)
    model.save("./hrpe.model")
    print("Saved model.")

    df = pd.DataFrame(logging)
    df.to_csv("./hrpe.csv", index=False)
    print("Saved logging.")
    
