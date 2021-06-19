import cntk as C
import cntkx as Cx
import cv2
import h5py
import numpy as np
import os
import pandas as pd
import random

from cntk.layers import BatchNormalization, Convolution2D, MaxPooling
from cntkx.learners import CyclicalLearningRate

img_channel = 3
img_height = 320
img_width = 480
jpu_channel = 512
num_classes = 150 + 1  # 150 categories + background

epoch_size = 100
minibatch_size = 8
num_samples = 20210

step_size = num_samples // minibatch_size * 10


class RealTimeSemanticSegmentation:
    def __init__(self, map_file, is_train):
        self.sample_count = 0
        self.minibatch_count = 0
        
        with open(map_file) as f:
            self.map_list = f.readlines()
        if is_train:
            random.shuffle(self.map_list)

    def next_minibatch(self, minibatch_size):
        self.minibatch_count = minibatch_size
        
        batch_image = np.zeros((minibatch_size, img_channel, img_height, img_width), dtype="float32")
        batch_label = np.zeros((minibatch_size, num_classes, img_height, img_width), dtype="float32")

        batch_file = self.map_list[self.sample_count: self.sample_count + minibatch_size]
        for i, line in enumerate(batch_file):
            img_file, ann_file = line[:-1].split("\t")

            batch_image[i] = np.ascontiguousarray(cv2.imread(img_file).transpose(2, 0, 1), dtype="float32")
            batch_label[i] = np.load(ann_file)

        self.sample_count += self.minibatch_count

        return batch_image, batch_label

    def randomize(self):
        random.shuffle(self.map_list)


def convolution(weights, pad=True, stride=1, name=''):
    W = C.Constant(value=weights, name='W')

    @C.BlockFunction('Convolution2D', name)
    def conv2d(x):
        return C.convolution(W, x, strides=[stride, stride], auto_padding=[False, pad, pad])

    return conv2d


def batch_normalization(scale, bias, mean, variance, spatial=True, name=''):
    scale = C.Constant(value=scale, name='scale')
    bias = C.Constant(value=bias, name='bias')
    mu = C.Constant(value=mean, name='aggregate_mean')
    sigma = C.Constant(value=variance, name='aggregate_variance')

    @C.BlockFunction('BatchNormalization', name)
    def batch_norm(x):
        return C.batch_normalization(x, scale, bias, mu, sigma, spatial=spatial, running_count=C.constant(5000))

    return batch_norm


def SeparableConvolution2D(in_channel, out_channel, kernel_size, pad=True, stride=1, dilation=1, name=''):
    """ Separable Convolution 2D (2017) """
    W_depthwise = C.parameter(shape=(in_channel, 1, kernel_size, kernel_size), init=C.he_normal(), name='W')
    bn_depthwise = BatchNormalization(map_rank=1, use_cntk_engine=True)

    W_pointwise = C.parameter(shape=(out_channel, in_channel, 1, 1), init=C.he_normal(), name='W')
    bn_pointwise = BatchNormalization(map_rank=1, use_cntk_engine=True)

    #
    # strides, padding, and dilation
    #
    strides = []
    padding = [False]
    dilated = [1]
    for _ in range(2):
        strides.append(stride)
        padding.append(pad)
        dilated.append(dilation)

    @C.BlockFunction('SeparableConvolution2D', name)
    def separable_convolution(x):
        h = C.convolution(W_depthwise, x, strides=strides, auto_padding=padding, dilation=dilated, groups=in_channel)
        h = bn_depthwise(h)
        h = C.convolution(W_pointwise, h, strides=strides, auto_padding=padding)
        h = bn_pointwise(h)
        return h

    return separable_convolution


def Upsample(shape, name=''):
    @C.BlockFunction('Upsample', name)
    def upsample(x):
        xr = C.reshape(x, (shape[0], shape[1], 1, shape[2], 1))
        xx = C.splice(xr, xr, axis=-1)
        xy = C.splice(xx, xx, axis=-3)
        r = C.reshape(xy, (shape[0], shape[1] * 2, shape[2] * 2))
        return r

    return upsample


def JointPyramidUpsampling(shape, name=''):
    """ Joint Pyramid Upsampling """
    conv3 = Convolution2D((3, 3), shape[0], init=C.he_normal(), pad=True, strides=1, bias=False)
    conv4 = Convolution2D((3, 3), shape[0], init=C.he_normal(), pad=True, strides=1, bias=False)
    conv5 = Convolution2D((3, 3), shape[0], init=C.he_normal(), pad=True, strides=1, bias=False)

    bn3 = BatchNormalization(map_rank=1, use_cntk_engine=True)
    bn4 = BatchNormalization(map_rank=1, use_cntk_engine=True)
    bn5 = BatchNormalization(map_rank=1, use_cntk_engine=True)

    dilated_conv1 = SeparableConvolution2D(shape[0] * 3, shape[0], 3, dilation=1)
    dilated_conv2 = SeparableConvolution2D(shape[0] * 3, shape[0], 3, dilation=2)
    dilated_conv3 = SeparableConvolution2D(shape[0] * 3, shape[0], 3, dilation=4)
    dilated_conv4 = SeparableConvolution2D(shape[0] * 3, shape[0], 3, dilation=8)

    @C.BlockFunction('JointPyramidUpsampling', name)
    def jpu(h3, h4, h5):
        h3 = Cx.mish(bn3(conv3(h3)))

        h4 = Cx.mish(bn4(conv4(h4)))
        h4 = Upsample((shape[0], shape[1] // 2, shape[2] // 2))(h4)

        h5 = Cx.mish(bn5(conv5(h5)))
        h5 = Upsample((shape[0], shape[1] // 4, shape[2] // 4))(h5)
        h5 = Upsample((shape[0], shape[1] // 2, shape[2] // 2))(h5)

        h = C.splice(h3, h4, h5, axis=0)

        dilated1 = dilated_conv1(h)
        dilated2 = dilated_conv2(h)
        dilated3 = dilated_conv3(h)
        dilated4 = dilated_conv4(h)

        return Cx.mish(C.splice(dilated1, dilated2, dilated3, dilated4, axis=0))

    return jpu


def osa_module(h0, params1, params2, params3, params4, params5, params):
    h1 = Cx.mish(batch_normalization(params1[1], params1[2], params1[3], params1[4])(convolution(params1[0])(h0)))
    h2 = Cx.mish(batch_normalization(params2[1], params2[2], params2[3], params2[4])(convolution(params2[0])(h1)))
    h3 = Cx.mish(batch_normalization(params3[1], params3[2], params3[3], params3[4])(convolution(params3[0])(h2)))
    h4 = Cx.mish(batch_normalization(params4[1], params4[2], params4[3], params4[4])(convolution(params4[0])(h3)))
    h5 = Cx.mish(batch_normalization(params5[1], params5[2], params5[3], params5[4])(convolution(params5[0])(h4)))

    h = C.splice(h1, h2, h3, h4, h5, axis=0)

    return Cx.mish(batch_normalization(params[1], params[2], params[3], params[4])(convolution(params[0])(h)))


def vovnet57(h):
    def parameter_names(n):
        return ["conv%s/weights" % n, "bn%s/scale" % n, "bn%s/bias" % n, "bn%s/mean" % n, "bn%s/variance" % n]

    with h5py.File("./vovnet57.h5", "r") as f:
        for i, stride in enumerate([2, 1, 1]):
            h = convolution(f["params/conv%d/weights" % i][()], stride=stride)(h)
            h = batch_normalization(f["params/bn%d/scale" % i][()], f["params/bn%d/bias" % i][()],
                                    f["params/bn%d/mean" % i][()], f["params/bn%d/variance" % i][()])(h)
            h = Cx.mish(h)

        h = MaxPooling((3, 3), strides=2, pad=True)(h)

        h2 = osa_module(h,
                        [f["params/osa1_%s" % s][()] for s in parameter_names(1)],
                        [f["params/osa1_%s" % s][()] for s in parameter_names(2)],
                        [f["params/osa1_%s" % s][()] for s in parameter_names(3)],
                        [f["params/osa1_%s" % s][()] for s in parameter_names(4)],
                        [f["params/osa1_%s" % s][()] for s in parameter_names(5)],
                        [f["params/osa1_%s" % s][()] for s in parameter_names("")])

        h = MaxPooling((3, 3), strides=2, pad=True)(h2)

        h3 = osa_module(h,
                        [f["params/osa2_%s" % s][()] for s in parameter_names(1)],
                        [f["params/osa2_%s" % s][()] for s in parameter_names(2)],
                        [f["params/osa2_%s" % s][()] for s in parameter_names(3)],
                        [f["params/osa2_%s" % s][()] for s in parameter_names(4)],
                        [f["params/osa2_%s" % s][()] for s in parameter_names(5)],
                        [f["params/osa2_%s" % s][()] for s in parameter_names("")])

        h = MaxPooling((3, 3), strides=2, pad=True)(h3)

        h = osa_module(h,
                       [f["params/osa3_1_%s" % s][()] for s in parameter_names(1)],
                       [f["params/osa3_1_%s" % s][()] for s in parameter_names(2)],
                       [f["params/osa3_1_%s" % s][()] for s in parameter_names(3)],
                       [f["params/osa3_1_%s" % s][()] for s in parameter_names(4)],
                       [f["params/osa3_1_%s" % s][()] for s in parameter_names(5)],
                       [f["params/osa3_1_%s" % s][()] for s in parameter_names("")])
        h = osa_module(h,
                       [f["params/osa3_2_%s" % s][()] for s in parameter_names(1)],
                       [f["params/osa3_2_%s" % s][()] for s in parameter_names(2)],
                       [f["params/osa3_2_%s" % s][()] for s in parameter_names(3)],
                       [f["params/osa3_2_%s" % s][()] for s in parameter_names(4)],
                       [f["params/osa3_2_%s" % s][()] for s in parameter_names(5)],
                       [f["params/osa3_2_%s" % s][()] for s in parameter_names("")])
        h = osa_module(h,
                       [f["params/osa3_3_%s" % s][()] for s in parameter_names(1)],
                       [f["params/osa3_3_%s" % s][()] for s in parameter_names(2)],
                       [f["params/osa3_3_%s" % s][()] for s in parameter_names(3)],
                       [f["params/osa3_3_%s" % s][()] for s in parameter_names(4)],
                       [f["params/osa3_3_%s" % s][()] for s in parameter_names(5)],
                       [f["params/osa3_3_%s" % s][()] for s in parameter_names("")])
        h4 = osa_module(h,
                        [f["params/osa3_4_%s" % s][()] for s in parameter_names(1)],
                        [f["params/osa3_4_%s" % s][()] for s in parameter_names(2)],
                        [f["params/osa3_4_%s" % s][()] for s in parameter_names(3)],
                        [f["params/osa3_4_%s" % s][()] for s in parameter_names(4)],
                        [f["params/osa3_4_%s" % s][()] for s in parameter_names(5)],
                        [f["params/osa3_4_%s" % s][()] for s in parameter_names("")])

        h = MaxPooling((3, 3), strides=2, pad=True)(h4)

        h = osa_module(h,
                       [f["params/osa4_1_%s" % s][()] for s in parameter_names(1)],
                       [f["params/osa4_1_%s" % s][()] for s in parameter_names(2)],
                       [f["params/osa4_1_%s" % s][()] for s in parameter_names(3)],
                       [f["params/osa4_1_%s" % s][()] for s in parameter_names(4)],
                       [f["params/osa4_1_%s" % s][()] for s in parameter_names(5)],
                       [f["params/osa4_1_%s" % s][()] for s in parameter_names("")])
        h = osa_module(h,
                       [f["params/osa4_2_%s" % s][()] for s in parameter_names(1)],
                       [f["params/osa4_2_%s" % s][()] for s in parameter_names(2)],
                       [f["params/osa4_2_%s" % s][()] for s in parameter_names(3)],
                       [f["params/osa4_2_%s" % s][()] for s in parameter_names(4)],
                       [f["params/osa4_2_%s" % s][()] for s in parameter_names(5)],
                       [f["params/osa4_2_%s" % s][()] for s in parameter_names("")])
        h5 = osa_module(h,
                        [f["params/osa4_3_%s" % s][()] for s in parameter_names(1)],
                        [f["params/osa4_3_%s" % s][()] for s in parameter_names(2)],
                        [f["params/osa4_3_%s" % s][()] for s in parameter_names(3)],
                        [f["params/osa4_3_%s" % s][()] for s in parameter_names(4)],
                        [f["params/osa4_3_%s" % s][()] for s in parameter_names(5)],
                        [f["params/osa4_3_%s" % s][()] for s in parameter_names("")])

        return h2, h3, h4, h5


def rtss320x480(h):
    with C.layers.default_options(init=C.he_normal(), pad=True, strides=1, bias=False,
                                  map_rank=1, use_cntk_engine=True):
        #
        # backbone
        #
        h2, h3, h4, h5 = vovnet57(h)

        #
        # JPU
        #
        h = JointPyramidUpsampling((jpu_channel, img_height // 8, img_width // 8))(h3, h4, h5)

        #
        # decoder
        #
        h = Cx.mish(BatchNormalization()(Convolution2D((1, 1), 320)(h)))

        h = C.splice(Upsample(h.shape)(h), Cx.mish(BatchNormalization()(Convolution2D((1, 1), 64)(h2))), axis=0)

        h = Cx.mish(SeparableConvolution2D(384, 384, 3)(h))
        h = Upsample(h.shape)(h)
        h = Upsample(h.shape)(h)
        h = Convolution2D((1, 1), num_classes, activation=None, bias=True, init=C.glorot_uniform())(h)

        return h


def dice_coefficient(output_vector, target_vector, epsilon=1e-5):
    intersection = C.reduce_sum(output_vector * target_vector)
    union = C.reduce_sum(output_vector) + C.reduce_sum(target_vector) + epsilon

    return 2 * intersection / union


def generalized_dice_loss(output_vector, target_vector, epsilon=1e-5):
    weights = 1 / C.reduce_sum_square(target_vector, axis=(1, 2))
    intersection = C.reduce_sum(weights * C.reduce_sum(output_vector * target_vector, axis=(1, 2)))
    union = C.reduce_sum(weights * C.reduce_sum(output_vector + target_vector, axis=(1, 2))) + epsilon

    return 1 - 2 * (intersection / union)


if __name__ == "__main__":
    #
    # minibatch reader
    #
    train_reader = RealTimeSemanticSegmentation("./train_rtss320x480_map.txt", is_train=True)

    #
    # input, label, boundary, and model
    #
    input = C.input_variable(shape=(img_channel, img_height, img_width), dtype="float32", needs_gradient=True)
    label = C.input_variable(shape=(num_classes, img_height, img_width), dtype="float32")

    model = rtss320x480(input / 255.0)

    C.logging.log_number_of_parameters(model)

    #
    # loss function and error metrics
    #
    predict_loss = C.reduce_mean(Cx.focal_loss_with_softmax(model, label, axis=0))
    dice_loss = generalized_dice_loss(C.softmax(model, axis=0), label)

    loss = predict_loss + dice_loss
    dice = dice_coefficient(C.softmax(model, axis=0), label)

    #
    # optimizer and cyclical learning rate
    #
    learner = C.adam(model.parameters, lr=C.learning_parameter_schedule_per_sample(1e-3), momentum=0.9,
                     gradient_clipping_threshold_per_sample=minibatch_size, gradient_clipping_with_truncation=True)
    clr = CyclicalLearningRate(learner, base_lr=1e-5, max_lr=1e-3, ramp_up_step_size=step_size,
                               minibatch_size=minibatch_size, lr_policy="triangular2")
    progress_printer = C.logging.ProgressPrinter(tag="Training")

    trainer = C.Trainer(model, (loss, dice), [learner], [progress_printer])

    C.logging.log_number_of_parameters(model)

    #
    # training
    #
    logging = {"epoch": [], "loss": [], "dice": []}
    for epoch in range(epoch_size):
        sample_count = 0
        epoch_loss = 0
        epoch_metric = 0
        while sample_count < num_samples:
            batch_image, batch_label = train_reader.next_minibatch(min(minibatch_size, num_samples - sample_count))

            trainer.train_minibatch({input: batch_image, label: batch_label})

            clr.batch_step()

            sample_count += train_reader.minibatch_count
            epoch_loss += trainer.previous_minibatch_loss_average
            epoch_metric += trainer.previous_minibatch_evaluation_average

        #
        # loss and error logging
        #
        logging["epoch"].append(epoch + 1)
        logging["loss"].append(epoch_loss / (num_samples / minibatch_size))
        logging["dice"].append(epoch_metric / (num_samples / minibatch_size))

        trainer.summarize_training_progress()

        train_reader.sample_count = 0
        train_reader.randomize()

    #
    # save model and logging
    #
    model.save("./rtss.model")
    print("Saved model.")

    df = pd.DataFrame(logging)
    df.to_csv("./rtss.csv", index=False)
    print("Saved logging.")
    
