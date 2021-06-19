import cntk as C
import cntk.io.transforms as xforms
import h5py
import numpy as np
import os
import pandas as pd

from cntk.layers import BatchNormalization, Convolution2D
from cntkx.learners import CyclicalLearningRate

img_channel = 3
img_height = 416
img_width = 416

num_bboxes = 1330
num_classes = 80
num_channel = 5 + num_classes

epoch_size = 100
minibatch_size = 64
num_samples = 82801

anchor_boxes = np.load("./anchor_boxes.npy")
sample_size = 8
step_size = num_samples // sample_size * 10


class SingleShotMultiDetector:
    def __init__(self, map_file, box_file, train):
        self.sample_count = 0
        self.minibatch_count = 0
        
        self.noassign_bbox = np.concatenate(  # target for no assignment predict bounding box
            (np.concatenate(((center_offset(3, 7) + 0.5) / 7, prior_anchor(3, 7, anchor_boxes[2:])), axis=1),
             np.concatenate(((center_offset(3, 13) + 0.5) / 13, prior_anchor(3, 13, anchor_boxes[1:4])), axis=1),
             np.concatenate(((center_offset(1, 26) + 0.5) / 26, prior_anchor(1, 26, anchor_boxes[:1, :])), axis=1)),
            axis=0).reshape(1, -1, 4)
        
        self.bbox_reader = create_reader(map_file, box_file, train)
        self.input = C.input_variable(shape=(img_channel, img_height, img_width), dtype="float32")
        self.bbox = C.sequence.input_variable(shape=(4,), dtype="float32") * 1
        self.label = C.sequence.input_variable(shape=(num_classes,), dtype="float32") * 1

    def next_minibatch(self, minibatch_size, model):
        minibatch_data = self.bbox_reader.next_minibatch(minibatch_size,
                                                         input_map={self.input: self.bbox_reader.streams.images,
                                                                    self.bbox: self.bbox_reader.streams.bboxes,
                                                                    self.label: self.bbox_reader.streams.labels})

        self.minibatch_count = minibatch_data[self.bbox].num_sequences

        target_label = self.label.eval({self.label.arguments[0]: list(minibatch_data.values())[2]})
        target_bbox = self.bbox.eval({self.bbox.arguments[0]: list(minibatch_data.values())[1]})

        assign_image = list(minibatch_data.values())[0].asarray()
        assign_bbox = np.repeat(self.noassign_bbox, self.minibatch_count, axis=0)
        assign_conf = np.zeros((self.minibatch_count, num_bboxes, 1), dtype="float32")
        assign_label = np.zeros((self.minibatch_count, num_bboxes, num_classes), dtype="float32")

        lambda_bbox = np.ones((self.minibatch_count, num_bboxes, 1), dtype="float32") * 0.1
        lambda_conf = np.ones((self.minibatch_count, num_bboxes, 1), dtype="float32") * 0.1
        lambda_prob = np.zeros((self.minibatch_count, num_bboxes, 1), dtype="float32")

        output_model = model.eval({model.arguments[0]: assign_image})

        """ Dynamic Target Assignment """
        for N in range(self.minibatch_count):
            target_bboxes = np.repeat(target_bbox[N][np.newaxis, :, :], num_bboxes, axis=0).transpose(0, 2, 1)
            target_labels = target_label[N]

            iou_score = intersection_over_union(output_model[N, :, :4].reshape(num_bboxes, 4, 1), target_bboxes)

            row_iou, col_iou = np.where(iou_score >= 0.5)

            predict_bboxes_set, target_bboxes_set = set(), set()
            num_targets = target_bboxes.shape[-1]
            num_count = 0
            max_count = num_bboxes * num_targets
            while len(target_bboxes_set) < num_targets:
                if num_count == max_count:
                    print("Couldn't assign all boxes!")
                    break

                row, col = np.where(iou_score == iou_score.max())
                row, col = row[0], col[0]

                if row in predict_bboxes_set or col in target_bboxes_set:  # no duplicattion
                    iou_score[row, col] = 0
                    num_count += 1
                    continue
                else:
                    assign_bbox[N, row, :] = target_bbox[N][col]
                    assign_conf[N, row] = 1
                    assign_label[N, row] = target_labels[col]
                    lambda_bbox[N, row] = 1.0
                    lambda_conf[N, row] = 1.0
                    lambda_prob[N, row] = 1.0

                    iou_score[row, col] = 0
                    num_count += 1

                    predict_bboxes_set.add(row)
                    target_bboxes_set.add(col)
            #
            # IoU is larger than 0.5
            #
            for row, col in zip(row_iou, col_iou):
                if row in predict_bboxes_set:
                    continue
                else:
                    assign_bbox[N, row, :] = target_bbox[N][col]
                    assign_conf[N, row] = 1
                    assign_label[N, row] = target_labels[col]
                    lambda_bbox[N, row] = 1.0
                    lambda_conf[N, row] = 1.0
                    lambda_prob[N, row] = 1.0

        self.sample_count += self.minibatch_count

        return assign_image, assign_bbox, assign_conf, assign_label, lambda_bbox, lambda_conf, lambda_prob

    
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


def create_reader(map_file, box_file, train):
    transforms = [xforms.color(brightness_radius=0.2, contrast_radius=0.2, saturation_radius=0.2),
                  xforms.scale(width=img_width, height=img_height, channels=img_channel, interpolations="linear")]
    image_source = C.io.ImageDeserializer(map_file, C.io.StreamDefs(
        images=C.io.StreamDef(field="image", transforms=transforms), dummy=C.io.StreamDef(field="label", shape=1)))
    bbox_source = C.io.CTFDeserializer(box_file, C.io.StreamDefs(
        bboxes=C.io.StreamDef(field="bbox", shape=4, is_sparse=False),
        labels=C.io.StreamDef(field="label", shape=num_classes, is_sparse=True)))
    return C.io.MinibatchSource([image_source, bbox_source],
                                randomize=train, max_sweeps=C.io.INFINITELY_REPEAT if train else 1)


def center_offset(num_boxes, num_cell):
    cx, cy = [], []
    x_cur, y_cur = -1, -1
    x_div, y_div = num_boxes, num_boxes * num_cell
    for i in range(num_boxes * num_cell * num_cell):
        if i % x_div == 0: x_cur += 1
        if i % y_div == 0: y_cur += 1
        if x_cur == num_cell: x_cur = 0
        cx.append([x_cur])
        cy.append([y_cur])
    return np.ascontiguousarray(np.concatenate((np.asarray(cx), np.asarray(cy)), axis=1), dtype="float32")


def prior_anchor(num_boxes, num_cell, anchor_boxes):
    pwh = []
    for i in range(num_boxes):
        pwh.append([anchor_boxes[i][0], anchor_boxes[i][1]])
    return np.ascontiguousarray(np.asarray(pwh * num_cell * num_cell), dtype="float32")


def generalized_iou_loss(outputs, targets):
    x1p, y1p = outputs[:, 0] - 0.5 * outputs[:, 2], outputs[:, 1] - 0.5 * outputs[:, 3]
    x2p, y2p = outputs[:, 0] + 0.5 * outputs[:, 2], outputs[:, 1] + 0.5 * outputs[:, 3]
    x1g, y1g = targets[:, 0] - 0.5 * targets[:, 2], targets[:, 1] - 0.5 * targets[:, 3]
    x2g, y2g = targets[:, 0] + 0.5 * targets[:, 2], targets[:, 1] + 0.5 * targets[:, 3]

    x1p_hat, x2p_hat = C.element_max(C.element_min(x1p, x2p), 0), C.element_max(x1p, x2p)
    y1p_hat, y2p_hat = C.element_max(C.element_min(y1p, y2p), 0), C.element_max(y1p, y2p)

    Ag = (x2g - x1g) * (y2g - y1g)
    Ap = (x2p_hat - x1p_hat) * (y2p_hat - y1p_hat)

    x1L, x2L = C.element_max(x1p_hat, x1g), C.element_min(x2p_hat, x2g)
    y1L, y2L = C.element_max(y1p_hat, y1g), C.element_min(y2p_hat, y2g)

    intersection = C.element_max(x2L - x1L, 0) * C.element_max(y2L - y1L, 0)
    union = Ap + Ag - intersection

    x1c, x2c = C.element_min(x1p_hat, x1g), C.element_max(x2p_hat, x2g)
    y1c, y2c = C.element_min(y1p_hat, y1g), C.element_max(y2p_hat, y2g)

    Ac = C.element_max(x2c - x1c, 0) * C.element_max(y2c - y1c, 0)

    return 1 - ((intersection / union) - (Ac - union) / Ac)


def intersection_over_union(outputs, targets):  # [x-center, y-center, width, height]
    x1p, y1p, x2p, y2p = outputs[:, 0] - 0.5 * outputs[:, 2], outputs[:, 1] - 0.5 * outputs[:, 3], \
                         outputs[:, 0] + 0.5 * outputs[:, 2], outputs[:, 1] + 0.5 * outputs[:, 3]
    x1g, y1g, x2g, y2g = targets[:, 0] - 0.5 * targets[:, 2], targets[:, 1] - 0.5 * targets[:, 3], \
                         targets[:, 0] + 0.5 * targets[:, 2], targets[:, 1] + 0.5 * targets[:, 3]

    x1p_hat, x2p_hat = np.maximum(np.minimum(x1p, x2p), 0), np.maximum(x1p, x2p)
    y1p_hat, y2p_hat = np.maximum(np.minimum(y1p, y2p), 0), np.maximum(y1p, y2p)

    Ag = (x2g - x1g) * (y2g - y1g)
    Ap = (x2p_hat - x1p_hat) * (y2p_hat - y1p_hat)

    x1L, x2L = np.maximum(x1p_hat, x1g), np.minimum(x2p_hat, x2g)
    y1L, y2L = np.maximum(y1p_hat, y1g), np.minimum(y2p_hat, y2g)

    intersection = np.maximum(x2L - x1L, 0) * np.maximum(y2L - y1L, 0)
    union = Ap + Ag - intersection

    return intersection / union


def ssmd416(h, layers={}, filename="../COCO/coco21.h5"):
        with h5py.File("../COCO/coco21.h5", "r") as f:
        for l in range(20):
            h = convolution(f["params/conv%d/weights" % (l + 1)][()])(h)
            h = C.elu(h)
            h = batch_normalization(f["params/bn%d/scale" % (l + 1)][()], f["params/bn%d/bias" % (l + 1)][()],
                                    f["params/bn%d/mean" % (l + 1)][()], f["params/bn%d/variance" % (l + 1)][()])(h)
            if l in [1, 3, 6, 9, 14]:
                layers["layer%d" % (l + 1)] = h
                h = C.layers.MaxPooling((3, 3), strides=2, pad=True)(h)

    with C.layers.default_options(activation=C.elu, init=C.he_normal(), pad=True, strides=1, bias=False,
                                  map_rank=1, use_cntk_engine=True):
        h_small = Convolution2D((3, 3), 512)(layers["layer15"])  # 512 x 26 x26
        h_small = BatchNormalization()(h_small)
        h_small = Convolution2D((1, 1), num_channel, activation=None, bias=True, init=C.glorot_uniform())(h_small)
        h_small = C.reshape(C.transpose(h_small, (1, 2, 0)), (1 * 26 * 26, num_channel))

        h_medium = Convolution2D((3, 3), 1024)(h)  # 1024 x 13 x 13
        h_medium = BatchNormalization()(h_medium)
        h_medium = Convolution2D((1, 1), 3 * num_channel, activation=None, bias=True, init=C.glorot_uniform())(h_medium)
        h_medium = C.reshape(C.transpose(h_medium, (1, 2, 0)), (3 * 13 * 13, num_channel))

        h_large = Convolution2D((3, 3), 512, strides=2)(h)  # 512 x 7 x 7
        h_large = BatchNormalization()(h_large)
        h_large = Convolution2D((1, 1), 3 * num_channel, activation=None, bias=True, init=C.glorot_uniform())(h_large)
        h_large = C.reshape(C.transpose(h_large, (1, 2, 0)), (3 * 7 * 7, num_channel))

        h_xy = C.splice((C.sigmoid(h_large[:, :2]) + C.constant(center_offset(3, 7))) / 7,  # x-center, y-center
                        (C.sigmoid(h_medium[:, :2]) + C.constant(center_offset(3, 13))) / 13,
                        (C.sigmoid(h_small[:, :2]) + C.constant(center_offset(1, 26))) / 26, axis=0)
        h_wh = C.splice(C.constant(prior_anchor(3, 7, anchor_boxes[2:])) * C.softplus(h_large[:, 2:4]),  # width, height
                        C.constant(prior_anchor(3, 13, anchor_boxes[1:-1])) * C.softplus(h_medium[:, 2:4]),
                        C.constant(prior_anchor(1, 26, anchor_boxes[:1, :])) * C.softplus(h_small[:, 2:4]), axis=0)
        h_conf = C.sigmoid(C.splice(h_large[:, 4], h_medium[:, 4], h_small[:, 4], axis=0))  # objectness
        h_prob = C.splice(h_large[:, 5:], h_medium[:, 5:], h_small[:, 5:], axis=0)

    return C.splice(h_xy, h_wh, h_conf, h_prob, axis=1)


if __name__ == "__main__":
    #
    # minibatch reader
    #
    train_reader = SingleShotMultiDetector("./train2014_ssmd_images.txt", "./train2014_ssmd_bboxes.txt", is_train=True)

    
    #
    # input, bounding-box, confidence, label, and model
    #
    input = C.input_variable(shape=(img_channel, img_height, img_width), dtype="float32")
    bbox = C.input_variable(shape=(num_bboxes, 4), dtype="float32")
    conf = C.input_variable(shape=(num_bboxes, 1), dtype="float32")
    label = C.input_variable(shape=(num_bboxes, num_classes), dtype="float32")
    lambda_bbox = C.input_variable(shape=(num_bboxes, 1), dtype="float32")
    lambda_conf = C.input_variable(shape=(num_bboxes, 1), dtype="float32")
    lambda_prob = C.input_variable(shape=(num_bboxes, 1), dtype="float32")

    model = ssmd416(input / 255.0)

    #
    # loss function
    #
    giou_loss = generalized_iou_loss(model[:, :4], bbox) * lambda_bbox
    conf_loss = C.binary_cross_entropy(model[:, 4], conf) * lambda_conf
    prob_loss = C.cross_entropy_with_softmax(model[:, 5:], label, axis=1) * lambda_prob
    
    loss = giou_loss + prob_loss + conf_loss
    errs = C.classification_error(model[:, 5:], label, axis=1)
    
    #
    # optimizer and cyclical learning rate
    #
    learner = C.adam(model.parameters, lr=1e-3, momentum=0.9, gradient_clipping_threshold_per_sample=sample_size,
                     gradient_clipping_with_truncation=True)
    clr = CyclicalLearningRate(learner, base_lrs=1e-5, max_lrs=1e-3, minibatch_size=sample_size, step_size=step_size)
    progress_printer = C.logging.ProgressPrinter(tag="Training")

    trainer = C.Trainer(model, (loss, errs), [learner], [progress_printer])

    C.logging.log_number_of_parameters(model)

    #
    # training
    #
    plot_data = {"epoch": [], "giou_loss": [], "conf_loss": [], "prob_loss": []}
    for epoch in range(epoch_size):
        sample_count = 0
        epoch_giou = 0
        epoch_conf = 0
        epoch_prob = 0
        while sample_count < num_samples:
            batch_image, batch_bbox, batch_conf, batch_label, batch_bbox_lambda, batch_conf_lambda, batch_prob_lambda =\
                train_reader.next_minibatch(min(minibatch_size, num_samples - sample_count), model)

            trainer.train_minibatch({input: batch_image, bbox: batch_bbox, conf: batch_conf, label: batch_label,
                                     lambda_bbox: batch_bbox_lambda, lambda_conf: batch_conf_lambda,
                                     lambda_prob: batch_prob_lambda})

            clr.batch_step()

            sample_count += train_reader.minibatch_count
            epoch_giou += giou_loss.eval({input: batch_image, bbox: batch_bbox, lambda_bbox: batch_bbox_lambda}).sum()
            epoch_conf += conf_loss.eval({input: batch_image, conf: batch_conf, lambda_conf: batch_conf_lambda}).sum()
            epoch_prob += prob_loss.eval({input: batch_image, label: batch_label, lambda_prob: batch_prob_lambda}).sum()

        #
        # giou, conf, and prob logging
        #
        plot_data["epoch"].append(epoch + 1)
        plot_data["giou_loss"].append(epoch_giou / num_samples)
        plot_data["conf_loss"].append(epoch_conf / num_samples)
        plot_data["prob_loss"].append(epoch_prob / num_samples)

        trainer.summarize_training_progress()

        train_reader.sample_count = 0

    #
    # save model and logging
    #
    model.save("./ssmd.model")
    print("Saved model.")

    df = pd.DataFrame(plot_data)
    df.to_csv("./ssmd.csv", index=False)
    print("Saved logging.")
    
