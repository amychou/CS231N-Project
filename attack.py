import argparse
import colorsys
import imghdr
import os
import random
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

from yad2k.models.keras_yolo import yolo_eval, yolo_head, yolo_eval_adv, yolo_loss, preprocess_true_boxes
from retrain_yolo import get_detector_mask

parser = argparse.ArgumentParser(
    description='Run a YOLO_v2 style detection model on test images..')
parser.add_argument(
    'model_path',
    help='path to h5 model file containing body'
    'of a YOLO_v2 model')
parser.add_argument(
    '-a',
    '--anchors_path',
    help='path to anchors file, defaults to yolo_anchors.txt',
    default='model_data/yolo_anchors.txt')
parser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to coco_classes.txt',
    default='model_data/coco_classes.txt')
parser.add_argument(
    '-t',
    '--test_path',
    help='path to directory of test images, defaults to images/',
    default='images')
parser.add_argument(
    '-s',
    '--score_threshold',
    type=float,
    help='threshold for bounding box scores, default .3',
    default=.3)
parser.add_argument(
    '-iou',
    '--iou_threshold',
    type=float,
    help='threshold for non max suppression IOU, default .5',
    default=.5)
# New arguments
parser.add_argument(
    '-lr',
    '--learning_rate',
    type=float,
    default=1e-3
)
parser.add_argument(
    '-iter',
    '--iterations',
    type=int,
    default=5
)
parser.add_argument(
    '--clip',
    default=False,
    action='store_true'
)
parser.add_argument(
    '-b',
    '--bounding',
    default=False,
    action='store_true'
)
parser.add_argument(
    '-eps',
    '--epsilon',
    type=float,
    default=1e-1
)
parser.add_argument(
    '--sign',
    default=False,
    action='store_true'
)
parser.add_argument(
    '--images',
    type=int,
    default=-1
)
parser.add_argument(
    '--accuracy',
    default=False,
    action='store_true'
)

'''converts numpy array to PIL image'''
def getImage(image_data, original_size):
    image_data_processed = image_data.copy()
    image_data_processed = image_data_processed[0] # remove batch dimension
    image_data_processed *= 256
    im =  Image.fromarray(np.uint8(image_data_processed), 'RGB')
    resized_image = im.resize(original_size, Image.BICUBIC)
    return resized_image

def plot(loss_history, filepath, yax):
    print("Plotting")
    plt.figure()
    plt.plot(loss_history, color='black')
    plt.xlabel('iteration', fontsize=14)
    plt.ylabel(yax, fontsize=14)
    plt.savefig(filepath)
    plt.close()

def visualize(image_data):
    image = getImage(image_data, original_size)

def load_pascal_annotation(filename, class_to_index):
    """
    for a given index, load image and bounding boxes info from XML file
    :param index: index of a specific image
    :return: record['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(filename)
    objs = tree.findall('object')
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1
        cls = class_to_index[obj.find('name').text.lower().strip()]
        boxes[ix, :] = [x1, y1, x2, y2]
        gt_classes[ix] = cls

    return zip(boxes, gt_classes)

def get_iou(ground_truth, top, left, bottom, right, c):
    iou = 0
    for tb, tc in ground_truth:
        if c != tc:
            continue

        x11, y11, x12, y12 = left, top, right, bottom
        x21, y21, x22, y22 = tb[0], tb[1], tb[2], tb[3]

        xA = max(x11, x21)
        yA = max(y11, y21)
        xB = min(x12, x22)
        yB = min(y12, y22)

        interArea = max((xB - xA + 1), 0) * max((yB - yA + 1), 0)
        boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
        boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
        iou = max(iou, interArea / (boxAArea + boxBArea - interArea))
    return iou

def _main(args):
    model_path = os.path.expanduser(args.model_path)
    assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
    anchors_path = os.path.expanduser(args.anchors_path)
    classes_path = os.path.expanduser(args.classes_path)
    test_path = os.path.expanduser(args.test_path)
    eps = args.epsilon
    iters = args.iterations
    images = args.images


    # Customize output path
    output_path = 'images/'
    if (args.bounding):
        output_path += 'bounding_'
    else:
        output_path += 'class_'
    output_path += 'lr'+str(args.learning_rate)
    if (args.clip):
        output_path += 'clip' + str(eps)
    if (args.sign):
        output_path += '_signed'
    output_path = os.path.expanduser(output_path)
    print("Output folder " + output_path)

    if not os.path.exists(output_path):
        print('Creating output path {}'.format(output_path))
        os.mkdir(output_path)

    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    class_to_index = dict((class_names[i], i) for i in range(len(class_names)))

    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

    yolo_model = load_model(model_path) # https://keras.io/models/model/

    # Verify model, anchors, and classes are compatible
    num_classes = len(class_names)
    num_anchors = len(anchors)
    # TODO: Assumes dim ordering is channel last
    model_output_channels = yolo_model.layers[-1].output_shape[-1]
    assert model_output_channels == num_anchors * (num_classes + 5), \
        'Mismatch between model and given anchor and class sizes. ' \
        'Specify matching anchors and classes with --anchors_path and ' \
        '--classes_path flags.'
    print('{} model, anchors, and classes loaded.'.format(model_path))

    # Check if model is fully convolutional, assuming channel last order.
    model_image_size = yolo_model.layers[0].input_shape[1:3]
    is_fixed_size = model_image_size != (None, None)

    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.

    # Generate output tensor targets for filtered bounding boxes.
    # TODO: Wrap these backend operations with Keras layers.
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))

    # Get target scores for boxes, compute gradients
    boxes, scores, classes, target_scores = yolo_eval_adv(
        yolo_outputs,
        input_image_shape,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold)
    correct_scores_loss = K.sum(scores)
    grad_correct, = K.gradients(correct_scores_loss, yolo_model.input)
    target_scores_loss = K.sum(target_scores)
    grad_target, = K.gradients(target_scores_loss, yolo_model.input)

    # Bounding box attack
    target_boxes = np.array([[[0.25, 0.25, 0.25, 0.25, 0]]], dtype=np.float32)
    detectors_mask, matching_true_boxes = get_detector_mask(target_boxes, anchors)
    arg = (yolo_model.output, target_boxes, detectors_mask, matching_true_boxes)
    model_loss = yolo_loss(arg, anchors, len(class_names))
    model_loss_grad, = K.gradients(model_loss, yolo_model.input)

    for img_count, image_file in enumerate(os.listdir(test_path)):
        if images != -1 and img_count > images:
            break

        try:
            image_type = imghdr.what(os.path.join(test_path, image_file))
            if not image_type:
                continue
        except IsADirectoryError:
            continue

        image = Image.open(os.path.join(test_path, image_file))
        original_size = image.size
        if is_fixed_size:  # TODO: When resizing we can use minibatch input.
            resized_image = image.resize(
                tuple(reversed(model_image_size)), Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32')
        else:
            # Due to skip connection + max pooling in YOLO_v2, inputs must have
            # width and height as multiples of 32.
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            resized_image = image.resize(new_image_size, Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32')
            print(image_data.shape)

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        image_data_adv = image_data.copy()

        print("Generating adversarial image")
        loss_history = []
        accuracy_history = []
        for iteration in range(iters):
            print("Iteration " + str(iteration+1))
            if (iteration%10 == 0):
                image_adv = getImage(image_data_adv, original_size)

                print("Saving adversarial image")
                out_boxes, out_scores, out_classes = sess.run(
                    [boxes, scores, classes],
                    feed_dict={
                        yolo_model.input: image_data_adv,
                        input_image_shape: [image.size[1], image.size[0]],
                        K.learning_phase(): 0
                    })
                print('Found {} boxes for {}'.format(len(out_boxes), image_file))

                font = ImageFont.truetype(
                    font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
                thickness = (image.size[0] + image.size[1]) // 300

                for i, c in reversed(list(enumerate(out_classes))):
                    predicted_class = class_names[c]
                    box = out_boxes[i]
                    score = out_scores[i]

                    label = '{} {:.2f}'.format(predicted_class, score)

                    draw = ImageDraw.Draw(image_adv)
                    label_size = draw.textsize(label, font)

                    top, left, bottom, right = box
                    top = max(0, np.floor(top + 0.5).astype('int32'))
                    left = max(0, np.floor(left + 0.5).astype('int32'))
                    bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                    right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

                    if top - label_size[1] >= 0:
                        text_origin = np.array([left, top - label_size[1]])
                    else:
                        text_origin = np.array([left, top + 1])

                    # My kingdom for a good redistributable image drawing library.
                    for i in range(thickness):
                        draw.rectangle(
                            [left + i, top + i, right - i, bottom - i],
                            outline=colors[c])
                    draw.rectangle(
                        [tuple(text_origin), tuple(text_origin + label_size)],
                        fill=colors[c])
                    draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                    del draw

                if (args.accuracy and iteration > 0):
                    bounding_file = image_file[:-3] + "xml"
                    ground_truth = list(load_pascal_annotation(os.path.join(test_path, bounding_file), class_to_index))
                    iou_per_box = []

                    # Average of max IOU for each bounding box
                    for i, c in reversed(list(enumerate(out_classes))):
                        # Predicted bounding box
                        predicted_class = class_names[c]
                        box = out_boxes[i]
                        score = out_scores[i]

                        top, left, bottom, right = box
                        top = max(0, np.floor(top + 0.5).astype('int32'))
                        left = max(0, np.floor(left + 0.5).astype('int32'))
                        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

                        # Calculating IOU
                        iou = get_iou(ground_truth, top, left, bottom, right, c)
                        iou_per_box.append(iou)

                    accuracy_history.append(np.mean(iou_per_box))
                    print("Current Accuracy:", iou_per_box)

                name, ext = os.path.splitext(image_file)
                image_name = name + "_iter" + str(iteration) + ext
                image_adv.save(os.path.join(output_path, image_name), quality=90)
                if (args.bounding):
                    plot(loss_history, os.path.join(output_path, 'loss_' + name + "_iter" + str(iteration) + '.png'), "loss")
                if (args.accuracy):
                    plot(accuracy_history, os.path.join(output_path, 'accuracy_' + name + "_iter" + str(iteration) + '.png'), "accuracy")
                negative_image = getImage(image_data_adv-image_data, original_size)
                negative_image = getImage(image_data_adv-image_data, original_size)
                negative_image.save(os.path.join(output_path, 'neg_' + image_name), quality=90)

            if (args.bounding):
                loss, grad, = sess.run(
                    [model_loss, model_loss_grad],
                    feed_dict={
                        yolo_model.input: image_data_adv,
                        input_image_shape: [image.size[1], image.size[0]],
                        K.learning_phase(): 0
                    }
                )
                loss_history.append(loss)
                print("Loss " + str(loss))
            else:
                grad, = sess.run(
                    [grad_target],
                    feed_dict={
                        yolo_model.input: image_data,
                        input_image_shape: [image.size[1], image.size[0]],
                        K.learning_phase(): 0
                    }
                )
                grad = -grad
            if (args.sign):
                r = -args.learning_rate * np.sign(grad)
            else:
                r = -args.learning_rate * grad
                r /= (np.max(np.abs(grad)) + 1e-7)
            if (args.clip):
                image_data_adv = np.clip(image_data_adv+r, image_data-eps, image_data+eps)
            else:
                image_data_adv = image_data_adv+r
    sess.close()


if __name__ == '__main__':
    _main(parser.parse_args())
