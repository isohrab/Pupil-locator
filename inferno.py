import argparse
import os
import time
from PIL import Image


import cv2
import numpy as np
import tensorflow.compat.v1 as tf

from config import config
from logger import Logger
from models import Simple, NASNET, Inception, GAP, YOLO
from utils import annotator, change_channel, gray_normalizer

tf.disable_v2_behavior()


def load_model(session, m_type, m_name, logger):
    # load the weights based on best loss
    best_dir = "best_loss"

    # check model dir
    model_path = "models/" + m_name
    path = os.path.join(model_path, best_dir)
    if not os.path.exists(path):
        raise FileNotFoundError

    if m_type == "simple":
        model = Simple(m_name, config, logger)
    elif m_type == "YOLO":
        model = YOLO(m_name, config, logger)
    elif m_type == "GAP":
        model = GAP(m_name, config, logger)
    elif m_type == "NAS":
        model = NASNET(m_name, config, logger)
    elif m_type == "INC":
        model = Inception(m_name, config, logger)
    else:
        raise ValueError

    # load the best saved weights
    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logger.log('Reloading model parameters..')
        model.restore(session, ckpt.model_checkpoint_path)

    else:
        raise ValueError('There is no best model with given model')

    return model


def rescale(image):
    """
    If the input video is other than network size, it will resize the input video
    :param image: a frame form input video
    :return: scaled down frame
    """
    scale_side = max(image.shape)
    # image width and height are equal to 192
    scale_value = config["input_width"] / scale_side

    # scale down or up the input image
    scaled_image = cv2.resize(image, dsize=None, fx=scale_value, fy=scale_value)

    # convert to numpy array
    scaled_image = np.asarray(scaled_image, dtype=np.uint8)

    # one of pad should be zero
    w_pad = int((config["input_width"] - scaled_image.shape[1]) / 2)
    h_pad = int((config["input_width"] - scaled_image.shape[0]) / 2)

    # create a new image with size of: (config["image_width"], config["image_height"])
    new_image = np.ones((config["input_width"], config["input_height"]), dtype=np.uint8) * 250

    # put the scaled image in the middle of new image
    new_image[h_pad:h_pad + scaled_image.shape[0], w_pad:w_pad + scaled_image.shape[1]] = scaled_image

    return new_image


def upscale_preds(_preds, _shapes):
    """
    Get the predictions and upscale them to original size of video
    :param preds:
    :param shapes:
    :return: upscales x and y
    """
    # we need to calculate the pads to remove them from predicted labels
    pad_side = np.max(_shapes)
    # image width and height are equal to 384
    downscale_value = config["input_width"] / pad_side

    scaled_height = _shapes[0] * downscale_value
    scaled_width = _shapes[1] * downscale_value

    # one of pad should be zero
    w_pad = (config["input_width"] - scaled_width) / 2
    h_pad = (config["input_width"] - scaled_height) / 2

    # remove the pas from predicted label
    x = _preds[0] - w_pad
    y = _preds[1] - h_pad
    w = _preds[2]

    # calculate the upscale value
    upscale_value = pad_side / config["input_height"]

    # upscale preds
    x = x * upscale_value
    y = y * upscale_value
    w = w * upscale_value

    return x, y, w


# load a the model with the best saved state from file and predict the pupil location
# on the input video. finaly save the video with the predicted pupil on disk
def main(m_type, m_name, logger, video_path=None, write_output=True):
    with tf.Session() as sess:

        # load best model
        model = load_model(sess, m_type, m_name, logger)
        dir_list = os.listdir(video_path)
        print(video_path)
        print(dir_list)
        arr = video_path.split('/')
        arr = arr[:len(arr)-1]
        arr.append("output_dir")
        output_path = '/'.join(arr)
        try:
            os.mkdir(output_path)
        except:
            pass
        for file in dir_list:
            image_file = video_path + '/' + file
            output_dir = output_path + '/' + file
            preds = []
            # load the video or camera
            frame = cv2.imread(image_file)
            ori_image = cv2.imread(image_file)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            f_shape = frame.shape
            if frame.shape[0] != 192:
                frame = rescale(frame)

            image = gray_normalizer(frame)
            image = change_channel(image, config["input_channel"])

            [p] = model.predict(sess, [image])
            x, y, w = upscale_preds(p, f_shape)

            preds.append([x, y, w])

            labeled_img = annotator((0, 250, 250), ori_image, *preds[0])
            save_img = Image.fromarray(ori_image, 'RGB')
            save_img.save(output_dir)


if __name__ == "__main__":
    class_ = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=class_)

    parser.add_argument('--model_type',
                        help="INC, YOLO, simple",
                        default="INC")

    parser.add_argument('--model_name',
                        help="name of saved model (3A4Bh-Ref25)",
                        default="3A4Bh-Ref25")

    parser.add_argument('video_path',
                        help="path to video file, empty for camera")

    args = parser.parse_args()

    # model_name = args.model_name
    model_name = args.model_name
    model_type = args.model_type
    video_path = args.video_path

    # initial a logger
    logger = Logger(model_type, model_name, "", config, dir="models/")
    logger.log("Start inferring model...")

    main(model_type, model_name, logger, video_path)
