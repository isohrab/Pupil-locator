import os
import tensorflow as tf
import time
import argparse
import glob
import cv2
from tqdm import tqdm
import numpy as np
from config import config
from utils import anotator, change_channel, create_noisy_video
from Logger import Logger
from models import Simple, NASNET, Inception, GAP, YOLO
from augmentor import Augmentor


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

    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logger.log('Reloading model parameters..')
        model.restore(session, ckpt.model_checkpoint_path)

    else:
        raise ValueError('There is no best model with given model')

    return model


def read_csv(csv_path, img_shape):
    """
    read the CSV file and return the location array
    Also we do:
    x= txt_x/2
    y= image_size_y - (txt_y/2)
    :param csv_path:
    :return: return a nx2 numpy array float32
    """
    truth = []
    with open(csv_path, mode='r') as f:
        # read the header
        _ = f.readline()
        for line in f:
            values = line.split(' ')
            x = int(values[2]) / 2
            y = int(img_shape[0] - int(values[3]) / 2)
            truth.append([x, y])

    return np.asarray(truth, dtype=np.float32)


def rescale(image, label):
    scale_side = max(image.shape)
    # image width and height are equal to 192
    scale_value = config["image_width"] / scale_side

    # scale down or up the input image
    scaled_image = cv2.resize(image, dsize=None, fx=scale_value, fy=scale_value, interpolation=cv2.INTER_AREA)

    # convert to numpy array
    scaled_image = np.asarray(scaled_image, dtype=np.uint8)

    # rescale the label too
    label[0] = label[0] * scale_value
    label[1] = label[1] * scale_value

    # one of pad should be zero
    w_pad = int((scale_side - image.shape[1]) / 2)
    h_pad = int((scale_side - image.shape[0]) / 2)

    # add half of the pad to the label (x, y)
    label[0] += w_pad
    label[1] += h_pad

    # create a new image with size of: (config["image_width"], config["image_height"])
    new_image = np.zeros((config["image_width"], config["image_height"]), dtype=np.uint8)

    # put the scaled image in the middle of new image
    new_image[h_pad:h_pad + scaled_image.shape[0], w_pad:w_pad + scaled_image.shape[1]] = scaled_image

    return new_image, label


def main(m_type, m_name, logger, write_latex=True):
    with tf.Session() as sess:

        # load best model
        model = load_model(sess, m_type, m_name, logger)

        # get the path of datasets:
        globs = glob.glob('emma_data/*')
        globs = sorted(globs)
        dataset_folders = [d for i, d in enumerate(globs) if i % 2 == 0]
        dataset_csvs = [d for i, d in enumerate(globs) if i % 2 == 1]

        # we save the results of all dataset in to this list
        dataset_results = {}

        for f, f_path in enumerate(dataset_folders):

            # save the result (differences) in the list
            dataset_results[f_path] = []

            # dataset_name = dataset_names[index].split("/")[1]
            dataset_images = glob.glob("{}/*.png".format(f_path))

            # sort the images to match with the labels in csv
            dataset_images = sorted(dataset_images)

            # Just read the first image size for Dikablis converting!!!
            img_shape = cv2.imread(dataset_images[0], cv2.IMREAD_GRAYSCALE).shape
            true_loc = read_csv(dataset_csvs[f], img_shape)

            # loop over the images and feed them to the model
            with tqdm(total=len(dataset_images), unit='image') as t:
                t.set_description_str(f_path.split("/")[1])
                for i, i_path in enumerate(dataset_images):
                    # read the image
                    img = cv2.imread(i_path, cv2.IMREAD_GRAYSCALE)

                    # reshape the image and label respectively
                    img, lbl = rescale(img, true_loc[i])

                    img = np.expand_dims(img, -1)
                    p = model.predict(sess, [img])

                    p = p[0]
                    # calculate the difference
                    a = p[0] - lbl[0]
                    b = p[1] - lbl[1]

                    diff = np.sqrt((a * a + b * b))

                    dataset_results[f_path].append(diff)
                    t.update()


    # print the result for different pixel error
    pixel_errors = [1, 2, 3, 4, 5, 7, 10, 15, 20]

    # save the results in a dic
    # dataset_errors = {}

    for e in pixel_errors:
        for key, val in dataset_results.items():
            d = np.asarray(val, dtype=np.float32)
            acc = np.mean(np.asarray(d < e, dtype=np.int))
            print("{0} with {1} pixel error: {2:2.2f}%".format(key.split('/')[1], e, acc*100))

    print("Done...")


if __name__ == "__main__":
    class_ = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=class_)

    parser.add_argument('--model_type',
                        help="YOLO, simple",
                        default="YOLO")

    parser.add_argument('--model_name',
                        help="name of saved model")

    parser.add_argument('--video_input',
                        help="path to video file, empty for camera",
                        default="0")

    args = parser.parse_args()

    # model_name = args.model_name
    model_name = "YHN_XYW"
    model_type = args.model_type
    model_type = "YOLO"
    video_input = args.video_input

    logger = Logger(model_type, model_name, "", config, dir="models/")
    logger.log("Start inferring model...")

    # create a dummy video
    # ag = Augmentor('noisy_videos', config)
    # video_input = create_noisy_video(length=60, fps=5, augmentor=ag)
    video_input = "test_videos/5.mp4"
    main(model_type, model_name, logger)
