import os
import tensorflow as tf
import time
import argparse
import glob
import cv2
from tqdm import tqdm
import numpy as np
from config import config
import utils as u
from Logger import Logger
from models import Simple, NASNET, Inception, GAP, YOLO


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
    w_pad = int((config["image_width"] - scaled_image.shape[1]) / 2)
    h_pad = int((config["image_width"] - scaled_image.shape[0]) / 2)

    # add half of the pad to the label (x, y)
    label[0] += w_pad
    label[1] += h_pad

    # create a new image with size of: (config["image_width"], config["image_height"])
    new_image = np.zeros((config["image_width"], config["image_height"]), dtype=np.uint8)

    # put the scaled image in the middle of new image
    new_image[h_pad:h_pad + scaled_image.shape[0], w_pad:w_pad + scaled_image.shape[1]] = scaled_image

    return new_image, label


def real_image_name(img_name):
    """
    get the image name from CSV file and add zero pad before the file name
    convert 12345 to 0000012345
    """
    diff = 10 - len(img_name)
    pad = ['0' for i in range(diff)]
    return ''.join(pad) + img_name


def get_len(csv_path):
    """
    get a csv value path and return len of data
    :param csv_path:
    :return: len of data
    """
    counter = 0
    with open(csv_path, mode='r') as f:
        # pass the header
        f.readline()
        for line in f:
            counter += 1

    return counter


def read_batch(csv_path, b_size, d_name):
    images = []
    labels = []

    with open(csv_path, mode='r') as f:
        # pass the header
        f.readline()
        for line in f:
            values = line.split(" ")

            # read the image
            image_name = real_image_name(values[1])
            img = cv2.imread("emma_data/{}/{}.png".format(d_name, image_name), cv2.IMREAD_GRAYSCALE)

            # read and convert the labels
            x = int(values[2]) / 2
            y = int(img.shape[0] - int(values[3]) / 2)

            # rescale images to 192x192 pixels
            img, lbl = rescale(img, [x, y])

            # normalize image and label
            img = u.gray_normalizer(img)

            # expand the channel dimension
            img = np.expand_dims(img, -1)

            images.append(img)
            labels.append(lbl)
            if len(images) == b_size:
                yield images, np.asarray(labels, dtype=np.float32)
                images = []
                labels = []

    # yield the rest
    if len(images) != 0:
        yield images, np.asarray(labels, dtype=np.float32)


def video_creator(video_name, images, labels, fps=15):
    """
    get a list of images and their corresponidng labels and create a labeled video
    :param video_name: the output video name
    :param images: test images with shape (192,192,1) which should squeezed before writing the video
    :param labels: predicted labels with a value between 0-1
    :return: None
    """
    video = cv2.VideoWriter(video_name+".avi", cv2.VideoWriter_fourcc(*'XVID'), fps, (192, 192))

    for img, lbl in zip(images, labels):
        img = np.squeeze(img)
        img = u.gray_denormalizer(img)
        annotated_img = u.annotator(img, *lbl)
        video.write(annotated_img)

    cv2.destroyAllWindows()
    video.release()
    print("{} video has been created successfully".format(video_name))


def print_resutls(errors_dic, pixels_list):
    # sort the dataset errors to have a uniform results
    # generate the header
    header = "Dataset name: "
    for p in pixels_list:
        header += str(p) + "\t\t"

    print(header)

    def row_writer(title, errors_list):
        row = title + ": "
        for val in errors_list:
            row += " {:2.2f}\t".format(val*100)

        return row

    for key in sorted(errors_dic.keys()):
        print(row_writer(key, errors_dic[key]))

    # print average error
    errors = [val for key, val in errors_dic.items()]
    errors = np.asarray(errors, dtype=np.float32)
    avg = np.mean(errors, axis=0)
    print(row_writer("average error", avg))


def main(m_type, m_name, logger):
    with tf.Session() as sess:

        # load best model
        model = load_model(sess, m_type, m_name, logger)

        # get the csv files
        datasets = glob.glob('emma_data/*.txt')
        datasets = sorted(datasets)

        # we save the results of all dataset in to this list
        dataset_results = {}

        for d in datasets:

            # get the name of dataset from the path
            dataset_name = d.split("/")[1].split(".")[0]

            # save the result (differences) in the list
            dataset_results[dataset_name] = []

            dataset_len = get_len(d)

            batch_size = 64
            batch = read_batch(d, batch_size, dataset_name)

            # use tqdm progress bar
            tqdm_len = np.ceil(dataset_len / batch_size)
            with tqdm(total=tqdm_len, unit='batch') as t:
                # set the name of dataset as the title of progress bar
                t.set_description_str(dataset_name)

                test_images = []
                pred_labels = []

                # loop over batch of images
                for images, truths in batch:
                    predictions = model.predict(sess, images)

                    # get the image w/h for denormalizing the labels
                    w = images[0].shape[1]
                    h = images[0].shape[0]

                    # calculate the difference
                    a = predictions[:, 0] - truths[:, 0]
                    b = predictions[:, 1] - truths[:, 1]

                    diff = np.sqrt((a * a + b * b))

                    dataset_results[dataset_name].extend(diff)
                    t.update()

                    # add images and predicted labels to test_images and pred_labels to creating the video
                    test_images.extend(images)
                    pred_labels.extend(predictions)

                # create the predicted labels on test sets
                # video_creator(dataset_name, test_images, pred_labels)

    # print the result for different pixel error
    pixel_errors = [1, 2, 3, 4, 5, 7, 10, 15, 20]

    # save the results in a dic
    dataset_errors = {}

    for key, val in dataset_results.items():
        dataset_errors[key] = []
        for e in pixel_errors:
            d = np.asarray(val, dtype=np.float32)
            acc = np.mean(np.asarray(d < e, dtype=np.int))
            dataset_errors[key].append(acc)

    print_resutls(dataset_errors, pixel_errors)


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
    model_name = "Inception_test"
    model_type = args.model_type
    model_type = "INC"
    video_input = args.video_input

    logger = Logger(model_type, model_name, "", config, dir="models/")
    logger.log("Start inferring model...")

    main(model_type, model_name, logger)

