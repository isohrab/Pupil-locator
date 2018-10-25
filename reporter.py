import argparse
import glob
import os

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from config import config
from logger import Logger
from models import Simple, NASNET, Inception, GAP, YOLO
from utils import gray_normalizer, annotator, change_channel

dataset_names = ["data set I",
                 "data set II",
                 "data set III",
                 "data set IV",
                 "data set V",
                 "data set VI",
                 "data set VII",
                 "data set VIII",
                 "data set IX",
                 "data set X",
                 "data set XI",
                 "data set XII",
                 "data set XIII",
                 "data set XIV",
                 "data set XV",
                 "data set XVI",
                 "data set XVII",
                 "data set XVIII",
                 "data set XIX",
                 "data set XX",
                 "data set XXI",
                 "data set XXII",
                 "data set XXIII",
                 "data set XXIV",
                 "PupilNet I",
                 "PupilNet II",
                 "PupilNet III",
                 "PupilNet IV",
                 "PupilNet V"]


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


def swirski_reader(batch_size=64, normalize_image=True):
    # get trials
    trials = sorted(glob.glob("data/swirski/*"))

    # loop over the trials and read the pupil-ellipses.txt files
    for path in trials:
        print("reading and predicting {}".format(path))
        txt_path = path + "/pupil-ellipses.txt"
        dataset_name = path.split("/")[2]

        # loop over lines and read the labels and yield with corresponding images
        imgs_batch = []
        lbls_batch = []
        shapes = []
        with open(txt_path, mode='r') as f:
            for line in f:
                line = line.strip()
                (img_id, vals) = line.split(" | ")
                vals = vals.split(" ")
                x = float(vals[0])
                y = float(vals[1])

                # create image path
                img_path = "{0}/frames/{1}-eye.png".format(path, img_id)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                shapes.append(img.shape)

                # resize the input to model input size
                if img.shape != (config["input_height"], config["input_width"]):
                    img = rescale(img)

                if normalize_image:
                    img = gray_normalizer(img)

                # expand dimension
                img = change_channel(img)

                imgs_batch.append(img)
                lbls_batch.append([x, y])

                if len(imgs_batch) == batch_size:
                    yield imgs_batch, np.asarray(lbls_batch, dtype=np.float32),\
                          dataset_name, np.asarray(shapes, dtype=np.float32)
                    imgs_batch = []
                    lbls_batch = []
                    shapes = []

            # yield the rest
            if len(imgs_batch) > 0:
                yield imgs_batch, np.asarray(lbls_batch, dtype=np.float32),\
                      dataset_name, np.asarray(shapes, dtype=np.float32)


def lpw_reader(batch_size=64, normalize_image=True):
    """
    read LPW dataset.
    Yield: images, labels pairs + trial name (for naming porpuse)
    :return:
    """
    LPW_subject = glob.glob('data/LPW/*')
    LPW_subject = sorted(LPW_subject)

    # get all trial path
    trials_path = []
    for subj in LPW_subject:
        # get the video files
        avi_paths = glob.glob(subj + "/*.avi")
        trials = [p.split(".")[0] for p in avi_paths]
        trials_path.extend(sorted(trials))

    # loop over all trials and yield img + lbls
    for trial in trials_path:
        print("reading and predicting {}...".format(trial))
        avi_path = trial + ".avi"
        txt_path = trial + ".txt"
        f = open(txt_path, mode="r")
        cap = cv2.VideoCapture(avi_path)
        ret = True
        img_batch = []
        lbl_batch = []
        shapes = []
        while ret:
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                line = f.readline().strip()
                vals = line.split(" ")

                shapes.append(frame.shape)

                x = float(vals[0])
                y = float(vals[1])
                if frame.shape != (config["input_height"], config["input_width"]):
                    img = rescale(frame)

                if normalize_image:
                    img = gray_normalizer(img)

                img = change_channel(img)
                img_batch.append(img)
                lbl_batch.append([x, y])
                if len(img_batch) == batch_size:
                    yield img_batch,\
                          np.asarray(lbl_batch, dtype=np.float32),\
                          trial,\
                          np.asarray(shapes, dtype=np.float32)
                    img_batch = []
                    lbl_batch = []
                    shapes = []

        # yield the rest
        if len(img_batch) > 0:
            yield img_batch, np.asarray(lbl_batch, dtype=np.float32), trial, np.asarray(shapes, dtype=np.float32)

        # close file
        f.close()

        # close cv2.cap
        cap.release()
        cv2.destroyAllWindows()


def rescale(image):
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
    shapes = []
    pngs = []
    with open(csv_path, mode='r') as f:
        # pass the header
        f.readline()
        for line in f:
            values = line.split(" ")

            # read the image
            image_name = real_image_name(values[1])
            png = cv2.imread("data/emma_data/{}/{}.png".format(d_name, image_name), cv2.IMREAD_GRAYSCALE)

            shapes.append(png.shape)
            # read and convert the labels
            x = int(values[2]) / 2
            y = int(png.shape[0] - int(values[3]) / 2)

            # rescale images to 192x192 pixels
            img = rescale(png)

            # normalize image and label
            img = gray_normalizer(img)

            # expand the channel dimension
            img = np.expand_dims(img, -1)

            images.append(img)
            labels.append([x, y])
            pngs.append(png)
            if len(images) == b_size:
                yield images, np.asarray(labels, dtype=np.float32), np.asarray(shapes, dtype=np.float32), pngs
                images = []
                labels = []
                shapes = []
                pngs = []

    # yield the rest
    if len(images) != 0:
        yield images, np.asarray(labels, dtype=np.float32), np.asarray(shapes, dtype=np.float32), pngs


def video_creator(video_name, images, labels, fps=15):
    """
    get a list of images and their corresponidng labels and create a labeled video
    :param video_name: the output video name
    :param images: test images with shape (192,192,1) which should squeezed before writing the video
    :param labels: predicted labels with a value between 0-1
    :return: None
    """
    f_size = images[0].shape
    video = cv2.VideoWriter(video_name + " pred.mp4", cv2.VideoWriter_fourcc(*'MP4V'), fps, (f_size[1], f_size[0]))

    for img, lbl in zip(images, labels):
        img = np.squeeze(img)
        # img = gray_denormalizer(img)
        annotated_img = annotator((0, 250, 0), img, *lbl)
        video.write(annotated_img)

    cv2.destroyAllWindows()
    video.release()
    print("{} video has been created successfully".format(video_name))


def print_resutls(errors_dic, pixels_list, d_names=None):
    # sort the dataset errors to have a uniform results
    # generate the header
    header = "Dataset name: \t"
    for p in pixels_list:
        header += str(p) + "\t"

    print(header)

    def row_writer(title, errors_list):
        row = title + ": \t"
        for val in errors_list:
            row += " {:2.2f}\t".format(val * 100)

        return row

    if d_names is None:
        d_names = sorted(errors_dic.keys())

    for name in d_names:
        print(row_writer(name, errors_dic[name]))

    # print average error
    errors = [val for key, val in errors_dic.items()]
    errors = np.asarray(errors, dtype=np.float32)
    avg = np.mean(errors, axis=0)
    print(row_writer("average error", avg))


def real_image_name(img_name):
    """
    get the image name from CSV file and add zero pad before the file name
    convert 12345 to 0000012345
    """
    diff = 10 - len(img_name)
    pad = ['0' for i in range(diff)]
    return ''.join(pad) + img_name

def upscale_preds(_preds, _shapes):
    """
    Get the predictions and upscale them to original size of dataset
    :param preds:
    :param shapes:
    :return: upscales x and y
    """
    # we need to calculate the pads to remove them from predicted labels
    scale_side = np.max(_shapes, axis=1)
    # image width and height are equal to 384
    scale_value = config["input_width"] / scale_side

    scaled_height = _shapes[:, 0] * scale_value
    scaled_width = _shapes[:, 1] * scale_value

    # one of pad should be zero
    w_pad = (config["input_width"] - scaled_width) / 2
    h_pad = (config["input_width"] - scaled_height) / 2

    # remove the pad
    x = _preds[:, 0] - w_pad
    y = _preds[:, 1] - h_pad

    # get the image w/h for upscaling the predictions
    h_s = np.asarray(_shapes[:, 0] / config["input_height"])
    h_s = np.reshape(h_s, (-1, 1))
    w_s = np.asarray(_shapes[:, 1] / config["input_width"])
    w_s = np.reshape(w_s, (-1, 1))

    s = np.concatenate([h_s, w_s], axis=1)
    max_s = np.max(s, axis=1)

    x = x * max_s
    y = y * max_s
    w = _preds[:, 2] * max_s

    return x, y, w


def main(m_type, m_name, logger, save_videos=False):
    """
    run an evaluation on the Test datasets: ExCuSe, ElSe, PupilNet, Swirski, LPW
    :param m_type: need model type: inception, yolo, gap,...
    :param m_name: name of the model ( model folder name: 3A4Bh-Ref25)
    :param logger: need logger to log the events
    :return: show the results in terminal
    """
    run_meta = tf.RunMetadata()
    with tf.Session() as sess:

        # load best model
        model = load_model(sess, m_type, m_name, logger)
        # calculate the FLOPS
        opts_f = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(run_meta=run_meta, cmd='op', options=opts_f)

        opts_p = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
        params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts_p)

        if flops is not None:
            print('TF stats gives', flops.total_float_ops)

        if params is not None:
            print('TF stats gives', params.total_parameters)

        # print the result for different pixel error
        pixel_errors = [1, 2, 3, 4, 5, 7, 10, 15, 20]

        # get the csv files
        datasets = glob.glob('data/emma_data/*.txt')
        datasets = sorted(datasets)

        # we save the results of all dataset in to this list
        dataset_results = {}

        for d in datasets:

            # get the name of dataset from the path
            dataset_name = d.split("/")[2].split(".")[0]

            # save the result (differences) in the list
            dataset_results[dataset_name] = []

            dataset_len = get_len(d)

            batch_size = 2 * config["batch_size"]
            batch = read_batch(d, batch_size, dataset_name)

            # use tqdm progress bar
            tqdm_len = np.ceil(dataset_len / batch_size)
            with tqdm(total=tqdm_len, unit='batch') as t:
                # set the name of dataset as the title of progress bar
                t.set_description_str(dataset_name)

                test_images = []
                pred_labels = []

                # loop over batch of images
                for images, truths, shapes, pngs in batch:
                    predictions = model.predict(sess, images)

                    upscale_preds_x, upscale_preds_y, w = upscale_preds(predictions, shapes)
                    # calculate the difference
                    a = upscale_preds_x - truths[:, 0]
                    b = upscale_preds_y - truths[:, 1]

                    diff = np.sqrt((a * a + b * b))

                    dataset_results[dataset_name].extend(diff)
                    t.update()

                    # add images and predicted labels to test_images and pred_labels to creating the video
                    len_data = len(upscale_preds_x)
                    upscale_preds_x = np.reshape(upscale_preds_x, newshape=(len_data, 1))
                    upscale_preds_y = np.reshape(upscale_preds_y, newshape=(len_data, 1))
                    w = np.reshape(w, newshape=(len_data, 1))
                    upscale_center = np.concatenate((upscale_preds_x, upscale_preds_y, w), axis=1)
                    test_images.extend(pngs)
                    pred_labels.extend(upscale_center)

                # create the predicted labels on test sets
                if save_videos:
                    video_creator(dataset_name, test_images, pred_labels)

        # save the results in a dic
        dataset_errors = {}

        for key, val in dataset_results.items():
            dataset_errors[key] = []
            for e in pixel_errors:
                d = np.asarray(val, dtype=np.float32)
                acc = np.mean(np.asarray(d < e, dtype=np.int))
                dataset_errors[key].append(acc)

        print_resutls(dataset_errors, pixel_errors, dataset_names)
        return
        print("####### LPW #######")
        # run model on LPW dataset
        lpw_results = {}
        lpw_r = lpw_reader(batch_size=2 * config["batch_size"], normalize_image=True)
        for imgs, truths, d_name, shapes in lpw_r:
            # add dataset name to results dict
            if d_name not in lpw_results.keys():
                lpw_results[d_name] = []

            predictions = model.predict(sess, imgs)

            upscale_preds_x, upscale_preds_y, w = upscale_preds(predictions, shapes)

            # calculate the difference
            a = upscale_preds_x - truths[:, 0]
            b = upscale_preds_y - truths[:, 1]

            diff = np.sqrt((a * a + b * b))

            lpw_results[d_name].extend(diff)

        lpw_errors = {}

        for key, val in lpw_results.items():
            lpw_errors[key] = []
            for e in pixel_errors:
                d = np.asarray(val, dtype=np.float32)
                acc = np.mean(np.asarray(d < e, dtype=np.int))
                lpw_errors[key].append(acc)

        print_resutls(lpw_errors, pixel_errors)

        print("####### SWIRSKI #######")
        # run model on LPW dataset
        swk_results = {}
        swk_r = swirski_reader(batch_size=2 * config["batch_size"])
        for imgs, truths, d_name, shapes in swk_r:
            # add dataset name to results dict
            if d_name not in swk_results.keys():
                swk_results[d_name] = []

            predictions = model.predict(sess, imgs)

            upscale_preds_x, upscale_preds_y, w = upscale_preds(predictions, shapes)

            # calculate the difference
            a = upscale_preds_x - truths[:, 0]
            b = upscale_preds_y - truths[:, 1]

            diff = np.sqrt((a * a + b * b))

            swk_results[d_name].extend(diff)

        swk_errors = {}

        for key, val in swk_results.items():
            swk_errors[key] = []
            for e in pixel_errors:
                d = np.asarray(val, dtype=np.float32)
                acc = np.mean(np.asarray(d < e, dtype=np.int))
                swk_errors[key].append(acc)

        print_resutls(swk_errors, pixel_errors)


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

    args = parser.parse_args()

    # model_name = args.model_name
    model_name = args.model_name
    model_type = args.model_type

    # initial a logger
    logger = Logger(model_type, model_name, "", config, dir="models/")
    logger.log("Start reporting...")

    main(model_type, model_name, logger)
