import os
import tensorflow as tf
import time
import argparse
import cv2
import numpy as np
from config import config
from utils import anotator, change_channel, create_noisy_video
from Logger import Logger
from yolo import Model as YModel
from simple import Model as SModel
from gap import Model as GModel
from nasnet import Model as NModel
from augmentor import Augmentor


def load_model(session, m_type, m_name, logger):
    # load the weights based on best loss
    best_dir = "best_loss"

    # check model dir
    model_path = "models/"+m_name
    path = os.path.join(model_path, best_dir)
    if not os.path.exists(path):
        raise FileNotFoundError

    if m_type == "simple":
        model = SModel(m_name, config, logger)
    elif m_type == "YOLO":
        model = YModel(m_name, config, logger)
    elif m_type == "GAP":
        model = GModel(m_name, config, logger)
    elif m_type == "NAS":
        model = NModel(m_name, config, logger)
    else:
        raise ValueError

    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logger.log('Reloading model parameters..')
        model.restore(session, ckpt.model_checkpoint_path)

    else:
        raise ValueError('There is no best model with given model')

    return model


def main(m_type, m_name, logger, video_path=None, write_output=True):
    with tf.Session() as sess:

        # load best model
        model = load_model(sess, m_type, m_name, logger)

        # check input source is a file or camera
        if video_path == None:
            video_path = 0

        # load the video or camera
        cap = cv2.VideoCapture(video_path)
        ret = True
        counter = 0
        tic = time.time()
        frames = []
        preds = []
        # exponential weight decay


        while ret:
            ret, frame = cap.read()

            if ret:
                # Our operations on the frame come here
                if frame.shape[0] != 192:
                    frame = cv2.resize(frame, (192, 192))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                image = change_channel(gray, config["image_channel"])
                p = model.predict(sess, [image])
                preds.append(p[0])
                frames.append(gray)
                counter += 1

        toc = time.time()
        print("{0:0.2f} FPS".format(counter / (toc - tic)))

    # prepare a video write to show the result
    beta = np.ones((config["output_dim"]), dtype=np.float32) * 0.0
    loc = np.ones((config["output_dim"]), dtype=np.float32)
    if write_output:
        video = cv2.VideoWriter("predicted_video.avi", cv2.VideoWriter_fourcc(*"XVID"), 30, (192, 192))

        for i, img in enumerate(frames):
            loc = beta * loc + (1-beta) * preds[i]
            # if (i+1) % 2 == 0:
            labeled_img = anotator(img, preds[i])
            video.write(labeled_img)

        # close the video
        cv2.destroyAllWindows()
        video.release()
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
    main(model_type, model_name, logger, video_input)
