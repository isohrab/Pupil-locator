import argparse

import tensorflow as tf
from tqdm import tqdm

from augmentor import Augmentor
from batchizer import Batchizer
from logger import Logger
from models import Simple, NASNET, Inception, GAP, YOLO
from utils import *


def create_model(session, m_type, m_name, logger):
    """
    create or load the last saved model
    :param session: tf.session
    :param m_type: model type
    :param m_name: model name (equal to folder name)
    :param logger: logger
    :return: None
    """
    if m_type == "simple":
        model = Simple(m_name, config, logger)
    elif m_type == "YOLO":
        model = YOLO(m_name, config, logger)
    elif m_type == 'GAP':
        model = GAP(m_name, config, logger)
    elif m_type == 'NAS':
        model = NASNET(m_name, config, logger)
    elif m_type == 'INC':
        model = Inception(m_name, config, logger)
    else:
        raise ValueError

    ckpt = tf.train.get_checkpoint_state(model.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logger.log('Reloading model parameters..')
        model.restore(session, ckpt.model_checkpoint_path)

    else:
        logger.log('Created new model parameters..')
        session.run(tf.global_variables_initializer())

    return model


def print_predictions(result, logger):
    """
    print predicted results every epoch
    :param result: results
    :param logger: logger
    :return: None
    """
    logger.log("########### Print  Predictions ################")
    logger.log("label: [\tx\t y\t w\t h\t a]")
    for r in result:
        y = r[0]
        pred = r[1]
        img_path = r[2]

        logger.log("Path: " + img_path)
        logger.log("truth: {0:2.2f} {1:2.2f} {2:2.2f}".format(y[0],
                                                              y[1],
                                                              y[2]))
        logger.log("pred : {0:2.2f} {1:2.2f} {2:2.2f}\n".format(pred[0],
                                                                pred[1],
                                                                pred[2]))


def main(model_type, model_name, logger):
    """
    train model until the maximum number of steps reached
    :param model_type: model type
    :param model_name: model name
    :param logger: logger
    :return: None
    """

    with tf.Graph().as_default() as g:

        with tf.Session() as sess:

            # Create a new model or reload existing checkpoint
            model = create_model(sess, model_type, model_name, logger)

            # Create a log writer object
            log_writer = tf.summary.FileWriter(model.model_dir, graph=sess.graph)

            valid_loss = []
            train_loss = []

            # initial saver for
            #   1. save every 3 epcohs
            saver = tf.train.Saver(max_to_keep=3)
            #   2. the best loss
            best_saver = tf.train.Saver(max_to_keep=1)

            # CSV files for train and test set
            root_path = "data/"
            train_csv = "train_data.csv"
            valid_csv = "valid_data.csv"

            train_path = os.path.join(root_path, train_csv)
            valid_path = os.path.join(root_path, valid_csv)
            # initial batchizer
            train_batchizer = Batchizer(train_path, config["batch_size"])
            valid_batchizer = Batchizer(valid_path, config["batch_size"])

            # init augmentor only once for both train and validation set
            ag = Augmentor('data/noisy_videos/', config)
            train_batches = train_batchizer.batches(ag,
                                                    config["output_dim"],
                                                    num_c=config["input_channel"],
                                                    zero_mean=True)
            valid_batches = valid_batchizer.batches(ag,
                                                    config["output_dim"],
                                                    num_c=config["input_channel"],
                                                    zero_mean=True)

            while model.global_step.eval() < config["total_steps"]:
                # get the learning rate from config file
                lr_idx = int(model.global_step.eval() / config["decay_step"])
                lr_idx = min(lr_idx, len(config["learning_rate"]) - 1)
                lr = config["learning_rate"][lr_idx]

                # train phase
                with tqdm(total=config["validate_every"], unit="batch") as t:
                    for x, y, _ in train_batches:
                        if x is None:
                            continue

                        batch_loss, summary = model.train(sess, x, y, config["keep_prob"], lr)
                        train_loss.append(batch_loss)

                        t.set_description_str("batch_loss:{0:2.8f}, ".format(batch_loss))
                        log_writer.add_summary(summary, model.global_step.eval())
                        t.update(1)

                        if model.global_step.eval() % config["validate_every"] == 0:
                            break

                # validation phase
                valid_counter = 0
                pred_result = []
                with tqdm(total=config["validate_for"], unit="batch") as t:
                    for x, y, img in valid_batches:
                        if x is None:
                            continue

                        batch_loss, _, pred = model.eval(sess, x, y)
                        valid_loss.append(batch_loss)

                        t.set_description_str("batch_loss:{0:2.8f}".format(batch_loss))
                        valid_counter += 1

                        # select a random image from current batch and add it for visualization
                        # do it with a little chance! to reduce the size of output
                        if np.random.rand() > 0.95:
                            r = np.random.randint(0, high=len(x))
                            pred_result.append([y[r], pred[r], img[r]])

                        t.update(1)

                        if valid_counter == config["validate_for"]:
                            break

                # print the results of validation dataset
                print_predictions(pred_result, logger)
                train_mean_loss = np.mean(train_loss)
                valid_mean_loss = np.mean(valid_loss)
                logger.log(
                    'Step:{0:6}: avg train loss:{1:2.8f}, avg validation loss:{2:2.8f}'.format(model.global_step.eval(),
                                                                                               train_mean_loss,
                                                                                               valid_mean_loss))

                # save a checkpoint with the best loss value
                if valid_mean_loss < logger.best_loss:
                    logger.save_best_loss(valid_mean_loss)
                    best_path = os.path.join(model.model_dir, "best_loss/")
                    check_dir(best_path)
                    save_path = best_saver.save(sess, best_path, global_step=model.global_step)
                    logger.log("model saved with best loss {0} at {1}".format(valid_mean_loss,
                                                                              save_path))

                # save_every and validate_every should be dividable, otherwise this step will jump
                if model.global_step.eval() % config["save_every"] == 0:
                    save_path = saver.save(sess, model.model_dir, global_step=model.global_step)
                    logger.log("model saved at {}".format(save_path))
                summary = tf.Summary()
                summary.value.add(tag="train_loss", simple_value=train_mean_loss)
                summary.value.add(tag="valid_loss", simple_value=valid_mean_loss)

                log_writer.add_summary(summary, model.global_step.eval())
                train_loss = []
                valid_loss = []

            logger.log('Training is done.')


if __name__ == "__main__":
    class_ = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=class_)

    parser.add_argument('model_name',
                        help="name of saved model (3A4Bh-Ref25)",
                        )

    parser.add_argument('--model_type',
                        help="INC, YOLO, simple",
                        default="INC")

    parser.add_argument('--model_message',
                        help="briefly explain your model",
                        default="none")

    args = parser.parse_args()

    model_type = args.model_type
    model_name = args.model_name
    model_msg = args.model_message

    logger = Logger(model_type, model_name, model_msg, config, dir="models/")
    logger.log("Start training model...")
    main(model_type, model_name, logger)
