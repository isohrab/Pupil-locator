import tensorflow as tf
import argparse
from Pupil_locator_model import Model
from config import config
from Batchizer import Batchizer
from tqdm import tqdm
from utils import *
from Logger import Logger
import numpy as np


def create_model(session, model_name, logger):
    model = Model(model_name, config, logger)
    ckpt = tf.train.get_checkpoint_state(model.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logger.log('Reloading model parameters..')
        model.restore(session, ckpt.model_checkpoint_path)

    else:
        logger.log('Created new model parameters..')
        session.run(tf.global_variables_initializer())

    return model


def print_predictions(result, logger):
    # logger.log("########### Print  Predictions ################")
    # logger.log("label: [\tx\t\t y\t\t w\t\t h\t\t a\t\t]")
    # for r in result:
    #     pred = r[0]
    #     y = r[1]
    #     logger.log("truth: {0:8.2f} {1:8.2f} {2:8.2f} {3:8.2f} {4:8.2f}".format(y[0],
    #                                                                             y[1],
    #                                                                             y[2],
    #                                                                             y[3],
    #                                                                             y[4]))
    #     logger.log("pred : {0:8.2f} {1:8.2f} {2:8.2f} {3:8.2f} {4:8.2f}\n".format(pred[0],
    #                                                                               pred[1],
    #                                                                               pred[2],
    #                                                                               pred[3],
    #                                                                               pred[4]))

    logger.log("########### Print  Predictions ################")
    logger.log("label: [\tx\t y")
    for r in result:
        pred = r[0]
        y = r[1]
        logger.log("truth: {0:8.2f} {1:8.2f} ".format(y[0], y[1]))
        logger.log("pred : {0:8.2f} {1:8.2f}\n".format(pred[0], pred[1]))

    logger.log("###############  End  ###################")


def main(model_name, logger):
    with tf.Graph().as_default() as g:
        with tf.Session() as sess:

            # Create a new model or reload existing checkpoint
            model = create_model(sess, model_name, logger)

            # Create a log writer object
            log_writer = tf.summary.FileWriter(model.model_dir, graph=sess.graph)

            valid_loss = 0
            epoch_loss = 0

            saver = tf.train.Saver(max_to_keep=3)

            # initial batchizer
            train_batchizer = Batchizer('data/train_data.csv', config["batch_size"])
            valid_batchizer = Batchizer('data/valid_data.csv', config["batch_size"])
            train_batches = train_batchizer.batches()
            valid_batches = valid_batchizer.batches()

            while model.global_step.eval() < config["total_steps"]:

                with tqdm(total=config["validate_every"], unit="batches") as t:
                    for x, y in train_batches:
                        if x is None:
                            continue

                        batch_loss, summary = model.train(sess, x, y, config["keep_prob"])
                        t.set_description_str("batch_loss:{0:8.2f}".format(batch_loss))
                        epoch_loss += batch_loss
                        log_writer.add_summary(summary, model.global_step.eval())
                        t.update(1)

                        if model.global_step.eval() % config["validate_every"] == 0:
                            break

                valid_counter = 0
                pred_result = []
                with tqdm(total=config["validate_for"], unit="batches") as t:
                    for x, y in valid_batches:
                        if x is None:
                            continue

                        batch_loss, _, pred = model.eval(sess, x, y)
                        t.set_description_str("batch_loss:{0:8.2f}".format(batch_loss))
                        valid_loss += batch_loss
                        valid_counter += 1

                        # select a random image from current batch and add it for visualization
                        # do it with a little chance! to reduce the size of output
                        if np.random.rand() > 0.95:
                            r = np.random.randint(0, high=len(x))
                            pred_result.append([pred[r], y[r]])

                        t.update(1)

                        if valid_counter == config["validate_for"]:
                            break

                print_predictions(pred_result, logger)
                logger.log('Step:{0:6}: train loss:{1:8.2f}, validation loss:{2:8.2f}'.format(model.global_step.eval(),
                                                                                              epoch_loss,
                                                                                              valid_loss))

                # save_every and validate_every should be dividable, otherwise this step will jump
                if model.global_step.eval() % config["save_every"] == 0:
                    save_path = saver.save(sess, model.model_dir, global_step=model.global_step)
                    logger.log("model saved at {}".format(save_path))

                epoch_loss = 0
                valid_loss = 0

                # Increase the epoch index of the model
                # model.global_epoch_step_op.eval()

            logger.log('Training is done.')


if __name__ == "__main__":
    class_ = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=class_)

    model_name = "simple_XY_l2_loss"
    model_comment = "simple with X Y labels only. batch normalization + drop out, add l2 regularization"

    logger = Logger(model_name, model_comment, config, logdir="models/" + model_name + "/")
    logger.log("Start training model...")
    main(model_name, logger)
