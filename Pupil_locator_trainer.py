import numpy as np
import tensorflow as tf
import argparse
from Pupil_locator_model import Model
from config import config
from Batchizer import Batchizer
from tqdm import tqdm

def main():

    model_dir = "model/"

    with tf.Graph().as_default() as g:
        with tf.Session() as sess:
            epoch_loss = 0

            # Create a new model or reload existing checkpoint
            model = Model(config)
            sess.run(tf.global_variables_initializer())

            # Create a log writer object
            log_writer = tf.summary.FileWriter(model_dir, graph=sess.graph)

            valid_loss = 0

            saver = tf.train.Saver(max_to_keep=3)

            train_batchizer = Batchizer('data/train/', config["batch_size"], 'train')
            test_batchizer = Batchizer('data/test/', config["batch_size"], 'test')

            for e in range(config["n_epochs"]):
                # initial batchizer

                train_batches = train_batchizer.batches()
                test_batches = test_batchizer.batches()

                if model.global_epoch_step.eval() >= config["n_epochs"]:
                    print('Training is already complete.')
                    break

                for x, y in tqdm(train_batches, total=train_batchizer.n_batches, unit="batches"):
                    if x is None:
                        continue

                    batch_loss, summary = model.train(sess, x, y, config["keep_prob"])
                    epoch_loss += batch_loss
                    log_writer.add_summary(summary, model.global_step.eval())

                for x, y in tqdm(test_batches, total=test_batchizer.n_batches, unit="batches"):
                    batch_loss, _, _ = model.eval(sess, x, y)
                    valid_loss += batch_loss

                print("epoch:{0:2}: train loss:{1:8.2f},".format(e, epoch_loss),
                      "validation loss:{0:8.2f}".format(valid_loss))
                epoch_loss = 0
                valid_loss = 0

                # Increase the epoch index of the model
                model.global_epoch_step_op.eval()

        #             checkpoint_path = os.path.join(HP.MODEL_DIR, HP.MODEL_NAME)
        #             save_path = saver.save(sess, checkpoint_path, global_step=model.global_step)
        #             print('model saved at %s' % save_path)
        print('Training Terminated')


if __name__ == "__main__":
    # class_ = argparse.ArgumentDefaultsHelpFormatter
    # parser = argparse.ArgumentParser(description=__doc__,
    #                                  formatter_class=class_)
    # parser.add_argument('-d',
    #                     help='imdb or 20newsgroups',
    #                     default='imdb',
    #                     dest='dataset')
    # parser.add_argument('-w',
    #                     help='word embedding: glove or fasttext or lexvec',
    #                     default='glove',
    #                     dest='word_embedding')
    # parser.add_argument('--use_encoder',
    #                     help="use pretrained encoder",
    #                     default='yes',
    #                     dest='use_encoder')
    # parser.add_argument('--train_encoder',
    #                     help="allow pretrained encoder to train during the classification",
    #                     default='yes',
    #                     dest='train_encoder')
    #
    # args = parser.parse_args()
    #
    # use_encoder = True
    # if args.use_encoder.lower() == 'no':
    #     use_encoder = False
    #
    # train_encoder = True
    # if args.train_encoder.lower() == 'no':
    #     train_encoder = False

    main()