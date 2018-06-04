import os
import datetime
from utils import *


class Logger(object):
    """
    for each run, we create a log file, the file is like so:
    Run #: 1234
    date: XX:YY:ZZ
    Model name: a model name
    Model Comment: the details of model
    model run #: 23
    CONFIG PARAMETERS
    Key: value
    .
    .
    Output results:
    epoch 1: train XXX, .....
    epoch 2: ......
    .
    .

    """

    def __init__(self, model_type, model_name, model_comment, params, dir='models/'):
        self.params = params
        self.root = dir
        self.model_type = model_type
        # check if log directory exist
        check_dir(dir)

        # check model_name dir
        self.model_dir = os.path.join(self.root, model_name)
        check_dir(self.model_dir)

        # check if config file is exist
        self.config_fn = "log.cfg"

        self.cfg_path = os.path.join(self.model_dir, self.config_fn)
        if not os.path.isfile(self.cfg_path):
            # initial config file
            self.__config_initializer(model_type, model_name, model_comment)

        # read values from config files
        self.config = self.__read_config()

        # load the best validation loss
        self.best_loss = float(self.config["best_loss"])

        # increase the main run
        self.config["run"] = str(int(self.config["run"]) + 1)

        self.__update_config()

        # create a log file name for current run
        self.log_fn = os.path.join(self.model_dir, "log_{0}_{1}.txt".format(self.config["model_name"],
                                                                            self.config["run"]))

        # write header information to the file
        with open(self.log_fn, mode="w") as f:
            f.write("run: {0}\n".format(self.config["run"]))
            now = datetime.datetime.now()
            f.write(now.strftime("%Y-%m-%d %H:%M" + "\n"))
            f.write("model type: {0}\n".format(self.config["model_type"]))
            f.write("model name: {0}\n".format(self.config["model_name"]))
            f.write("comment: {0}\n".format(self.config["model_comment"]))

            # loop over hyper parameters
            f.write("### Hyper-parameters ###\n")
            for k, v in self.params.items():
                f.write("{0}: {1}\n".format(k, v))

            # write the outputs title
            f.write("### OUTPUTS ###\n")

    def log(self, msg, t=None):
        # open log file and append message to it
        with open(self.log_fn, mode="a") as f:
            f.write(msg + "\n")

        # print out the msg too
        # if tqdm is provided
        if t is not None:
            t.write(msg)
        else:
            print(msg)

    def __config_initializer(self, typ, name, comment):
        with open(self.cfg_path, mode='w') as f:
            f.write("run \t 0\n")
            f.write("model_type \t {}\n".format(typ))
            f.write("model_name \t {}\n".format(name))
            f.write("model_comment \t {}\n".format(comment))
            f.write("best_loss \t inf\n")

    def __read_config(self):
        with open(self.cfg_path, mode='r') as f:
            cfg = dict()
            for l in f:
                kv = l.split("\t")
                cfg[kv[0].strip()] = kv[1].strip()

        return cfg

    def __update_config(self):
        with open(self.cfg_path, mode='w') as f:
            for k, v in self.config.items():
                f.write("{0} \t {1}\n".format(k, v))

    def save_best_loss(self, loss):
        self.best_loss = loss
        self.config["best_loss"] = self.best_loss
        self.__update_config()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.model_type == "simple":
            code = open("Simple_model.py", mode='r').read()
            with open(self.log_fn, mode="a") as f:
                f.write(code)
        elif self.model_type == "YOLO":
            code = open("YOLO_model.py", mode='r').read()
            with open(self.log_fn, mode="a") as f:
                f.write(code)




if __name__ == "__main__":
    # define a dump dict for test porpuse
    param = {"first": 1,
             "second": 2,
             "third": 3,
             "last": "is another"}

    logger = Logger("simple2", "it is a simple2 model. it can detect everything.", param)
    logger.log("this is an output from simple 2.")
