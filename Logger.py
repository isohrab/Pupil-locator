import os
import datetime


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
    def __init__(self, model_name, model_comment, params, logdir='log/'):
        self.params = params
        self.root = logdir
        # check if log directory exist
        if not os.path.isdir(self.root):
            os.mkdir(self.root)

        # check if config file is exist
        self.config_fn = "log.cfg"

        self.cfg_path = os.path.join(self.root, self.config_fn)
        if not os.path.isfile(self.cfg_path):
            # initial config file
            self.__config_initializer()

        # read values from config files
        self.config = self.__read_config()

        # increase the main run
        self.config["run"] = int(self.config["run"]) + 1

        # check if it is the same model name
        if self.config["model_name"] == model_name:
            # increase the model_run
            self.config["model_run"] = int(self.config["model_run"]) + 1
            self.__update_config()
        else:
            # change the model name and comment, and set model run to 1
            self.config["model_name"] = model_name
            self.config["model_comment"] = model_comment
            self.config["model_run"] = 1
            self.__update_config()

        # create a log file name
        self.log_fn = os.path.join(self.root, "log_{0}_{1}_{2}.txt".format(self.config["run"],
                                                                           self.config["model_name"],
                                                                           self.config["model_run"]))

        # write header information to the file
        with open(self.log_fn, mode="w") as f:
            f.write("run: {0}\n".format(self.config["run"]))
            now = datetime.datetime.now()
            f.write(now.strftime("%Y-%m-%d %H:%M"+"\n"))
            f.write("model name: {0}\n".format(self.config["model_name"]))
            f.write("comment: {0}\n".format(self.config["model_comment"]))
            f.write("model run: {0}\n".format(self.config["model_run"]))

            # loop over hyper parameters
            f.write("### Hyper-parameters ###\n")
            for k, v in self.params.items():
                f.write("{0}: {1}\n".format(k, v))

            # write the outputs title
            f.write("### OUTPUTS ###\n")

    def log(self, msg):
        # open log file and append message to it
        with open(self.log_fn, mode="a") as f:
            f.write(msg + "\n")

        # print out the msg too
        print(msg)

    def __config_initializer(self, ):
        with open(self.cfg_path, mode='w') as f:
            f.write("run \t 0\n")
            f.write("model_name \t null\n")
            f.write("model_comment \t no comment\n")
            f.write("model_run \t 0\n")

    def __read_config(self):
        with open(self.cfg_path, mode='r') as f:
            cfg = dict()
            for l in f:
                kv = l.split("\t")
                cfg[kv[0].strip()] = kv[1].strip()

        return cfg

    def __update_config(self):
        with open(self.cfg_path, mode='w') as f:
            f.write("run \t {0}\n".format(self.config["run"]))
            f.write("model_name \t {0}\n".format(self.config["model_name"]))
            f.write("model_comment \t {0}\n".format(self.config["model_comment"]))
            f.write("model_run \t {0}\n".format(self.config["model_run"]))

if __name__ == "__main__":
    # define a dump dict
    param = {"first": 1,
             "second": 2,
             "third": 3,
             "last": "is another"}

    logger = Logger("simple2", "it is a simple2 model. it can detect everything.", param)
    logger.log("this is an output from simple 2.")