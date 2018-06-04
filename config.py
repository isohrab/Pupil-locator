config = dict()
config["batch_size"] = 32
config["total_steps"] = 400000
config["validate_every"] = 500
config["validate_for"] = 200
config["save_every"] = 3 * config["validate_every"]

config["n_filters"] =    [16, 32, 128, 128, 256, 256]
config["filter_sizes"] = [3, 3  , 3  , 3  , 3  , 3]
config["max_pool"] =     [1, 1  , 1  , 1  , 1  , 1]
# check layer size >>    [96,48, 24,12, 6 , 3 ]
config["fc_layers"] = [256, 128]
# check layer size >>    [96, 48, 24, 12 , ]
config["learning_rate"] = [0.001, 0.0001, 0.00001, 0.000001]
config["decay_rate"] = 0.96
# Usually decay every half of epochs
config["decay_step"] = 100000
config["optimizer"] = "adam"
config["keep_prob"] = 0.7
config["MAX_GRADIANT_NORM"] = 5.0
# input info
config["image_width"] = 192
config["image_height"] = 192
config["image_channel"] = 1
# Output shape
config["output_dim"] = 4



# TODO:
# 2. Run a 3 layer network
# 3. Use googLeNet transfer learning
# split learning rate in x step (polynomial decay)
# avgPooling in the last layer!
#
