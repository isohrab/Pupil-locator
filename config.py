config = dict()
config["batch_size"] = 64
config["total_steps"] = 400000
config["validate_every"] = 4000
config["validate_for"] = 500
config["save_every"] = 3 * config["validate_every"]

config["n_filters"] =    [16, 32, 128, 128, 256, 256]
config["filter_sizes"] = [3, 3 , 3, 3 , 3 , 3]
config["max_pool"] =     [1, 1 , 1, 1 , 1 , 1]
# check layer size >>    [96,48, 24,12, 6 , 3 ]
config["fc_layers"] = [256, 128]
# check layer size >>    [96, 48, 24, 12 , ]
config["learning_rate"] = 0.01
config["decay_rate"] = 0.95
# Usually decay every half of epochs
config["decay_step"] = 5000
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
# -1. check YOLO last layer pooling
# 0. Check avgPooling on the last layer
# 1. show image + labels on it
# 2. Run a 3 layer network
# 3. Use googLeNet transfer learning
