config = dict()
config["batch_size"] = 32
config["total_steps"] = 200000
config["validate_every"] = 2000
config["validate_for"] = 500
config["save_every"] = 6000

config["n_filters"] =    [16, 32, 128, 128, 256, 256]
config["filter_sizes"] = [3, 3 , 3, 3 , 3 , 3]
config["max_pool"] =     [1, 1 , 1, 1 , 1 , 1]
# check layer size >>    [96,48, 24,12, 6 , 3 ]
config["fc_layers"] = [256, 128]
# check layer size >>    [96, 48, 24, 12 , ]
config["learning_rate"] = 0.001
config["decay_rate"] = 0.95
config["decay_step"] = 2000
config["optimizer"] = "adam"
config["keep_prob"] = 0.85
config["MAX_GRADIANT_NORM"] = 5.0
# input info
config["image_width"] = 192
config["image_height"] = 192
config["image_channel"] = 1
# Output shape
config["output_dim"] = 2



# TODO:
# -1. check YOLO last layer pooling
# 0. Check avgPooling on the last layer
# 1. show image + labels on it
# 2. Run a 3 layer network
# 3. Use googLeNet transfer learning
