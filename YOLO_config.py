config = dict()
config["batch_size"] = 32
config["total_steps"] = 200000
config["validate_every"] = 2000
config["validate_for"] = 500
config["save_every"] = 6000

config["n_filters"] =    [16, 16, 64, 64, 128, 128, 256, 256, 512, 512, 1024, 512, 5]
config["filter_sizes"] = [3 , 3 , 3 , 3 , 3  , 3  , 3  , 3  , 3  , 3  , 3   , 3  , 1]
config["max_pool"] =     [1 , 0 , 1 , 0 , 1  , 0  , 1  , 0  , 1  , 0  , 1   , 1  , 0]
# check layer size >>    [96, 96, 48, 48, 24 , 24 , 12 , 12 , 6  , 6  , 3   , 2  , 1]
config["learning_rate"] = 0.007
config["decay_rate"] = 0.95
config["decay_step"] = 2000
config["optimizer"] = "adam"
config["keep_prob"] = 0.75
config["MAX_GRADIANT_NORM"] = 5.0
# input info
config["image_width"] = 192
config["image_height"] = 192
config["image_channel"] = 1
# Output shape
config["output_dim"] = 5



# TODO:
# -1. check YOLO last layer pooling
# 0. Check avgPooling on the last layer
# 1. show image + labels on it
# 2. Run a 3 layer network
# 3. Use googLeNet transfer learning
