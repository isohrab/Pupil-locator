config = dict()
config["batch_size"] = 32
config["n_epochs"] = 50
# config["n_conv_layer"] = 4
config["n_filters"] =    [64, 128, 256, 256, 256, 256, 512, 512, 512, 1024, 1024, 256, 5]
config["filter_sizes"] = [3 , 3  , 3  , 3  , 3  , 3  , 3  ,  3 , 3  , 3   , 3   , 1  , 1]
config["max_pool"] =     [1 , 1  , 0  , 1  , 0  , 1  , 0  ,  0 , 0  , 1   , 1   , 1  , 1]
# check layer size >>    [96, 48 , 48 , 24 , 24 , 12 , 12 ,  12, 12 , 6   , 3   , 2  , 1]
# config["fc_layers"] = [1024, 512, 128]
config["learning_rate"] = 0.01
config["decay_rate"] = 0.95
config["decay_step"] = 5000
config["optimizer"] = "adam"
config["keep_prob"] = 0.85
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
