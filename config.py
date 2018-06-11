config = dict()
config["batch_size"] = 32
config["total_steps"] = 400000
config["validate_every"] = 4000
config["validate_for"] = 1300
config["save_every"] = 3 * config["validate_every"]

config["n_filters"] =    [16, 32, 128, 128, 256, 256]
config["filter_sizes"] = [3, 3  , 3  , 3  , 3  , 3]
config["max_pool"] =     [1, 1  , 1  , 1  , 1  , 1]
# check layer size >>    [96,48, 24,12, 6 , 3 ]
config["fc_layers"] = [256, 128]
# check layer size >>    [96, 48, 24, 12 , ]
config["learning_rate"] = [0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001]
config["decay_rate"] = 0.96
# Usually decay every half of epochs
config["decay_step"] = 50000
config["optimizer"] = "adam"
config["keep_prob"] = 0.75
config["MAX_GRADIANT_NORM"] = 5.0
# input info
config["image_width"] = 192
config["image_height"] = 192
config["image_channel"] = 1
# Output shape
config["output_dim"] = 4

# Augmenation parameters
config["prob_downscale"] = 0.75
config["max_downscale"] = 0.5
config["min_downscale"] = 0.9

config["prob_reflection"] = 0.75
config["min_reflection"] = 0.35
config["max_reflection"] = 0.75

config["prob_blur"] = 0.5
config["min_blurSize"] = 3
config["max_blurSize"] = 9
config["min_sigmaRatio"] = 0.25
config["max_sigmaRatio"] = 0.75

config["prob_occlusion"] = 0.5
config["min_occlusion"] = 0.25
config["max_occlusion"] = 0.50


# TODO:
# 2. Run a 3 layer network
# 3. Use googLeNet transfer learning
# split learning rate in x step (polynomial decay)
# avgPooling in the last layer!
#
