config = dict()
config["batch_size"] = 64
config["total_steps"] = 400000
config["validate_every"] = 5000
config["validate_for"] = 400
config["save_every"] = 3 * config["validate_every"]
config["n_epochs"] = 50

config["n_filters"] =    [16, 32, 128, 128, 256, 256]
config["filter_sizes"] = [3, 3  , 3  , 3  , 3  , 3]
config["max_pool"] =     [1, 1  , 1  , 1  , 1  , 1]
# check layer size >>    [96,48, 24,12, 6 , 3 ]
config["fc_layers"] = [256, 128]
# check layer size >>    [96, 48, 24, 12 , ]
config["learning_rate"] = [0.001, 0.0009, 0.0006, 0.0003, 0.0001, 0.00005, 0.00001]
config["decay_rate"] = 0.96
# Usually decay every half of epochs
config["decay_step"] = 40000
config["optimizer"] = "RMSProb"
config["keep_prob"] = 0.75
config["MAX_GRADIANT_NORM"] = 5.0
# input info
config["input_width"] = 192
config["input_height"] = 192
config["input_channel"] = 1
# Output shape
config["output_dim"] = 5
config["output_weights"] = [2.0, 2.0, 1.0, 1.0, 0.5]

# Augmenation parameters
config["prob_upscale"] = 0.75
config["max_upscale"] = 1.2
config["min_upscale"] = 2

config["prob_reflection"] = 0.75
config["min_reflection"] = 0.35
config["max_reflection"] = 0.85

config["prob_blur"] = 0.25
config["min_blurSize"] = 3
config["max_blurSize"] = 9
config["min_sigmaRatio"] = 0.25
config["max_sigmaRatio"] = 0.75

# config["prob_occlusion"] = 0.5
config["min_occlusion"] = 0.05
config["max_occlusion"] = 0.25
config["occlusion_max_obj"] = 6

# exposure on noisy frames
config["prob_exposure"] = 0.5
config["min_exposure"] = 0.8
config["max_exposure"] = 1.2

# crop input image
config["crop_probability"] = 0.5
config["crop_min_ratio"] = 0.8
config["crop_max_ratio"] = 0.95

# flip image
config["flip_probability"] = 0.5

# L2 regularization
config["l2_beta"] = 0.05

# VOC configs
config["voc_outputs"] = 21

