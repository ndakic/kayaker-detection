from mrcnn.config import Config


# define a configuration for the model
class KayakerConfig(Config):
    # define the name of the configuration
    NAME = "kayaker_cfg"
    # number of classes (background + kayaker)
    NUM_CLASSES = 1 + 1

    IMAGE_MIN_DIM = 384
    IMAGE_MAX_DIM = 384

    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    DETECTION_MAX_INSTANCES = 5
    DETECTION_MIN_CONFIDENCE = 0.90
    STEPS_PER_EPOCH = 33

    # DETECTION_NMS_THRESHOLD = 0.6
    # TRAIN_ROIS_PER_IMAGE = 32
    # MAX_GT_INSTANCES = 4


# define the prediction configuration
class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "kayaker_cfg"
    # number of classes (background + kayaker)
    NUM_CLASSES = 1 + 1
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MAX_INSTANCES = 1
    DETECTION_MIN_CONFIDENCE = 0.90
