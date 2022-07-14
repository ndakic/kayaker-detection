from mrcnn.config import Config


# define a configuration for the model
class KayakerConfig(Config):
    # define the name of the configuration
    NAME = "kayaker_cfg"
    # number of classes (background + kayaker)
    NUM_CLASSES = 1 + 1
    # IMAGE_MIN_DIM = 384
    # IMAGE_MAX_DIM = 384
    DETECTION_MAX_INSTANCES = 5
    DETECTION_MIN_CONFIDENCE = 0.90
    STEPS_PER_EPOCH = 35


# define the prediction configuration
class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "kayaker_cfg"
    # number of classes (background + kayaker)
    NUM_CLASSES = 1 + 1
    DETECTION_MAX_INSTANCES = 1
    DETECTION_MIN_CONFIDENCE = 0.95
