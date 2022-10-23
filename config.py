from mrcnn.config import Config


# define a configuration for the model
class KayakerConfig(Config):
    # define the name of the configuration
    NAME = "kayaker_cfg"
    # number of classes (background + kayaker)
    NUM_CLASSES = 1 + 1
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # anchor side in pixels
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32
    DETECTION_MAX_INSTANCES = 1
    # DETECTION_MIN_CONFIDENCE = 0.90
    STEPS_PER_EPOCH = 100
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


# define the prediction configuration
class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "kayaker_cfg"
    # number of classes (background + kayaker)
    NUM_CLASSES = 1 + 1
    DETECTION_MAX_INSTANCES = 1
    DETECTION_MIN_CONFIDENCE = 0.95
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
