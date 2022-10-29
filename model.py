from numpy import expand_dims
from numpy import mean

from mrcnn.model import mold_image
from mrcnn.model import load_image_gt
from mrcnn.utils import compute_ap
from mrcnn.model import MaskRCNN


class KayakerModel:
    def __init__(self, model_weights_path, mode, exclude, config):
        self.model_weights_path = model_weights_path
        self.mode = mode
        self.exclude = exclude
        self.config = config
        self.model = MaskRCNN(mode=mode, model_dir='./', config=self.config)
        self.model.load_weights(filepath=self.model_weights_path,
                                by_name=True,
                                exclude=exclude)

    def train(self, train_set, test_set):
        self.model.train(train_set, test_set,
                         learning_rate=self.config.LEARNING_RATE * 2,
                         epochs=20,
                         layers='heads')


def evaluate_model(dataset, model, cfg):
    APs = list()
    for image_id in dataset.image_ids:
        # print(f'Processing image with id: {image_id}')
        # load image, bounding boxes and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id,
                                                                         use_mini_mask=False)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)
        # extract results for first sample
        r = yhat[0]
        # calculate statistics, including AP
        AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        # store
        APs.append(AP)
    # calculate the mean AP across all images
    mAP = mean(APs)
    return mAP


def load_and_evaluate_model(model_path, train_set, test_set, config):
    # define the model
    model = MaskRCNN(mode='inference', model_dir='./', config=config)
    # load model weights
    model.load_weights(model_path, by_name=True)
    # evaluate model on training dataset
    train_mAP = evaluate_model(train_set, model, config)
    print("Train mAP: %.3f" % train_mAP)
    # evaluate model on test dataset
    test_mAP = evaluate_model(test_set, model, config)
    print("Test mAP: %.3f" % test_mAP)
