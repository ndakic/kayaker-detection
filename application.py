from dataset import KayakerDataset
from config import KayakerConfig, PredictionConfig
from model import KayakerModel
from util import plot_actual_vs_predicted
from video import predict_kayaker_on_video

VIDEO_PATH = "files/kayaker_malmo-1920x1080.mp4"
M_RCNN_COCO_MODEL_PATH = 'model/mask_rcnn_coco.h5'


def train_model():
    train_set = KayakerDataset()
    train_set.load_dataset('dataset', "train")
    train_set.prepare()
    print('Train: %d' % len(train_set.image_ids))

    test_set = KayakerDataset()
    test_set.load_dataset('dataset', "test")
    test_set.prepare()
    print('Test: %d' % len(test_set.image_ids))

    config = KayakerConfig()
    config.display()
    model = KayakerModel(model_weights_path=M_RCNN_COCO_MODEL_PATH,
                         mode='training',
                         exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"],
                         config=config)
    model.train(train_set, test_set)


def show_predictions(model):
    config = PredictionConfig()

    train_set = KayakerDataset()
    train_set.load_dataset('dataset', "train")
    train_set.prepare()
    print('Train: %d' % len(train_set.image_ids))

    # test/val set
    test_set = KayakerDataset()
    test_set.load_dataset('dataset', "test")
    test_set.prepare()
    print('Test: %d' % len(test_set.image_ids))

    plot_actual_vs_predicted(train_set, model, config)
    plot_actual_vs_predicted(test_set, model, config)


def pretrained_kayaker_model(trained_model_path):
    kayaker_config = PredictionConfig()
    kayaker_model = KayakerModel(model_weights_path=trained_model_path,
                                 mode="inference",
                                 exclude=[],
                                 config=kayaker_config)
    return kayaker_model.model


if __name__ == '__main__':
    print("Started")

    # train_model()

    model = pretrained_kayaker_model("model/mask_rcnn_kayaker_cfg_0006.h5")
    show_predictions(model)

    # predict_kayaker_on_video(model, VIDEO_PATH)
