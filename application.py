from dataset import KayakerDataset
from config import KayakerConfig, PredictionConfig
from model import KayakerModel, load_and_evaluate_model
from util import plot_actual_vs_predicted
from video import predict_kayaker_on_video

INPUT_VIDEO_PATH = "files/kayaker_malmo_8_sec_compressed.mp4"
M_RCNN_COCO_MODEL_PATH = 'model/mask_rcnn_coco.h5'
MODEL_PATH = "model/mask_rcnn_kayaker_M2.h5"  # SET YOUR MODEL PATH


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


def pretrained_kayaker_model(trained_model_path):
    kayaker_config = PredictionConfig()
    kayaker_model = KayakerModel(model_weights_path=trained_model_path,
                                 mode="inference",
                                 exclude=[],
                                 config=kayaker_config)
    return kayaker_model.model


def evaluate():
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

    load_and_evaluate_model(MODEL_PATH, train_set, test_set, config)


if __name__ == '__main__':
    print("Started")

    # uncomment this to train model
    # train_model()

    # uncomment this to generate output video
    # model = pretrained_kayaker_model(MODEL_PATH)
    # predict_kayaker_on_video(model, INPUT_VIDEO_PATH)

    print("Done!")
