from dataset import KayakerDataset
from config import KayakerConfig, PredictionConfig
from model import KayakerModel
from util import plot_actual_vs_predicted
import cv2.cv2 as cv2
from mrcnn import visualize
from util import COCO_DATASET_CLASSES

VIDEO_PATH = "files/kayaker_malmo-1920x1080.mp4"
M_RCNN_COCO_MODEL_PATH = 'model/mask_rcnn_coco.h5'


def train_model():
    train_set = KayakerDataset()
    train_set.load_dataset('dataset', is_train=True)
    train_set.prepare()
    print('Train: %d' % len(train_set.image_ids))

    test_set = KayakerDataset()
    test_set.load_dataset('dataset', is_train=False)
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
    train_set.load_dataset('dataset', is_train=True)
    train_set.prepare()
    print('Train: %d' % len(train_set.image_ids))

    # test/val set
    test_set = KayakerDataset()
    test_set.load_dataset('dataset', is_train=False)
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


def predict_kayaker_on_video(trained_model_path):
    model = pretrained_kayaker_model(trained_model_path)
    video = cv2.VideoCapture(VIDEO_PATH)
    total_frames = 0
    while True:
        _, frame = video.read()
        if frame is not None and total_frames % 10 == 0:
            total_frames += 1
            results = model.detect([frame], verbose=1)
            result = results[0]
            visualize.display_instances(image=frame,
                                        boxes=result['rois'],
                                        masks=result['masks'],
                                        class_ids=result['class_ids'],
                                        class_names=["bg", "kayaker"],
                                        scores=result['scores'],
                                        show_mask=False, )
            print(result)
        else:
            total_frames += 1

        if total_frames > 135:
            break

    print("Total frames: ", total_frames)
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("Started")

    train_model()
    # predict_kayaker_on_video("kayaker_cfg20220713T2041/mask_rcnn_kayaker_cfg_0002.h5")

    # model = pretrained_kayaker_model("model/mask_rcnn_kayaker_cfg_0015.h5")
    # show_predictions(model)
