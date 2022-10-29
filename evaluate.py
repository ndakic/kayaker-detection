from dataset import KayakerDataset
from config import PredictionConfig
from model import load_and_evaluate_model

ALL_LAYERS_TRAINED_MODELS = ["kayaker_test_1_all/mask_rcnn_kayaker_cfg_0005.h5",
                             "kayaker_test_1_all/mask_rcnn_kayaker_cfg_0010.h5",
                             "kayaker_test_1_all/mask_rcnn_kayaker_cfg_0015.h5",
                             "kayaker_test_1_all/mask_rcnn_kayaker_cfg_0020.h5"]

HEADS_LAYERS_TRAINED_MODELS = ["kayaker_test_3_heads/mask_rcnn_kayaker_cfg_0005.h5",
                               "kayaker_test_3_heads/mask_rcnn_kayaker_cfg_0010.h5",
                               "kayaker_test_3_heads/mask_rcnn_kayaker_cfg_0015.h5",
                               "kayaker_test_3_heads/mask_rcnn_kayaker_cfg_0020.h5"]


def evaluate_model(model_path):
    config = PredictionConfig()
    train_set = KayakerDataset()
    train_set.load_dataset('dataset', "train")
    train_set.prepare()
    test_set = KayakerDataset()
    test_set.load_dataset('dataset', "test")
    test_set.prepare()
    load_and_evaluate_model(model_path, train_set, test_set, config)


def compare_models():
    for model in ALL_LAYERS_TRAINED_MODELS:
        evaluate_model(model)
    for model in HEADS_LAYERS_TRAINED_MODELS:
        evaluate_model(model)
    # NOTE: Results can be found in: files/model_results.txt
