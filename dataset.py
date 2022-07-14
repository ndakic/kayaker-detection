from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
import os
import json
import skimage.draw
import numpy as np


class KayakerDataset(Dataset):
    # deprecated TODO: remove later
    def load_dataset_(self, dataset_dir, is_train=True):
        self.add_class("dataset", 1, "kayaker")
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'
        for filename in listdir(images_dir):
            image_id = filename[:-4]
            if is_train and int(image_id) >= 34:
                continue
            if not is_train and int(image_id) < 34:
                continue
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    def load_dataset(self, dataset_dir, subset):
        self.add_class("dataset", 1, "kayaker")
        assert subset in ["train", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)
        annotations_ = json.load(open(os.path.join(dataset_dir, "annotations.json")))
        annotations = list(annotations_.values())  # don't need the dict keys
        annotations = [a for a in annotations if a['regions']]
        for a in annotations:
            polygons = [r['shape_attributes'] for r in a['regions']]
            objects = [s['region_attributes']['name'] for s in a['regions']]
            name_dict = {"kayaker": 1}
            num_ids = [name_dict[a] for a in objects]
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            self.add_image(
                "dataset",
                image_id=a['filename'],
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids
            )

    # deprecated TODO: remove later
    def extract_boxes(self, filename):
        # load and parse the file
        tree = ElementTree.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

    # deprecated TODO: remove later
    def load_mask_(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        # load XML
        boxes, w, h = self.extract_boxes(path)
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('kayaker'))
        return masks, asarray(class_ids, dtype='int32')

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "dataset":
            return super(self.__class__, self).load_mask(image_id)
        info = self.image_info[image_id]
        if info["source"] != "dataset":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids

    # deprecated TODO: remove later
    def image_reference_(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "dataset":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
