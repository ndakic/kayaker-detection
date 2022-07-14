import os
import json
import skimage.draw
import numpy as np

from mrcnn.utils import Dataset


class KayakerDataset(Dataset):
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

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "dataset":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
