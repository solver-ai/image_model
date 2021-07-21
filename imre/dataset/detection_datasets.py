import numpy as np

import torch
import torchvision
from imre.module.utils import BoxList


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, transforms=None, remove_images_without_annotations=True
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)
        target = [obj["bbox"]+[obj["category_id"]] for obj in anno]
        if self._transforms is not None:
            img = np.array(img)
            transformed = self._transforms(image=img, bboxes=target)
            image = transformed['image']
            targets = transformed['bboxes']
        targets = torch.tensor(targets)
        target = BoxList(targets[:,:-1], (640,640), mode="xywh").convert("xyxy")
        target.add_field("labels", targets[:,-1])

        return image, target, idx


    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data