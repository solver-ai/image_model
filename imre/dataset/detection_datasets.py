import numpy as np

import torch
import torchvision
# from module.utils import BoxList

class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, transforms=None
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
        
        return image, targets

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data


if __name__ == "__main__":
    import cv2
    import albumentations as A
    from albumentations.pytorch.transforms import ToTensorV2
    valid_transform = A.Compose([
        A.LongestMaxSize(512),
        A.PadIfNeeded(
            min_height=512,
            min_width=512, 
            position='top_left', 
            border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
        ToTensorV2(),]

        ,bbox_params=A.BboxParams(format='coco')
    )

    valid_dataset = COCODataset(
        "/Users/hansoleom/Desktop/Lightning/image_model/datasets/deepfashion2/valid.json", 
        "/Users/hansoleom/Desktop/Lightning/image_model/datasets/deepfashion2/valid",
        valid_transform
    )
    def collate_fn(batch):
        return tuple(zip(*batch))


    # image, target, idx = next(iter(valid_dataset))
    test_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn)
    for data in test_dataloader:
        print(data[0])
        print(data[1])
        break