import os
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision.ops import box_convert
from functools import lru_cache


class CocoDataset(Dataset):
    def __init__(self, root, annFile, transform=None, category=None, num_queries=100):
        if category is None:
            category = []
        self.root = root
        self.coco = COCO(annFile)
        self.category_ids = self.coco.getCatIds(catNms=category)
        self.ids = list(sorted(self.coco.getImgIds()))
        self.transform = transform
        self.num_queries = num_queries

    # @lru_cache(maxsize=1024)
    def __getitem__(self, index):
        coco = self.coco
        # get ids
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=self.category_ids)

        # get image
        img = coco.loadImgs(img_id)[0]
        path = img['file_name']
        img_x, img_y = img['width'], img['height']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # get Annotations
        target = coco.loadAnns(ann_ids)
        if target:
            cat = [x['category_id'] for x in target]
            bbox = [[x['bbox'][0] / img_x,
                     x['bbox'][1] / img_y,
                     x['bbox'][2] / img_x,
                     x['bbox'][3] / img_y] for x in target]
            ## to tensor
            cat = torch.tensor(cat)
            bbox = torch.tensor(bbox)
            ## bbox format: cx, cy, w, h
            bbox = box_convert(bbox, 'xywh', 'cxcywh')
            ## pad to 100
            cat = torch.cat([cat, torch.zeros(self.num_queries - len(cat), dtype=torch.long)])
            bbox = torch.cat([bbox, torch.ones(self.num_queries - len(bbox), 4)])
        else:
            cat = torch.zeros(self.num_queries, dtype=torch.long)
            bbox = torch.ones(self.num_queries, 4)
        return img, (cat, bbox)

    def __len__(self):
        return len(self.ids)


def collate_fn(batch):
    tmp = list(zip(*batch))
    img = torch.stack(tmp[0])
    cat = torch.stack([target[0] for target in tmp[1]])
    bbox = torch.stack([target[1] for target in tmp[1]])
    return img, (cat, bbox)
