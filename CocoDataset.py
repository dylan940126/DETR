import os
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class CocoDataset(Dataset):
    def __init__(self, root, annFile, transform=None, category=[], device='cpu'):
        self.root = root
        self.coco = COCO(annFile)
        self.category_ids = self.coco.getCatIds(catNms=category)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform
        self.device = device

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
        img = img.to(self.device)
        trans_x, trans_y = img.shape[-1], img.shape[-2]

        # get Annotations
        target = coco.loadAnns(ann_ids)
        if target:
            cat = [x['category_id'] for x in target]
            bbox = [[x['bbox'][0] / img_x * trans_x,
                     x['bbox'][1] / img_y * trans_y,
                     x['bbox'][2] / img_x * trans_x,
                     x['bbox'][3] / img_y * trans_y] for x in target]
            ## to tensor
            cat = torch.tensor(cat, device=self.device)
            bbox = torch.tensor(bbox, device=self.device)
            ## pad to 100
            cat = torch.cat([cat, torch.zeros(100 - len(cat), device=self.device, dtype=torch.long)])
            bbox = torch.cat([torch.tensor(bbox), torch.zeros(100 - len(bbox), 4, device=self.device)])
        else:
            cat = torch.zeros(100, device=self.device, dtype=torch.long)
            bbox = torch.zeros(100, 4, device=self.device)
        return img, (cat, bbox)

    def __len__(self):
        return len(self.ids)


def collate_fn(batch):
    tmp = list(zip(*batch))
    img = torch.stack(tmp[0])
    cat = torch.stack([target[0] for target in tmp[1]])
    bbox = torch.stack([target[1] for target in tmp[1]])
    return img, cat, bbox
