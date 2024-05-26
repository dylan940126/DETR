import torch
from torchvision.transforms import ConvertImageDtype
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes


class MyVisualizer:
    def __init__(self, cat_names):
        self.cat_names = {c['id']: c['name'] for c in cat_names.values()}
        self.cat_names[0] = 'no object'

    def draw_bbox(self, img, cat, bbox, mask_cat=None, color='green', bbox_format='cxcywh'):
        """
        Draw bounding boxes on an image.
        This function returns a tensor in uint8 format.

        :param img: tensor image (C, H, W), support uint8 and float32
        :param cat: tensor category (N)
        :param bbox: tensor bounding boxes (N, 4)
        :param mask_cat: tensor mask category (N), default is None
        :param color: color of bounding boxes, default is 'green'
        :param bbox_format: input format of bounding boxes, default is 'cxcywh'
        :return: image with bounding boxes, tensor (C, H, W) uint8
        """
        if mask_cat is None:
            mask_cat = cat
        cvt_in = ConvertImageDtype(torch.uint8)
        cvt_out = ConvertImageDtype(torch.float32)
        img = img.clone()
        bbox = bbox.clone()
        cat = cat.clone()
        img = cvt_in(img)
        w, h = img.shape[-2:]
        bbox[:, [0, 2]] *= w
        bbox[:, [1, 3]] *= h
        if mask_cat is not None:
            bbox = bbox[mask_cat != 0]
            cat = cat[mask_cat != 0]
        if len(bbox) == 0:
            return img
        bbox = bbox.to(torch.int64)

        bbox = box_convert(bbox, bbox_format, 'xyxy')
        img = draw_bounding_boxes(img, bbox, colors=color, width=1,
                                  labels=[self.cat_names.get(c.item(), 'unknown') for c in cat])
        img = cvt_out(img)
        return img
