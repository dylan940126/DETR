import torch
from torchvision.transforms import ConvertImageDtype
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes


def plot(img, cat, bbox, color='green', bbox_format='cxcywh'):
    """
    Draw bounding boxes on an image.
    This function returns a tensor in uint8 format.

    :param img: tensor image (C, H, W), support uint8 and float32
    :param cat: tensor category (N)
    :param bbox: tensor bounding boxes (N, 4)
    :param color: color of bounding boxes, default is 'green'
    :param bbox_format: input format of bounding boxes, default is 'cxcywh'
    :return: image with bounding boxes, tensor (C, H, W) uint8
    """
    convert = ConvertImageDtype(torch.uint8)
    if img.dtype != torch.uint8:
        img = convert(img)
        w, h = img.shape[-2:]
        bbox[:, [0, 2]] *= w
        bbox[:, [1, 3]] *= h
    bbox = bbox[cat != 0]
    if len(bbox) == 0:
        return img
    cat = cat[cat != 0]
    bbox = bbox.to(torch.int64)

    bbox = box_convert(bbox, bbox_format, 'xyxy')
    img = draw_bounding_boxes(img, bbox, colors=color, width=1)
    return img
