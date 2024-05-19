import torch
from torchvision.transforms import ConvertImageDtype
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes


def draw_bbox(image, cat, bbox, color='green', bbox_format='cxcywh'):
    convert = ConvertImageDtype(torch.uint8)
    image = image.cpu()
    cat = cat.cpu().detach()
    bbox = bbox.cpu().detach()
    if image.dtype != torch.uint8:
        image = convert(image)
    W, H = image.shape[-2:]
    bbox = bbox[cat != 0]
    cat = cat[cat != 0]
    bbox = bbox * torch.tensor([W, H, W, H], dtype=bbox.dtype)
    bbox = bbox.to(torch.int64)

    bbox = box_convert(bbox, bbox_format, 'xyxy')
    image = draw_bounding_boxes(image, bbox, colors=color, width=1)
    return image