import torch
import psutil
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.backends import cudnn
from MyCocoDataset import CocoDataset, collate_fn
from MyDETR import DETR
from tqdm import tqdm
from torchvision.ops import box_convert
import datetime
import json
from torchinfo import summary


def get_result():
    # Load COCO dataset
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    coco_val = CocoDataset(root='coco/images', annFile=annFile,
                           transform=transform, category=[], num_queries=num_queries)
    val_loader = DataLoader(coco_val, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers,
                            collate_fn=collate_fn)

    results = []

    # Validation
    model.eval()
    with torch.no_grad():
        val_loader_tqdm = tqdm(val_loader)
        for i, (images, target) in enumerate(val_loader_tqdm):
            img_ids = images[0]
            images = images[1].to(device)
            targ_cat = target[0].to(device)
            targ_bbox = target[1].to(device)
            pred_cat, pred_bbox = model(images)
            pred_score = pred_cat.softmax(dim=-1).max(dim=-1).values
            pred_cat = pred_cat.argmax(dim=-1)

            for img_id, cats, bboxes, scores in zip(img_ids, pred_cat, pred_bbox, pred_score):
                for cat, bbox, score in zip(cats, bboxes, scores):
                    if cat == 0:
                        continue
                    img_info = coco_val.coco.imgs[img_id]
                    w, h = img_info['width'], img_info['height']
                    bbox *= torch.tensor([w, h, w, h], dtype=torch.float32, device=device)
                    bbox = box_convert(bbox, 'cxcywh', 'xywh')
                    results.append({
                        'image_id': img_id,
                        'category_id': cat.tolist(),
                        'bbox': bbox.tolist(),
                        'score': score.item()
                    })

    with open(results_path, 'w') as f:
        json.dump(results, f)


def eval():
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco = COCO(annFile)
    cocoDt = coco.loadRes(results_path)
    cocoEval = COCOeval(coco, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

def show_summary():
    summary(model, input_size=(eval_batch_size, 3, 512, 512))


if __name__ == "__main__":
    # Parameters
    eval_batch_size = 16
    num_workers = 3
    num_classes = 91
    num_queries = 100
    hidden_dim = 128
    nheads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dropout = 0.1
    annFile = 'coco/annotations/instances_val2017.json'
    results_path = 'results.json'

    # Device
    batt = psutil.sensors_battery()
    if torch.cuda.is_available() and (batt is None or batt.power_plugged is True) or True:
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    print(device)

    # Load model
    model = DETR(num_classes, hidden_dim, nheads, num_encoder_layers, num_decoder_layers, num_queries, dropout)
    model.to(device)
    path = input('Input model path: ')
    if path != '':
        load_checkpoint = torch.load(path)
        model.load_state_dict(load_checkpoint['model'])
        print(f'Loaded model from {path}')
    else:
        load_checkpoint = None

    torch.multiprocessing.set_start_method('forkserver')
    show_summary()
    get_result()
    eval()
