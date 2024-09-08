import torch
import psutil
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.backends import cudnn
from MyCocoDataset import CocoDataset, collate_fn
from MyDETR import DETR
from MyHungarianLoss import HungarianLoss
from torch.utils.tensorboard import SummaryWriter
from MyVisualizer import MyVisualizer
from tqdm import tqdm
import datetime


def train():
    # Load COCO dataset
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    coco_val = CocoDataset(root='coco/images', annFile='coco/annotations/instances_val2017.json',
                           transform=transform, category=[], num_queries=num_queries)
    val_loader = DataLoader(coco_val, batch_size=val_batch_size, shuffle=True, num_workers=num_workers,
                            collate_fn=collate_fn)

    # Visualizer
    visualizer = MyVisualizer(coco_val.coco.cats)

    # Tensorboard
    log_dir = 'logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)

    # Validation
    model.eval()
    with torch.no_grad():
        val_loader_tqdm = tqdm(val_loader)
        for i, (images, target) in enumerate(val_loader_tqdm):
            images = images[1].to(device)
            targ_cat = target[0].to(device)
            targ_bbox = target[1].to(device)
            pred_cat, pred_bbox, intermediate = model(images)
            show = torch.zeros((24, 3, 512, 512), dtype=torch.float32, device=device)
            show[:, 0] = transforms.Resize((512, 512))(intermediate[0, :24])
            img_pred = visualizer.draw_bbox(images[0], pred_cat[0].argmax(dim=-1), pred_bbox[0],
                                            color='blue')
            writer.add_images('Intermediate', images[0] * 0.8 + show * 0.8, i)
            writer.add_image('Image', img_pred, i)
    writer.close()


if __name__ == "__main__":
    # Parameters
    val_batch_size = 1
    num_workers = 0
    num_classes = 91
    num_queries = 100
    hidden_dim = 128
    nheads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dropout = 0

    # Device
    batt = psutil.sensors_battery()
    if torch.cuda.is_available() and (batt is None or batt.power_plugged is True) or True:
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    print(device)

    # Load model
    model = DETR(num_classes, hidden_dim, nheads, num_encoder_layers, num_decoder_layers, num_queries, dropout, True)
    model.to(device)
    path = input('Input model path: ')
    if path != '':
        load_checkpoint = torch.load(path)
        model.load_state_dict(load_checkpoint['model'])
        print(f'Loaded model from {path}')
    else:
        load_checkpoint = None

    torch.multiprocessing.set_start_method('forkserver')
    train()
