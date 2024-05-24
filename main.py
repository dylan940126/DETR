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


def train(epoch=1):
    # Load COCO dataset
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    coco_train = CocoDataset(root='coco/images', annFile='coco/annotations/instances_val2017.json',
                             transform=transform, category=[], num_queries=num_queries)
    data_loader = DataLoader(coco_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                             collate_fn=collate_fn, pin_memory=True)

    # Set model to training mode
    model.train()

    # Loss function and optimizer
    hungarian_loss = HungarianLoss(cost_iou=cost_iou, cost_l1=cost_l1, cost_cat=cost_cat, loss_cat=loss_cat)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Tensorboard
    writer = SummaryWriter(log_dir='logs')
    global_step = 0

    # Visualizer
    visualizer = MyVisualizer(coco_train.coco.cats)

    # Train model
    for round in range(epoch):
        print(f'Epoch {round}')
        for i, (images, target) in enumerate(data_loader):
            # preprocess
            images = images.to(device)  # (B, 3, 512, 512)
            targ_cat = target[0].to(device)  # (B, 100)
            targ_bbox = target[1].to(device)  # (B, 100, 4)
            optimizer.zero_grad()
            pred_cat, pred_bbox = model(images)  # (48, 100, 91), (48, 100, 4)
            bat_loss, assign = hungarian_loss(pred_cat, pred_bbox, targ_cat, targ_bbox)
            bat_loss.backward()
            writer.add_scalar('Loss', bat_loss.item(), global_step)
            print(f'Batch {i + 1} / {len(data_loader)}, Loss: {bat_loss.item()}')
            optimizer.step()

            if i % 5 == 0:
                # Visualize
                img_pred = visualizer.draw_bbox(images[0], pred_cat[0].argmax(dim=-1), pred_bbox[0],
                                                color='blue')
                # img_pred = visualizer.draw_bbox(images[0], target[0][0], predict[1][0, assign[0]], color='green')
                writer.add_image('Predict', img_pred, global_step)
                # img_targ = visualizer.draw_bbox(images[0], target[0][0], target[1][0], color='red')
                # writer.add_image('Target', img_targ, global_step)
            global_step += 1
        if round % 1 == 0:
            torch.save(model.state_dict(), f'checkpoint_{round}.pth')
    writer.close()


if __name__ == "__main__":
    # Parameters
    batch_size = 8
    num_workers = 4
    num_classes = 91
    num_queries = 100
    hidden_dim = 128
    nheads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    epoch = 400
    cost_iou = 2.0
    cost_l1 = 5.0
    cost_cat = 1.0
    loss_cat = 1.0

    # Device
    batt = psutil.sensors_battery()
    if torch.cuda.is_available() and (batt is None or batt.power_plugged is True):
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    print(device)

    # Load model
    model = DETR(num_classes, hidden_dim, nheads, num_encoder_layers, num_decoder_layers, num_queries)
    model.to(device)
    path = input('Input model path: ')
    if path != '':
        model.load_state_dict(torch.load(path))

    torch.multiprocessing.set_start_method('forkserver')
    train(epoch)
