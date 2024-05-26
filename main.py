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


def train(epoch=1):
    # Load COCO dataset
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    coco_train = CocoDataset(root='coco/images', annFile='coco/annotations/instances_val2017.json',
                             transform=transform, category=[], num_queries=num_queries)
    train_loader = DataLoader(coco_train, batch_size=train_batch_size, shuffle=True, num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=True)
    coco_val = CocoDataset(root='coco/images', annFile='coco/annotations/instances_val2017.json',
                           transform=transform, category=[], num_queries=num_queries)
    val_loader = DataLoader(coco_val, batch_size=val_batch_size, shuffle=False, num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=True)

    # Loss function and optimizer
    hungarian_loss = HungarianLoss(cost_iou=cost_iou, cost_l1=cost_l1, cost_cat=cost_cat, loss_cat=loss_cat)
    # set backbone lr=1e-5, other lr=1e-4
    params = list(filter(lambda kv: 'backbone' in kv[0], model.named_parameters()))
    base_params = list(filter(lambda kv: 'backbone' not in kv[0], model.named_parameters()))
    optimizer = optim.AdamW([
        {'params': [p for n, p in base_params]},
        {'params': [p for n, p in params], 'lr': lr_backbone},
    ], lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
    for i in range(start_epoch):
        scheduler.step()

    # Tensorboard
    writer = SummaryWriter(log_dir='logs')
    loss_list = []
    global_step = 0
    loss_global_step = 0

    # Visualizer
    visualizer = MyVisualizer(coco_train.coco.cats)

    # Train model
    for round in range(start_epoch, epoch):
        print(f'Epoch: {round}, lr: {scheduler.get_last_lr()}')
        # Training
        model.train()
        loss_list.clear()
        for i, (images, target) in enumerate(tqdm(train_loader)):
            # preprocess
            images = images.to(device)  # (B, 3, 512, 512)
            targ_cat = target[0].to(device)  # (B, 100)
            targ_bbox = target[1].to(device)  # (B, 100, 4)
            optimizer.zero_grad()
            pred_cat, pred_bbox = model(images)  # (B, 100, 91), (B, 100, 4)
            bat_loss, assign = hungarian_loss(pred_cat, pred_bbox, targ_cat, targ_bbox)
            bat_loss.backward()
            writer.add_scalar('Loss', bat_loss.item(), loss_global_step)
            loss_global_step += 1
            loss_list.append(bat_loss.item())
            optimizer.step()

        torch.save(model.state_dict(), f'checkpoint_{round}.pth')
        writer.add_scalar('Train Loss', sum(loss_list) / len(loss_list), round)
        print(f'Loss: {sum(loss_list) / len(loss_list)}')
        # Validation
        model.eval()
        loss_list.clear()
        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(val_loader)):
                images = images.to(device)
                targ_cat = target[0].to(device)
                targ_bbox = target[1].to(device)
                pred_cat, pred_bbox = model(images)  # (B, 100, 91), (B, 100, 4)
                bat_loss, assign = hungarian_loss(pred_cat, pred_bbox, targ_cat, targ_bbox)
                loss_list.append(bat_loss.item())
                # Visualize
                img_pred = visualizer.draw_bbox(images[0], pred_cat[0, assign[0]].argmax(dim=-1),
                                                pred_bbox[0, assign[0]], mask_cat=targ_cat[0], color='green')
                img_pred = visualizer.draw_bbox(img_pred, pred_cat[0].argmax(dim=-1), pred_bbox[0],
                                                color='blue')
                writer.add_image('Predict', img_pred, global_step)
                img_targ = visualizer.draw_bbox(images[0], targ_cat[0], targ_bbox[0], color='red')
                writer.add_image('Target', img_targ, global_step)
                global_step += 1
            writer.add_scalar('Val Loss', sum(loss_list) / len(loss_list), round)
            print(f'Val Loss: {sum(loss_list) / len(loss_list)}')

        # Update learning rate
        scheduler.step()
    writer.close()


if __name__ == "__main__":
    # Parameters
    train_batch_size = 4
    val_batch_size = 48
    num_workers = 2
    num_classes = 91
    num_queries = 100
    hidden_dim = 128
    nheads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dropout = 0
    epoch = 400
    cost_iou = 2.0
    cost_l1 = 5.0
    cost_cat = 2.0
    loss_cat = 10.0
    lr = 1e-4
    lr_backbone = 1e-5
    step_size = 200

    # Device
    batt = psutil.sensors_battery()
    if torch.cuda.is_available() and (batt is None or batt.power_plugged is True):
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    print(device)

    # Load model
    model = DETR(num_classes, hidden_dim, nheads, num_encoder_layers, num_decoder_layers, num_queries, dropout)
    model.to(device)
    path = input('Input model path: ')
    start_epoch = 0
    if path != '':
        model.load_state_dict(torch.load(path))
        start_epoch = int(path.split('_')[-1].split('.')[0]) + 1
        print(f'Loaded model from {path} and start at epoch {start_epoch}')

    torch.multiprocessing.set_start_method('forkserver')
    train(epoch)
