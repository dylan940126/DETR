import torch
import psutil
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.backends import cudnn
from MyCocoDataset import CocoDataset, collate_fn
from MyDETR import DETR
from MyHungarianLoss import HungarianLoss
from MyVisualizer import plot


def train(epoch=1):
    # Load COCO dataset
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    coco_train = CocoDataset(root='coco/images', annFile='coco/annotations/instances_val2017.json',
                             transform=transform, device=device, category=[], num_queries=num_queries)
    data_loader = DataLoader(coco_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                             collate_fn=collate_fn)

    # Set model to training mode
    model.train()

    # Loss function and optimizer
    hungarian_loss = HungarianLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train model
    for _ in range(epoch):
        print(f'Epoch {_ + 1}')
        for i, (images, target) in enumerate(data_loader):  # (48, 3, 128, 128), ((48, 100), (48, 100, 4))
            optimizer.zero_grad()
            predict = model(images)  # ((48, 100, 92), (48, 100, 4))
            bat_loss, assign = hungarian_loss(target[0], target[1], predict[0], predict[1])
            bat_loss.backward()
            print(f'Batch {i + 1} / {len(data_loader)}, Loss: {bat_loss.item()}')
            optimizer.step()

            if i % 10 == 0:
                plot(images, predict[0].argmax(-1), predict[1], device=device)
        if _ % 5 == 0:
            torch.save(model.state_dict(), f'checkpoint_{_}.pth')


if __name__ == "__main__":
    # Parameters
    batch_size = 2
    num_workers = 4
    num_classes = 91
    num_queries = 100
    hidden_dim = 256
    nheads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6

    # Device
    batt = psutil.sensors_battery()
    if torch.cuda.is_available() and (batt is None or batt.power_plugged is True):
        device = torch.device('cuda')
        cudnn.benchmark = True
        print(device)
    else:
        device = torch.device('cpu')
        print(device)

    # Load model
    model = DETR(num_classes, hidden_dim, nheads, num_encoder_layers, num_decoder_layers, num_queries)
    model.to(device)
    path = input('Input model path: ')
    if path != '':
        model.load_state_dict(torch.load(path))

    torch.multiprocessing.set_start_method('forkserver')  # good solution !!!!
    # for _ in range(100):
    train(epoch=100)
