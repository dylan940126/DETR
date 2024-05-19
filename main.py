import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.backends import cudnn
from MyCocoDataset import CocoDataset, collate_fn
from MyDETR import DETR
from MyHungarianLoss import HungarianLoss
from MyVisualizer import draw_bbox
import matplotlib.pyplot as plt


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
    hungarian_loss = HungarianLoss(bbox_weight=6.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train model
    for round in range(epoch):
        print(f'Epoch {round + 1}')
        for i, (images, target) in enumerate(data_loader):  # (48, 3, 128, 128), ((48, 100), (48, 100, 4))
            optimizer.zero_grad()
            predict = model(images)  # ((48, 100, 92), (48, 100, 4))
            bat_loss = hungarian_loss(predict, target)
            bat_loss = bat_loss.mean()
            bat_loss.backward()
            print(f'Batch {i + 1} / {len(data_loader)}, Loss: {bat_loss.item()}')
            optimizer.step()

            if i % 20 == 0:
                # Visualize
                image = draw_bbox(images[0], predict[0][0].argmax(-1), predict[1][0], color='green')
                image = draw_bbox(image, target[0][0], target[1][0], color='red')
                plt.imshow(image.permute(1, 2, 0))
                plt.show()
        if round % 5 == 0:
            torch.save(model.state_dict(), f'checkpoint_{round}.pth')


if __name__ == "__main__":
    # Parameters
    batch_size = 48
    num_workers = 4
    num_classes = 91
    num_queries = 100
    hidden_dim = 256
    nheads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    epoch = 100

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    # Load model
    model = DETR(num_classes, hidden_dim, nheads, num_encoder_layers, num_decoder_layers, num_queries)
    model.to(device)
    path = input('Input model path: ')
    if path != '':
        model.load_state_dict(torch.load(path))

    torch.multiprocessing.set_start_method('forkserver')
    train(epoch)
