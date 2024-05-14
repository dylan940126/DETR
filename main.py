import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from CocoDataset import CocoDataset, collate_fn
from DETR import DETR
from scipy.optimize import linear_sum_assignment
from concurrent.futures import ThreadPoolExecutor
from torch.backends import cudnn


class HungarianLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.classLoss = nn.CrossEntropyLoss(reduction='none')
        self.bboxLoss = nn.SmoothL1Loss(reduction='none')

    def assign_loss(self, loss_mat):
        assign = linear_sum_assignment(loss_mat)
        return assign

    def forward(self, predict, target):
        bat_pred_cat, bat_pred_bbox = predict  # (48, 100, 92), (48, 100, 4)
        bat_tar_cat, bat_tar_bbox = target  # (48, 100), (48, 100, 4)
        # Compute loss matrix for matchs between predictions and targets
        # targets: loss_mat[i, j] = loss between pred[i] and tar[j]
        num_queries = bat_pred_cat.shape[1]
        bat_pred_cat = bat_pred_cat.unsqueeze(2).expand(-1, -1, num_queries, -1)  # (48, 100, 100, 92)
        bat_tar_cat = bat_tar_cat.unsqueeze(1).expand(-1, num_queries, -1)  # (48, 100, 100)
        bat_pred_bbox = bat_pred_bbox.unsqueeze(2).expand(-1, -1, num_queries, -1)  # (48, 100, 100, 4)
        bat_tar_bbox = bat_tar_bbox.unsqueeze(1).expand(-1, num_queries, -1, -1)  # (48, 100, 100, 4)
        mask = bat_tar_cat.detach() != 0
        bat_loss_mat = self.classLoss(bat_pred_cat.permute(0, 3, 1, 2), bat_tar_cat) + \
                       self.bboxLoss(bat_pred_bbox, bat_tar_bbox).sum(dim=-1) * mask  # (48, 100, 100)
        # Hungarian algorithm to match predictions and targets
        with ThreadPoolExecutor() as executor:
            res = executor.map(self.assign_loss, bat_loss_mat.detach().cpu().numpy())
        res = list(res)
        bat_loss = torch.stack([bat_loss_mat[i, res[i][0], res[i][1]] for i in range(len(res))])
        return bat_loss  # (48, 100)


def train(epoch=100):
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
            bat_loss = hungarian_loss(predict, target)
            bat_loss = bat_loss.mean()
            bat_loss.backward()
            print(f'Batch {i + 1} / {len(data_loader)}, Loss: {bat_loss.item()}')
            optimizer.step()

            # show the first image and bbox
            if i % 10 == 0:
                import matplotlib.pyplot as plt
                import numpy as np
                img = images[0].permute(1, 2, 0).detach().cpu().numpy()
                plt.imshow(img)
                bbox = predict[1][0].detach().cpu().numpy()
                bbox = bbox[bbox[:, 0] != 0]
                bbox = bbox * 128
                for box in bbox:
                    plt.gca().add_patch(
                        plt.Rectangle((box[0], box[1]), box[2], box[3], fill=False, edgecolor='r', linewidth=1))
                plt.show()
        if _ % 5 == 0:
            torch.save(model.state_dict(), f'checkpoint_{_}.pth')


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

    torch.multiprocessing.set_start_method('forkserver')  # good solution !!!!
    for _ in range(100):
        train()
