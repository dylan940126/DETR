import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from CocoDataset import CocoDataset, collate_fn
from DETR import DETR
from auction_lap import auction_lap

batch_size = 16
num_workers = 2
num_classes = 91
num_queries = 100
hidden_dim = 256
nheads = 8
num_encoder_layers = 6
num_decoder_layers = 6

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load COCO dataset
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

coco_train = CocoDataset(root='coco/images', annFile='coco/annotations/instances_val2017.json',
                         transform=transform, device=device)

data_loader = DataLoader(coco_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                         collate_fn=collate_fn)

# Load model
model = DETR(num_classes, hidden_dim, nheads, num_encoder_layers, num_decoder_layers, num_queries)
model.train()
model.to(device)

# Loss function
classLoss = nn.CrossEntropyLoss(reduction='none')
bboxLoss = nn.SmoothL1Loss(reduction='none')

# Load optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def hungarianLoss(bat_pred_cat, bat_tar_cat, bat_pred_bbox, bat_tar_bbox):
    # Compute loss matrix for matchs between predictions and targets: loss_mat[i, j] = loss between pred[i] and tar[j]
    bat_pred_cat = bat_pred_cat.unsqueeze(2).expand(-1, -1, num_queries, -1)  # (48, 100, 100, 92)
    bat_tar_cat = bat_tar_cat.unsqueeze(1).expand(-1, num_queries, -1)  # (48, 100, 100)
    bat_pred_bbox = bat_pred_bbox.unsqueeze(2).expand(-1, -1, num_queries, -1)  # (48, 100, 100, 4)
    bat_tar_bbox = bat_tar_bbox.unsqueeze(1).expand(-1, num_queries, -1, -1)  # (48, 100, 100, 4)
    mask = bat_tar_cat.detach() != 0
    bat_loss_mat = classLoss(bat_pred_cat.permute(0, 3, 1, 2), bat_tar_cat) + \
                   bboxLoss(bat_pred_bbox, bat_tar_bbox).sum(dim=-1) * mask  # (48, 100, 100)
    # Hungarian algorithm to match predictions and targets
    bat_loss = torch.zeros((batch_size, num_queries), device=device)
    for i, loss_mat in enumerate(bat_loss_mat):
        assign = auction_lap(-loss_mat.detach())
        bat_loss[i] = loss_mat[torch.arange(num_queries), assign]
    return bat_loss


def main():
    # Train model
    for images, bat_tar_cat, bat_tar_bbox in data_loader:  # (48, 3, 128, 128), (48, 100), (48, 100, 4)
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        bat_pred_cat, bat_pred_bbox = model(images)  # (48, 100, 92), (48, 100, 4)

        bat_loss = hungarianLoss(bat_pred_cat, bat_tar_cat, bat_pred_bbox, bat_tar_bbox)

        # Backward pass
        bat_loss = bat_loss.mean()
        bat_loss.backward()
        print(bat_loss.item())
        optimizer.step()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')  # good solution !!!!
    main()
