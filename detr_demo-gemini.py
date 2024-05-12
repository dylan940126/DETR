import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import CocoDetection
from torchvision.models import resnet50, ResNet50_Weights
from auction_lap import auction_lap


from CocoDataset import CocoDataset, collate_fn
from DETR import DETR

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

# Load COCO dataset
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

coco_train = CocoDataset(root='coco/images', annFile='coco/annotations/instances_val2017.json',
                         transform=transform, device=device)

data_loader = DataLoader(coco_train, batch_size=48, shuffle=True, num_workers=0, collate_fn=collate_fn)

# Load model
model = DETR(91, 256, 8, 6, 6)
model.train()
model.to(device)

# Load optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-2)

# Train model
for images, targets in data_loader:
    try:
        optimizer.zero_grad()
        images = images.to(device)
        classes_p, boxes_p = model(images)  # (1,100,92), (1,100,4)
        for class_preds, box_preds in zip(classes_p, boxes_p):
            loss_mat = torch.zeros((len(targets), 100), device=device)
            for i, obj in enumerate(targets):
                loss_mat[i] = -torch.log(class_preds.T[obj['category_id']])
            for i, obj in enumerate(targets):
                for j, box_pred in enumerate(box_preds):
                    loss_mat[i][j] += F.smooth_l1_loss(box_pred, torch.cat(obj['bbox']).to(device=device).float())
            assign = auction_lap(loss_mat)  # (5,)
            loss = torch.sum(loss_mat[torch.arange(len(assign)), assign])
            loss += torch.sum(-torch.log(class_preds[[i for i in range(100) if i not in assign]].T[0]))
            print(loss)
            loss.backward()
            optimizer.step()
    except Exception as e:
        print(e)
        continue
