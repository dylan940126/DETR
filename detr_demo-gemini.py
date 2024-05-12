import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from CocoDataset import CocoDataset, collate_fn
from DETR import DETR

batch_size = 48
num_workers = 0
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

data_loader = DataLoader(coco_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

# Load model
model = DETR(num_classes, hidden_dim, nheads, num_encoder_layers, num_decoder_layers, num_queries)
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
