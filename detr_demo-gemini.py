import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import CocoDetection
from torchvision.models import resnet50, ResNet50_Weights
from auction_lap import auction_lap


class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim, nheads,
                 num_encoder_layers, num_decoder_layers, device):
        super().__init__()
        # We take only convolutional layers from ResNet-50 model
        self.backbone = nn.Sequential(*list(resnet50(weights=ResNet50_Weights.DEFAULT).children())[:-2])
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        self.transformer = nn.Transformer(hidden_dim, nheads,
                                          num_encoder_layers, num_decoder_layers)
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.device = device
        self.to(device)

    def forward(self, inputs):
        inputs = inputs.to(self.device)
        x = self.backbone(inputs)
        h = self.conv(x)
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1)
        pos = pos.flatten(0, 1)
        pos = pos.unsqueeze(1).repeat(1, inputs.shape[0], 1)
        h = self.transformer(pos + h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1).repeat(1, inputs.shape[0], 1))
        return self.linear_class(h).sigmoid().permute(1, 0, 2), self.linear_bbox(h).permute(1, 0, 2)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

# Load COCO dataset
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

coco_train = CocoDetection(root='coco/images', annFile='coco/annotations/instances_val2017.json', transform=transform)

data_loader = DataLoader(coco_train, batch_size=1, shuffle=True)

# Load model
model = DETR(91, 256, 8, 6, 6, device)
model.train()

# Load optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-2)

# Train model
for images, targets in data_loader:
    try:
        optimizer.zero_grad()
        classes_p, boxes_p = model(images)  # (1,100,92), (1,100,4)
        for class_preds, box_preds in zip(classes_p, boxes_p):
            class_targs = torch.cat([target['category_id'] for target in targets]).to(device)
            box_targs = torch.stack([torch.cat(target['bbox']) for target in targets]).to(device)
            loss_mat = [[F.l1_loss(box_pred, box_targ) + (-torch.log(class_pred[class_targ]))
                         for class_pred, box_pred in zip(class_preds, box_preds)]
                        for class_targ, box_targ in zip(class_targs, box_targs)]
            loss_mat = torch.stack([torch.stack(row) for row in loss_mat]).float()
            assign = auction_lap(loss_mat) # (5,)
            loss = torch.sum(loss_mat[torch.arange(len(assign)), assign])
            print(loss)
            loss.backward()
            optimizer.step()
    except Exception as e:
        print(e)
        continue
