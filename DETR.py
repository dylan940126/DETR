import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights


class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim, nheads,
                 num_encoder_layers, num_decoder_layers, num_queries):
        super().__init__()
        # We take only convolutional layers from ResNet-50 model
        self.backbone = nn.Sequential(*list(resnet50(weights=ResNet50_Weights.DEFAULT).children())[:-2])
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        self.transformer = nn.Transformer(hidden_dim, nheads,
                                          num_encoder_layers, num_decoder_layers)
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        self.query_pos = nn.Parameter(torch.rand(num_queries, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        x = self.backbone(inputs)
        h = self.conv(x)
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1)
        pos = pos.flatten(0, 1)
        pos = pos.unsqueeze(1).repeat(1, batch_size, 1)
        h = self.transformer(pos + h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1).repeat(1, batch_size, 1))
        return self.linear_class(h).permute(1, 0, 2), self.linear_bbox(h).permute(1, 0, 2)
