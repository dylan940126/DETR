from concurrent.futures import ThreadPoolExecutor

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn as nn
from torchvision.ops import complete_box_iou_loss
from torchvision.ops import box_convert


class HungarianLoss(nn.Module):
    def __init__(self, bbox_weight=1.0):
        super().__init__()
        self.classLoss = nn.CrossEntropyLoss(reduction='none')
        self.bboxLoss = complete_box_iou_loss
        self.weight = bbox_weight

    def assign_loss(self, loss_mat):
        assign = linear_sum_assignment(loss_mat.detach().cpu().numpy())
        return assign

    def forward(self, predict, target):
        bat_pred_cat, bat_pred_bbox = predict  # (48, 100, 92), (48, 100, 4)
        bat_tar_cat, bat_tar_bbox = target  # (48, 100), (48, 100, 4)
        # Compute loss matrix for matchs between predictions and targets
        # targets: loss_mat[i, j] = loss between pred[i] and tar[j]
        num_queries = bat_pred_cat.shape[1]

        # cat loss
        bat_pred_cat = bat_pred_cat.unsqueeze(2).expand(-1, -1, num_queries, -1)  # (48, 100, 100, 92)
        bat_tar_cat = bat_tar_cat.unsqueeze(1).expand(-1, num_queries, -1)  # (48, 100, 100)
        bat_loss_cat = self.classLoss(bat_pred_cat.permute(0, 3, 1, 2), bat_tar_cat)
        # masked bbox loss
        bat_pred_bbox = box_convert(bat_pred_bbox, 'cxcywh', 'xyxy')  # (48, 100, 4)
        bat_tar_bbox = box_convert(bat_tar_bbox, 'cxcywh', 'xyxy')  # (48, 100, 4)
        bat_pred_bbox = bat_pred_bbox.unsqueeze(2).expand(-1, -1, num_queries, -1)  # (48, 100, 100, 4)
        bat_tar_bbox = bat_tar_bbox.unsqueeze(1).expand(-1, num_queries, -1, -1)  # (48, 100, 100, 4)
        bat_loss_bbox = self.bboxLoss(bat_pred_bbox, bat_tar_bbox) * self.weight  # (48, 100, 100)
        mask = bat_tar_cat.detach() != 0
        bat_loss_bbox *= mask
        # Combine cat and bbox loss
        bat_loss_mat = bat_loss_bbox
        # Hungarian algorithm to match predictions and targets
        with ThreadPoolExecutor() as executor:
            res = executor.map(self.assign_loss, bat_loss_mat)
        res = list(res)
        # Compute final loss
        loss_cat = torch.stack([bat_loss_cat[i, res[i][0], res[i][1]] for i in range(len(res))]).mean(dim=-1)
        loss_bbox = torch.stack([bat_loss_bbox[i, res[i][0], res[i][1]] for i in range(len(res))])
        loss_bbox = loss_bbox.sum(dim=-1) / ((loss_bbox != 0).sum(dim=-1) + 1e-7)
        bat_loss = loss_cat + loss_bbox
        print("Cat Loss: ", loss_cat.mean().item())
        print("BBox Loss: ", loss_bbox.mean().item())
        return bat_loss
