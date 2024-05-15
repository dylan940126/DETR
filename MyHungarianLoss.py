from concurrent.futures import ThreadPoolExecutor

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn as nn
from torchvision.ops import complete_box_iou_loss


class HungarianLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.classLoss = nn.CrossEntropyLoss(reduction='none')
        self.bboxLoss = complete_box_iou_loss

    def assign_loss(self, loss_mat):
        assign = linear_sum_assignment(loss_mat.detach().cpu().numpy())
        return assign

    def forward(self, predict, target):
        bat_pred_cat, bat_pred_bbox = predict  # (48, 100, 92), (48, 100, 4)
        bat_tar_cat, bat_tar_bbox = target  # (48, 100), (48, 100, 4)
        # Compute loss matrix for matchs between predictions and targets
        # targets: loss_mat[i, j] = loss between pred[i] and tar[j]
        num_queries = bat_pred_cat.shape[1]
        bat_pred_cat = bat_pred_cat.unsqueeze(2).expand(-1, -1, num_queries, -1)  # (48, 100, 100, 92)
        bat_tar_cat = bat_tar_cat.unsqueeze(1).expand(-1, num_queries, -1)  # (48, 100, 100)
        # x, y, w, h -> x1, y1, x2, y2
        bat_pred_bbox = bat_pred_bbox.clone()
        bat_tar_bbox = bat_tar_bbox.clone()
        bat_pred_bbox[:, :, 2:] += bat_pred_bbox[:, :, :2]
        bat_tar_bbox[:, :, 2:] += bat_tar_bbox[:, :, :2]
        bat_pred_bbox = bat_pred_bbox.unsqueeze(2).expand(-1, -1, num_queries, -1)  # (48, 100, 100, 4)
        bat_tar_bbox = bat_tar_bbox.unsqueeze(1).expand(-1, num_queries, -1, -1)  # (48, 100, 100, 4)
        mask = bat_tar_cat.detach() != 0
        bat_loss_cat = self.classLoss(bat_pred_cat.permute(0, 3, 1, 2), bat_tar_cat)
        bat_loss_bbox = self.bboxLoss(bat_pred_bbox, bat_tar_bbox) * mask  # (48, 100, 100)
        bat_loss_mat = bat_loss_bbox
        # Hungarian algorithm to match predictions and targets
        with ThreadPoolExecutor() as executor:
            res = executor.map(self.assign_loss, bat_loss_mat)
        res = list(res)
        bat_loss = torch.stack([bat_loss_mat[i, res[i][0], res[i][1]] for i in range(len(res))])
        print("cat Loss:", torch.stack([bat_loss_cat[i, res[i][0], res[i][1]] for i in range(len(res))]).mean().item())
        print("bbox Loss:",
              torch.stack([bat_loss_bbox[i, res[i][0], res[i][1]] for i in range(len(res))]).mean().item())
        return bat_loss  # (48, 100)
