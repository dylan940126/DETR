from concurrent.futures import ThreadPoolExecutor

import numpy
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn as nn
from torchvision.ops import generalized_box_iou_loss
from torchvision.ops import box_convert


class MyBBoxLoss(nn.Module):
    def __init__(self, cost_iou=1.0, cost_l1=1.0):
        super().__init__()
        self.loss_iou = generalized_box_iou_loss
        self.loss_l1 = nn.L1Loss(reduction='none')
        self.cost_iou = cost_iou
        self.cost_l1 = cost_l1

    def forward(self, pred_bbox, targ_bbox):
        """
        Compute loss matrix for bounding boxes

        :param pred_bbox: predictions of bounding boxes (B, M, 4)
        :param targ_bbox: targets of bounding boxes (B, N, 4)
        :return: loss matrix (B, N, M)
        """
        loss1 = self.loss_iou(pred_bbox, targ_bbox) * self.cost_iou
        loss2 = self.loss_l1(pred_bbox, targ_bbox).mean(-1) * self.cost_l1
        # print(f"Loss1: {loss1.mean().item():.4f} Loss2: {loss2.mean().item():.4f}")
        return loss1 + loss2


class HungarianLoss(nn.Module):
    def __init__(self, bbox_format='cxcywh', cost_iou=2.0, cost_l1=5.0, cost_cat=1.0, loss_cat=1.0):
        """
        Hungarian Loss for DETR

        :param cost_weight: weight for cost matrix (cat_cost, bbox_cost)
        :param loss_weight: weight for final loss (cat_loss, bbox_loss)
        :param bbox_format: format of bbox, default is 'cxcywh'
        """
        super().__init__()
        self.classLoss = nn.CrossEntropyLoss(reduction='none')
        self.bboxLoss = MyBBoxLoss(cost_iou, cost_l1)
        self.bbox_format = bbox_format
        self.cost_cat = cost_cat
        self.loss_cat = loss_cat

    @staticmethod
    def _assign(loss_mat):
        """
        Hungarian algorithm to match predictions and targets

        No batch dimension is allowed in this function.
        :param loss_mat: loss matrix (N, M)
        :return: assignment
        """
        row_ind, col_ind = linear_sum_assignment(loss_mat.detach().cpu().numpy())
        return torch.from_numpy(col_ind)

    def _cat_cost(self, targ_cat: torch.Tensor, pred_cat: torch.Tensor):
        """
        Compute loss matrix for category

        cat_cost[b][n][m] == -pred_cat[b, m, targ_cat[b][n]]

        :param targ_cat: targets of category (B, N)
        :param pred_cat: predictions of category (B, M, C)
        :return: loss matrix (B, N, M)
        """
        pred_cat = pred_cat.softmax(dim=-1)  # (B, M, C)
        cat_cost = 1 - torch.stack([pred_cat[i, :, targ_cat[i]] for i in range(pred_cat.shape[0])]).mT  # (B, N, M)
        return cat_cost * self.cost_cat

    def _bbox_cost(self, targ_bbox: torch.Tensor, pred_bbox: torch.Tensor, targ_cat: torch.Tensor):
        """
        Compute loss matrix for bounding boxes

        cost_bbox[b, n, m] == self.bboxLoss(pred_bbox[b, m], targ_bbox[b, n]) or
        cost_bbox[b, n, m] == True if targ_cat[b, n] == 0


        :param targ_bbox: targets of bounding boxes (B, N, 4)
        :param pred_bbox: predictions of bounding boxes (B, M, 4)
        :param targ_cat: targets of category (B, N) for mask
        :return: loss matrix (B, N, M)
        """
        # prepossessing
        n, m = targ_bbox.shape[1], pred_bbox.shape[1]
        targ_bbox = box_convert(targ_bbox, self.bbox_format, 'xyxy')  # (B, N, 4)
        pred_bbox = box_convert(pred_bbox, self.bbox_format, 'xyxy')  # (B, M, 4)
        # masked bbox cost (B, N, M)
        pred_bbox = pred_bbox.unsqueeze(1).expand(-1, n, -1, -1)  # (B, N, M, 4)
        targ_bbox = targ_bbox.unsqueeze(2).expand(-1, -1, m, -1)  # (B, N, M, 4)
        cost_bbox = self.bboxLoss(pred_bbox, targ_bbox)  # (B, N, M)
        mask = (targ_cat != 0).unsqueeze(-1).expand(-1, -1, pred_bbox.shape[2])
        cost_bbox *= mask
        return cost_bbox

    def _cat_loss(self, targ_cat: torch.Tensor, pred_cat: torch.Tensor):
        """
        Compute final loss according to assignment

        :param targ_cat: targets category (B, N)
        :param pred_cat: predictions category (B, N, C)
        :return: final loss (B,)
        """
        loss_cat = self.classLoss(pred_cat.mT, targ_cat)  # (B, N)
        return loss_cat.mean(dim=-1) * self.loss_cat

    def _bbox_loss(self, targ_bbox: torch.Tensor, pred_bbox: torch.Tensor, targ_cat: torch.Tensor):
        """
        Compute final loss according to assignment

        :param targ_bbox: targets of bounding boxes (B, N, 4)
        :param pred_bbox: predictions of bounding boxes (B, M, 4)
        :param targ_cat: targets of category (B, N) for mask
        :return: final loss (B,)
        """
        # prepossessing
        targ_bbox = box_convert(targ_bbox, self.bbox_format, 'xyxy')  # (B, N, 4)
        pred_bbox = box_convert(pred_bbox, self.bbox_format, 'xyxy')  # (B, N, 4)

        # masked bbox loss (B * N,)
        loss_bbox = self.bboxLoss(pred_bbox, targ_bbox)  # (B, N)
        mask = targ_cat != 0  # (B, N)
        loss_bbox[~mask] = 0
        return loss_bbox.sum(dim=-1) / (mask.sum(dim=-1) + 1e-7)

    def forward(self, pred_cat: torch.Tensor, pred_bbox: torch.Tensor, targ_cat: torch.Tensor, targ_bbox: torch.Tensor):
        """
        Calculate cost matrix, assign predictions to targets, and compute loss.

        :param pred_cat: predictions category (B, M, C)
        :param pred_bbox: predictions bounding boxes (B, M, 4)
        :param targ_cat: targets category (B, N)
        :param targ_bbox: targets bounding boxes (B, N, 4)
        :return: final loss
        """
        # compute cost matrix
        with torch.no_grad():
            cat_cost = self._cat_cost(targ_cat, pred_cat)  # (B, N, M)
            # cannot overwrite the original tensor since requires grad
            bbox_cost = self._bbox_cost(targ_bbox, pred_bbox, targ_cat)  # (B, N, M)
            cost_mat = cat_cost + bbox_cost  # (B, N, M)

            # Hungarian algorithm to match predictions and targets
            with ThreadPoolExecutor() as executor:
                assign = executor.map(self._assign, cost_mat)
            assign = list(assign)
            assign = torch.stack(assign)  # (B, N)
            # cat_cost = cat_cost[torch.arange(cat_cost.shape[0]).unsqueeze(-1), assign]  # (B, N)
            # bbox_cost = bbox_cost[torch.arange(bbox_cost.shape[0]).unsqueeze(-1), assign]  # (B, N)
            # print(f"Cat Cost: {cat_cost.mean().item():.4f} BBox Cost: {bbox_cost.mean().item():.4f}")

        # rearrange predictions according to assignment
        pred_cat = pred_cat[torch.arange(pred_cat.shape[0]).unsqueeze(-1), assign]  # (B, N, C)
        pred_bbox = pred_bbox[torch.arange(pred_bbox.shape[0]).unsqueeze(-1), assign]  # (B, N, 4)

        # compute loss
        loss_cat = self._cat_loss(targ_cat, pred_cat)  # (B,)
        loss_bbox = self._bbox_loss(targ_bbox, pred_bbox, targ_cat)  # (B,)
        loss = loss_cat + loss_bbox  # (B,)
        loss = loss.mean()

        # print(f"Cat Loss: {loss_cat.mean().item():.4f} BBox Loss: {loss_bbox.mean().item():.4f}")
        return loss, assign  # (B,), (B, N)
