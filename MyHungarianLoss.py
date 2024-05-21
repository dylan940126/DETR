from concurrent.futures import ThreadPoolExecutor

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn as nn
from torchvision.ops import complete_box_iou_loss
from torchvision.ops import box_convert


class HungarianLoss(nn.Module):
    def __init__(self, bbox_weight=1.0, bbox_format='cxcywh', mask=True):
        """
        Hungarian Loss for DETR

        :param bbox_weight: weight for bbox loss, default is 1.0
        :param bbox_format: format of bbox, default is 'cxcywh'
        :param mask: mask bbox loss for no object, default is True
        """
        super().__init__()
        self.classLoss = nn.CrossEntropyLoss(reduction='none')
        self.bboxLoss = complete_box_iou_loss
        self.weight = bbox_weight
        self.bbox_format = bbox_format
        self.mask = mask

    @staticmethod
    def _assign(loss_mat):
        """
        Hungarian algorithm to match predictions and targets

        No batch dimension is allowed in this function.
        :param loss_mat: loss matrix (N, M)
        :return: assignment
        """
        row_ind, col_ind = linear_sum_assignment(loss_mat.detach().cpu().numpy())
        return col_ind

    def _cat_cost(self, targ_cat: torch.Tensor, pred_cat: torch.Tensor):
        """
        Compute loss matrix for category

        :param targ_cat: targets of category (B, N)
        :param pred_cat: predictions of category (B, M, C)
        :return: loss matrix (B, N, M)
        """
        return torch.stack([pred_cat[i, :, targ_cat[i]] for i in range(pred_cat.shape[0])]).mT  # (B, N, M)

    def _bbox_cost(self, targ_bbox: torch.Tensor, pred_bbox: torch.Tensor, targ_cat=None):
        """
        Compute loss matrix for bounding boxes

        :param targ_bbox: targets of bounding boxes (B, N, 4)
        :param pred_bbox: predictions of bounding boxes (B, M, 4)
        :param targ_cat: targets of category (B, N) for mask, default is None
        :return: loss matrix (B, N, M)
        """
        # prepossessing
        n, m = targ_bbox.shape[1], pred_bbox.shape[1]
        targ_bbox = box_convert(targ_bbox, self.bbox_format, 'xyxy')  # (B, N, 4)
        pred_bbox = box_convert(pred_bbox, self.bbox_format, 'xyxy')  # (B, M, 4)
        # masked bbox cost/loss (B, N, M)
        pred_bbox = pred_bbox.unsqueeze(1).expand(-1, n, -1, -1)  # (B, N, M, 4)
        targ_bbox = targ_bbox.unsqueeze(2).expand(-1, -1, m, -1)  # (B, N, M, 4)
        cost_bbox = self.bboxLoss(pred_bbox, targ_bbox)  # (B, N, M)
        if targ_cat is not None:
            mask = targ_cat != 0  # (B, N)
            mask = mask.unsqueeze(2).expand_as(cost_bbox)  # (B, N, M)
            cost_bbox *= mask
        return cost_bbox

    def _bbox_loss(self, targ_bbox, pred_bbox, targ_cat=None):
        """
        Compute final loss according to assignment

        :param targ_bbox: targets of bounding boxes (B, N, 4)
        :param pred_bbox: predictions of bounding boxes (B, M, 4)
        :param targ_cat: targets of category (B, N) for mask, default is None
        :return: final loss (B,)
        """
        pred_bbox = pred_bbox.flatten(0, 1)  # (B * N, 4)
        targ_bbox = targ_bbox.flatten(0, 1)  # (B * N, 4)
        loss_bbox = self.bboxLoss(pred_bbox, targ_bbox)  # (B * N,)
        if targ_cat is not None:
            mask = targ_cat.flatten() != 0  # (B * N,)
            loss_bbox = loss_bbox[mask]  # (<=B * N,)
        return loss_bbox

    def _cat_loss(self, targ_cat, pred_cat):
        """
        Compute final loss according to assignment

        :param targ_cat: targets category (B, N)
        :param pred_cat: predictions category (B, M, C)
        :return: final loss (B,)
        """
        pred_cat = pred_cat.flatten(0, 1)  # (B * N, C)
        targ_cat = targ_cat.flatten()  # (B * N,)
        loss_cat = self.classLoss(pred_cat, targ_cat)  # (B * N,)
        return loss_cat

    def forward(self, targ_cat, targ_bbox, pred_cat, pred_bbox):
        """
        Calculate cost matrix, assign predictions to targets, and compute loss.

        :param targ_cat: targets category (B, N)
        :param targ_bbox: targets bounding boxes (B, N, 4)
        :param pred_cat: predictions category (B, M, C)
        :param pred_bbox: predictions bounding boxes (B, M, 4)
        :return: final loss
        """
        # compute cost matrix
        cat_cost = self._cat_cost(targ_cat, pred_cat)  # (B, N, M)
        if self.mask:
            bbox_cost = self._bbox_cost(targ_bbox, pred_bbox, targ_cat)  # (B, N, M)
        else:
            bbox_cost = self._bbox_cost(targ_bbox, pred_bbox)
        cost_mat = cat_cost + bbox_cost * self.weight  # (B, N, M)

        # Hungarian algorithm to match predictions and targets
        with ThreadPoolExecutor() as executor:
            assign = executor.map(self._assign, cost_mat)
        assign = list(assign)
        assign = torch.tensor(assign)  # (B, N)

        # rearrange predictions according to assignment
        pred_cat = pred_cat[torch.arange(pred_cat.shape[0]).unsqueeze(-1), assign]  # (B, N, C)
        pred_bbox = pred_bbox[torch.arange(pred_bbox.shape[0]).unsqueeze(-1), assign]  # (B, N, 4)

        # compute loss
        loss_cat = self._cat_loss(targ_cat, pred_cat).mean()
        if self.mask:
            loss_bbox = self._bbox_loss(targ_bbox, pred_bbox, targ_cat).mean()
        else:
            loss_bbox = self._bbox_loss(targ_bbox, pred_bbox).mean()
        loss = loss_cat + loss_bbox * self.weight

        print("Cat Loss: ", loss_cat.item())
        print("BBox Loss: ", loss_bbox.item())
        return loss, assign
