import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist

class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


def TverskyLoss(y_true, y_pred, alpha=0.5, beta=0.5, smooth=0.0):
    y_pred = torch.sigmoid(y_pred)

    tp = (y_true * y_pred).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()

    tversky_index = (tp + smooth) / (tp + alpha * fn + beta * fn + smooth)
    return tversky_index


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # y_pred is raw logits; apply sigmoid for binary classification
        y_pred = torch.sigmoid(y_pred)

        #Tversky loss
        tp = (y_true * y_pred).sum()
        fp = ((1 - y_true) * y_pred).sum()
        fn = (y_true * (1 - y_pred)).sum()

        tversky_index = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)

        loss = (1- tversky_index) ** self.gamma
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, smooth=0.0, reduction='mean'):
        """
        alpha: Weighting factor for the class (can be a scalar or a list for multi-class). Up-scale positives
        gamma: Focusing parameter for modulating factor (1-pt), how aggressive to suppress easy exampls. 0=BCE, 2=Focal
        smooth: Label smoothing factor
        reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        # y_pred is raw logits; apply sigmoid for binary classification
        ce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true.float(), reduction='none')
        # ce_loss = F.binary_cross_entropy_with_logits(y_pred.view(-1), y_true.view(-1), reduction='none')
        pt = torch.exp(-ce_loss)

        if isinstance(self.alpha, (list, torch.Tensor)):
            at = torch.tensor(self.alpha, device=y_pred.device)[y_true]
        else:
            at = self.alpha
        
        focal = at * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        else:
            return focal