import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class FocalLoss(nn.Module):

    def __init__(self, alpha=.2, gamma=3., reduction="mean"):
        super(FocalLoss, self).__init__()
#         self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, recon_x, x):
        # Calculate BCE loss
        BCE_loss = F.binary_cross_entropy(recon_x, x)
        
        # Calculate focal loss
        pt = torch.exp(-BCE_loss)
        # p = torch.sigmoid(recon_x)
        # pt = p * x + (1 - p) * (1 - x)
        # pt = recon_x * x + (1 - recon_x) * (1 - x)
        F_loss = (1-pt)**self.gamma * BCE_loss
    
        if self.alpha > 0:
            at = self.alpha * x + (1 - self.alpha) * (1 - x)
            F_loss = at * F_loss
        
        if self.reduction == "mean":
            F_loss = F_loss.mean()
        elif self.reduction == "sum":
            F_loss = F_loss.sum()
        
        return F_loss