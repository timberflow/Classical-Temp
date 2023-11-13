import torch.nn as nn
import torch.nn.functional as F

class MultiChoiceCrossEntropyLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.compute_loss = nn.KLDivLoss(reduction = "mean")

    def forward(self, logits, target_distribution):
        return self.compute_loss(logits, target_distribution)
    
class CrossEntropyLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.compute_loss = nn.CrossEntropyLoss(reduction = "mean")
    
    def forward(self, logits, target):
        return self.compute_loss(logits, target)