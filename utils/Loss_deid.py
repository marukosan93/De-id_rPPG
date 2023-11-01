import torch
from torch import nn

class Deid_loss(nn.Module):    # Deid loss
    def __init__(self,wt1,wt2):
        super(Deid_loss,self).__init__()
        self.wt1 = wt1
        self.wt2 = wt2
        return

    def forward(self, preds, labels):       # tensor [Batch, Temporal]
        batch_size = preds.size()[0]
        term1= -torch.mean(torch.sum(labels.view(batch_size, -1) * torch.log(preds.view(batch_size, -1)), dim=1))
        print(term1)
        exit()
        term2 = torch.sqrt(torch.sum(preds*preds,1))
        lossid = self.wt1*torch.mean(term1) #+ self.wt2*torch.mean(term2)
        return lossid
