from pandas import isna
import torch
import torch.nn as nn
class Losses(nn.Module):
    def __init__(self,lamma, *args, **kwargs) -> None:
        super(Losses,self).__init__(*args, **kwargs)
        self.lamma=torch.tensor(lamma).requires_grad_(False)
        self.MSE=nn.MSELoss()
    def forward(self,Y_label,Y_pre,Value):
        shape=Y_label.shape
        sum_loss=0
        for dim in range(shape[0]):
            item=self.MSE(Y_pre[dim],Y_label[dim])
            cos=torch.sum(Value[dim]).requires_grad_(True)
            sum_loss=item+self.lamma*cos+sum_loss
        return torch.tensor(sum_loss).float().requires_grad_(True)
if __name__=="__main__":
    Y_label=torch.rand((7,2,96))
    Y_pre=torch.rand((7,2,96))
    Value=torch.rand((7,2,2))
    loss=Losses(1e-5)
    su=loss(Y_label,Y_pre,Value)
    print(su)