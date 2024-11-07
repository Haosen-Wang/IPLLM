import numpy 
from sympy import beta
import torch
import torch.nn as nn
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas
class Time_Series_Tokenization(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def transformed(self,data_x:torch.Tensor,epsi=0):
        gamma_t=torch.nn.Parameter(torch.rand(data_x.size()))
        beta_t=torch.nn.Parameter(torch.rand(data_x.size()))
        mean=torch.mean(data_x,dim=1)
        var=torch.var(data_x,dim=1)
        mean=mean/torch.sqrt(var+epsi)
        mean=mean.unsqueeze(1).expand_as(data_x)
        return gamma_t*(data_x-mean)+beta_t
    def season(self,data_x):
        return 0
    def forward(self,data_x:torch.Tensor,data_y:torch.Tensor):
        data_x=self.transformed(data_x)
        return 0


        
if  __name__=="__main__":
    data_x=torch.rand((64,96,7))
    for item in data_x:
        item=item.numpy()
        res=seasonal_decompose(item, model='additive',period=12)
        seasonal,trend,resid=res.seasonal,res.trend,res.resid
        print(trend.size)
        break
