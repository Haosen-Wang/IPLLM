import numpy 
import torch
import torch.nn as nn
import pandas
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
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
        data_x=torch.rand((64,96,7))
        trend_pt=torch.zeros(data_x.shape)
        seasonal_pt=torch.zeros(data_x.shape)
        resid_pt=torch.zeros(data_x.shape)
        j=0
        for item in data_x:
            item=torch.transpose(item,0,1)
            item=item.numpy()
            trend=numpy.zeros(item.shape)
            seasonal=numpy.zeros(item.shape)
            resid=numpy.zeros(item.shape)
            i=0
            for row in item:
                decomposition = sm.tsa.seasonal_decompose(row, model='additive',period=12)
                trend[i]=decomposition.trend
                seasonal[i]=decomposition.seasonal
                resid[i]=decomposition.resid
                i=i+1
            trend_pt[j]=torch.transpose(torch.from_numpy(trend),dim0=0,dim1=1)
            seasonal_pt[j]=torch.transpose(torch.from_numpy(seasonal),dim0=0,dim1=1)
            resid_pt[j]=torch.transpose(torch.from_numpy(resid),dim0=0,dim1=1)
            j=j+1
        return trend_pt,seasonal_pt,resid_pt

    def forward(self,data_x:torch.Tensor,data_y:torch.Tensor):
        data_x=self.transformed(data_x)
        trend,seasonal,resid=self.season(data_x)
        return 0


        
if  __name__=="__main__":
   print(1)
