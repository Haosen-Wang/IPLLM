from dotenv import get_key
import numpy 
import torch
import torch.nn as nn
import pandas
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
class Time_Series_Tokenization(nn.Module):
    def __init__(self,series_input_length,Transformer_dim,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.series_input_length=series_input_length
        self.Transformer_dim=Transformer_dim
    
    def transformed(self,data_x:torch.Tensor,epsi=0):
        gamma_t=torch.nn.Parameter(torch.rand(data_x.size()))
        beta_t=torch.nn.Parameter(torch.rand(data_x.size()))
        mean=torch.mean(data_x,dim=1)
        var=torch.var(data_x,dim=1)
        mean=mean/torch.sqrt(var+epsi)
        mean=mean.unsqueeze(1).expand_as(data_x)
        return gamma_t*(data_x-mean)+beta_t
    
    def season(self,data_x):
        data_x=data_x.detach()
        trend_pt=torch.zeros(data_x.shape)
        seasonal_pt=torch.zeros(data_x.shape)
        resid_pt=torch.zeros(data_x.shape)
        j=0
        for item in data_x:
            item=item.numpy()
            trend=numpy.zeros(item.shape)
            seasonal=numpy.zeros(item.shape)
            resid=numpy.zeros(item.shape)
            i=0
            for row in item:
                decomposition = sm.tsa.seasonal_decompose(row, model='additive',period=7)
                trend[i]=decomposition.trend
                seasonal[i]=decomposition.seasonal
                resid[i]=decomposition.resid
                i=i+1
            trend_pt[j]=torch.from_numpy(trend)
            seasonal_pt[j]=torch.from_numpy(seasonal)
            resid_pt[j]=torch.from_numpy(resid)
            j=j+1
        return trend_pt,seasonal_pt,resid_pt
    
    def patches(self,trend,seasonal,resid,L_p=16,stride=8):
        N_p=int((self.series_input_length--L_p)/stride+2)
        shape=trend.size()
        input_dim=self.series_input_length
        size_trans_trend=nn.Linear(input_dim,L_p)
        size_trans_seasonal=nn.Linear(input_dim,L_p)
        size_trans_resid=nn.Linear(input_dim,L_p)
        trend=size_trans_trend(trend).unsqueeze(2).expand(int(shape[0]),int(shape[1]),N_p,L_p)
        seasonal=size_trans_seasonal(seasonal).unsqueeze(2).expand(int(shape[0]),int(shape[1]),N_p,L_p)
        resid=size_trans_resid(resid).unsqueeze(2).expand(int(shape[0]),int(shape[1]),N_p,L_p)
        P=torch.cat([trend,seasonal,resid],dim=-1)
        g_linear=nn.Linear(3*L_p,self.Transformer_dim)
        P=g_linear(P)
        return P

    def forward(self,data_x:torch.Tensor,data_y:torch.Tensor):
        data_x=self.transformed(data_x)
        trend,seasonal,resid=self.season(data_x)
        P=self.patches(trend,seasonal,resid)
        return P
        
if  __name__=="__main__":
   data_x=torch.rand((64,7,512))
   time=Time_Series_Tokenization(512,64)
   print(time(data_x,1).shape)
   
