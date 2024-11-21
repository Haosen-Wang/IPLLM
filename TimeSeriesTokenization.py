from dotenv import get_key
import numpy 
import torch
import torch.nn as nn
import pandas
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
class Time_Series_Tokenization(nn.Module):
    def __init__(self,series_input_length,Transformer_dim,L_p=16,*args, **kwargs) -> None:
        super(Time_Series_Tokenization,self).__init__(*args, **kwargs)
        self.series_input_length=series_input_length
        self.Transformer_dim=Transformer_dim
        self.L_p=L_p
        self.size_trans_trend=nn.Linear(self.series_input_length,self.L_p)
        self.size_trans_seasonal=nn.Linear(self.series_input_length,self.L_p)
        self.size_trans_resid=nn.Linear(self.series_input_length,self.L_p)
        self.g_linear=nn.Linear(3*self.L_p,self.Transformer_dim)
    
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
                row = numpy.where(numpy.isinf(row), 0, row)
                row = numpy.where(numpy.isnan(row), 0, row)
                decomposition = sm.tsa.seasonal_decompose(row, model='additive',period=int(self.series_input_length/2))
                trend[i]=decomposition.trend
                trend[i]=numpy.where(numpy.isnan(trend[i]), 0, trend[i])
                seasonal[i]=decomposition.seasonal
                seasonal[i]=numpy.where(numpy.isnan(seasonal[i]), 0, seasonal[i])
                resid[i]=decomposition.resid
                resid[i]=numpy.where(numpy.isnan(resid[i]), 0, resid[i])
                i=i+1
            trend_pt[j]=torch.from_numpy(trend)
            seasonal_pt[j]=torch.from_numpy(seasonal)
            resid_pt[j]=torch.from_numpy(resid)
            j=j+1
        return trend_pt,seasonal_pt,resid_pt
    
    def patches(self,trend,seasonal,resid,stride=8):
        N_p=int((self.series_input_length-self.L_p)/stride+2)
        L_p=self.L_p
        shape=trend.size()
        trend=self.size_trans_trend(trend).unsqueeze(2).expand(int(shape[0]),int(shape[1]),N_p,L_p)
        seasonal=self.size_trans_seasonal(seasonal).unsqueeze(2).expand(int(shape[0]),int(shape[1]),N_p,L_p)
        resid=self.size_trans_resid(resid).unsqueeze(2).expand(int(shape[0]),int(shape[1]),N_p,L_p)
        P=torch.cat([trend,seasonal,resid],dim=-1)
        P=self.g_linear(P)
        return P

    def forward(self,data_x:torch.Tensor):
        data_x=self.transformed(data_x)
        trend,seasonal,resid=self.season(data_x)
        P=self.patches(trend,seasonal,resid)
        return P
        
if  __name__=="__main__":
   data_x=torch.rand((7,2,512))#dim, batch, series
   time=Time_Series_Tokenization(512,768)
   print(time(data_x).shape)#dim, batch,N_p,D