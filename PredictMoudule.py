import torch
import torch.nn as nn
from statsmodels.tsa.seasonal import seasonal_decompose
class PredictModule(nn.Module):
     def __init__(self, N_p,K,L_p,output_length,*args, **kwargs) -> None:
          super(PredictModule,self).__init__(*args, **kwargs)
          self.N_p=N_p+K
          self.L_p=L_p
          self.tre_Linear=nn.Linear(self.N_p*L_p,output_length)
          self.sea_Linear=nn.Linear(self.N_p*L_p,output_length)
          self.res_Linear=nn.Linear(self.N_p*L_p,output_length)
     def forward(self,X):
         shape=X.shape
         X_tre=X[:,:,:self.L_p].reshape(shape[0],-1)
         X_sea=X[:,:,self.L_p:2*self.L_p].reshape(shape[0],-1)
         X_res=X[:,:,2*self.L_p:3*self.L_p].reshape(shape[0],-1)
         Z_tre=self.tre_Linear(X_tre)
         Z_sea=self.sea_Linear(X_sea)
         Z_res=self.res_Linear(X_res)
         Z=Z_tre+Z_sea+Z_res
         return Z
if __name__=="__main__":
    X=torch.rand((2,66,48))#batch,N_p+K,D#batch,N_p+K,3*L_p
    PM=PredictModule(64,2,16,96)
    print(PM(X).shape)