from PredictModule import PredictModule
from SemanticSpaceInformedPrompting import Semantic_Space_Informed_Prompting
from TimeSeriesTokenization import Time_Series_Tokenization
from TransformerBlock import TransformerBlocks
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
class SIP_LLM(nn.Module):
     def __init__(self,series_input_length,Transformer_dim,L_p,word_embeddings_path,vocab_dim,prompt_length,model,Transform_blocks_num,output_length,stride=8,*args, **kwargs) -> None:
          super(SIP_LLM,self).__init__(*args, **kwargs)
          N_p=int((series_input_length-L_p)/stride+2)
          self.L_p=L_p
          self.output_length=output_length
          self.TST=Time_Series_Tokenization(series_input_length,Transformer_dim,L_p)
          self.SSIP=Semantic_Space_Informed_Prompting(word_embeddings_path,vocab_dim,prompt_length)
          self.TB=TransformerBlocks(model,Transform_blocks_num,L_p)
          self.PM=PredictModule(N_p,prompt_length,L_p,output_length)
     def forward(self,X):
         """X=dim,batch,series"""
         X=self.TST(X)
         X,Value=self.SSIP(X)
         shape=X.shape
         Z=torch.zeros(size=(shape[0],shape[1],self.output_length))
         for dim in range(shape[0]):
             Y=self.TB(X[dim])
             Z[dim]=self.PM(Y)
         return Z,Value
if __name__=="__main__":
    X=torch.rand(size=(7,2,512))
    tokenizer=GPT2Tokenizer.from_pretrained("/data/huggingface/models/gpt2")
    model=GPT2LMHeadModel.from_pretrained("/data/huggingface/models/gpt2")
    SIP=SIP_LLM(512,768,16,'/home/user/whs/test/IPLLM/word_embedding_weight.pth',5000,2,model,6,96)
    print(SIP(X)[0].shape,SIP(X)[1].shape)#7，2，96