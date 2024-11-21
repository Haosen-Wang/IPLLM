import torch
import transformers
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
class TransformerBlock(nn.Module):
     def __init__(self,layer,*args, **kwargs) -> None:
          super().__init__(*args, **kwargs)
          self.ln_1=layer.ln_1
          self.attn=layer.attn
          self.ln_2=layer.ln_2
          self.mlp=layer.mlp
     def forward(self,X):
         X=self.ln_1(X)
         with torch.no_grad():
             X=self.attn(X)[0]
         X=self.ln_2(X)
         with torch.no_grad():
             X=self.mlp(X)
         return X
         
class TransformerBlocks(nn.Module):
     def __init__(self,model,Transform_blocks_num,L_p,*args, **kwargs) -> None:
          super().__init__(*args, **kwargs)
          self.Transformer_blocks=nn.Sequential()
          for layer in range(Transform_blocks_num):
              self.Transformer_blocks.append(TransformerBlock(model.transformer.h[layer]))
          self.Linear=nn.Linear(768,3*L_p)
     def forward(self,X):
         "Y.size()=dim,batch,N_p,3L_p"
         Y=self.Transformer_blocks(X)
         Y=self.Linear(Y)
         return Y
if __name__=="__main__":
    tokenizer=GPT2Tokenizer.from_pretrained("/data/huggingface/models/gpt2")
    model=GPT2LMHeadModel.from_pretrained("/data/huggingface/models/gpt2")
    positional_embeddings=model.transformer.wpe.weight
    print(positional_embeddings.shape)
    print(model.transformer.h[:6])
    TB=TransformerBlocks(model,6,16)
    X=torch.rand((2,66,768))#batch,N_p+K,D
    print(TB(X).shape)#batch,N_p+K,D#batch,N_p+K,3*L_p
    