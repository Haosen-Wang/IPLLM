import torch
import transformers
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn.functional as F
class Semantic_Space_Informed_Prompting(nn.Module):
     def __init__(self,word_embeddings_path,vocab_dim:int,prompt_length=2,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vocab_dim=vocab_dim
        self.prompt_length=prompt_length
        self.word_embeddings=torch.load(word_embeddings_path).t()
        self.f_e=nn.Linear(self.word_embeddings.shape[-1],self.vocab_dim)
     def get_cos_score(self,P_i,word_embeddings):
         """word_embeddings.size()=vocabsize,D"""
         shape=word_embeddings.shape
         word_embeddings=word_embeddings.unsqueeze(dim=1).expand(shape[0],(P_i.shape[0])//shape[-1],shape[-1])
         word_embeddings=word_embeddings.reshape((shape[0],-1))
         cosine_similarity = F.cosine_similarity(word_embeddings, P_i, dim=1)
         vector_expanded = P_i.expand(word_embeddings.size(0), -1)
         return cosine_similarity
     def get_Z_top_K(self,P,word_embeddings):
         """P.size()=batch,dim,N_p*D
            word_embeddings.size()=vocabsize,D
         """
         shape=P.shape
         P=P.view(shape[0],shape[1],-1)
         K=self.prompt_length
         Z=torch.zeros(size=(shape[0],shape[1],shape[2]+K,shape[3]))
         Values=torch.zeros(size=(shape[0],shape[1],K))
         for vari in range(0,shape[0]):
             for dim in range(0,shape[1]):
                 cos_score_vector=self.get_cos_score(P[vari][dim],word_embeddings)
                 values, indices = torch.topk(cos_score_vector, k=K)
                 Values[vari][dim]=values
                 e_k=word_embeddings[indices]
                 Z_i=torch.cat((e_k,P[vari][dim].reshape((shape[2],shape[3]))),dim=0)
                 Z[vari][dim]=Z_i
         return Z,Values
     def forward(self,P):
        """P.size()=batch,dim,N_p,D
        Z.size()=batch,dim,N_p+K,D
        Value.size()=batch,dim,K"""
        word_embeddings=self.f_e(self.word_embeddings).t()
        Z,Values=self.get_Z_top_K(P,word_embeddings)
        #P.size()=batch,dim,N_p*D
        return Z,Values
    
    
    
    
if __name__=="__main__":
    tokenizer=GPT2Tokenizer.from_pretrained("/data/huggingface/models/gpt2")
    model=GPT2LMHeadModel.from_pretrained("/data/huggingface/models/gpt2")
    word_embeddings = model.transformer.wte.weight
    torch.save(word_embeddings,'word_embedding_weight.pth')
    word_embeddings=torch.load('/home/user/whs/test/IPLLM/word_embedding_weight.pth')
    print(word_embeddings.t().shape)
    P=torch.rand(size=(7,2,64,768))#dim,batch,N_p,D
    SS=Semantic_Space_Informed_Prompting('/home/user/whs/test/IPLLM/word_embedding_weight.pth',5000)
    Z,Values=SS(P)
    print(Z.shape,Values.shape)#dim,batch,N_p+k,D  dim,