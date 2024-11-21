from model import SIP_LLM
from transformers import GPT2Tokenizer,GPT2LMHeadModel
from torch.utils.data import DataLoader
from get_data import ExDataset
from get_data import split_set
import torch.optim as optim
from Losses import Losses
from train import train
import torch
import datasets
from torch.utils.data import Dataset
tokenizer=GPT2Tokenizer.from_pretrained("/data/huggingface/models/gpt2")
model_gpt2=GPT2LMHeadModel.from_pretrained("/data/huggingface/models/gpt2")
param_dict={"series_input_length":512,
            "Transformer_dim":768,
            "L_p":16,
            "word_embeddings_path":'/home/user/whs/test/IPLLM/word_embedding_weight.pth',
            "vocab_dim":5000,
            "prompt_length":2,
            "model":model_gpt2,
            "Transform_blocks_num":6,
            "output_length":96,
            "stride":8}
data_path='/home/user/whs/test/IPLLM/ETTh1.csv'
dataset=ExDataset(data_path,512,96,'date')
trainset,testset,valset=split_set(dataset)
trainloader = DataLoader(trainset,batch_size=4,shuffle=True)
valloader = DataLoader(valset,batch_size=4,shuffle=False)
testloader = DataLoader(testset,batch_size=4,shuffle=False)
#model=SIP_LLM(**param_dict)
model=torch.load("model.pth")
lr=0.1
optimizer=optim.Adam(model.parameters(),lr=lr)
loss=Losses(lamma=1e-1)
epoch=50
train(model,optimizer,loss,epoch,trainloader,valloader,testloader)
