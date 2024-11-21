import torch
import datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import csv
import pandas as pd
import random
class ExDataset(Dataset):
    def __init__(self,data_path,series_length:int,series_output_length:int,date_label) -> None:
        super().__init__()
        self.data=[]
        df = pd.read_csv(data_path)
        df = df.drop(date_label, axis=1)
        np=df.values
        for i in range(len(np)):
            if i+series_length+series_output_length>=len(np):
                break
            data_x=torch.tensor(np[i:i+series_length]).transpose(dim0=0,dim1=1)
            data_y=torch.tensor(np[i+series_length:i+series_length+series_output_length]).transpose(dim0=0,dim1=1)
            self.data.append((data_x,data_y))
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    

def split_set(data_set:ExDataset,proportion:list=[0.6,0.2,0.2]):
    datase=random.shuffle(data_set.data)
    lengths=[int(len(dataset) * p) for p in proportion]
    trainset=dataset[0:lengths[0]]
    testset=dataset[lengths[0]:lengths[0]+lengths[1]]
    valset=dataset[lengths[0]+lengths[1]:lengths[0]+lengths[1]+lengths[2]]
    return trainset,testset,valset
    
    
if __name__=="__main__":
    data_path='ETTh1.csv'
    dataset=ExDataset(data_path,512,96,'date')
    trainset,testset,valset=split_set(dataset)
    trainloader = DataLoader(trainset,batch_size=64,shuffle=True)
    for x,y in trainloader:
        print(x,y)
        print(x.size(),y.size())
        break