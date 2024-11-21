from math import isnan
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
def train(model,optimizer,loss,epoch,train_loader,val_loader,test_loader):
    model=model
    i=0
    Loss=loss
    for epoch in range(epoch):
        epoch_loss=0
        val_loss=0
        test_loss=0
        train_avg_losses=[]
        test_avg_losses=[]
        val_avg_losses=[]
        for x,y in train_loader:
                x=x.transpose(0,1)
                y=y.transpose(0,1)
                optimizer.zero_grad()
                outputs,values=model(x)
                loss=Loss(y,outputs,values)
                loss.backward()
                optimizer.step()
                epoch_loss=epoch_loss+loss.item()
                ie_loss=loss.item()
                i=i+1
                if i%100==0:
                    torch.save(model,'model.pth')
                    print("model has saved")
                print(f"epoch:{epoch},train_loss:{ie_loss}")
                torch.cuda.memory.empty_cache()
        train_avg_loss=epoch_loss/len(train_loader)
        train_avg_losses.append(train_avg_loss)
        
        for x,y in val_loader:
            try:
                x=x.transpose(0,1)
                y=y.transpose(0,1)
                output,value=model(x)
                loss=Loss(y,output,value).item()
                val_loss=loss+val_loss
            except Exception:
                pass
        val_avg_loss=val_loss/len(val_loader)
        val_avg_losses.append(val_avg_loss)
        for x,y in test_loader:
            try:
                x=x.transpose(0,1)
                y=y.transpose(0,1)
                output,value=model(x)
                loss=Loss(y,output,value).item()
                test_loss=loss+test_loss
            except Exception:
                pass
        test_avg_loss=test_loss/len(test_loader)
        test_avg_losses.append(test_avg_loss)
        print(f"epoch:{epoch},epoch_train_avgloss:{train_avg_loss},epoch_val_avgloss:{val_avg_loss},epoch_test_avgloss:{test_avg_loss}")
        plt.plot(train_avg_losses, label='train')
        plt.plot(test_avg_losses, label='test')
        plt.plot(val_avg_losses, label='val')
        plt.legend()
        plt.show()
        
        
