#!en-coding=utf-8
import os
import random
from PIL import Image
import torch
from torchvision import *
import torchvision
from torch.utils.data import *
import copy
def read_img(path):
    return Image.open(path)
   # 随机划分训练集和验证集合
path="dataset-resized"
model_path="0.8271276595744681.pth"
traind = {}
vald = {}
classes = os.listdir(path)
classes.remove(".DS_Store")
print(classes)
for garbage in classes:
    name = os.listdir(path + "/" + garbage)
    for imgname in name:
        probo = random.randint(1, 100)
        if probo > 85:
            vald[garbage + "/" + imgname] = garbage
        else:
            traind[garbage + "/" + imgname] = garbage
print("train{},val{}".format(len(traind.keys()), len(vald.keys())))

# 定义dataset
class myDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, trainorval,input_size,classes,transform, **kwargs):

        self.data_root=data_root
        self.dataid=[x for x in trainorval.keys()]
        self.data=trainorval
        self.classes=classes
        self.transform=transform
        self.input_size=input_size,
    def __len__(self):
        return len(self.data.keys())


    def __getitem__(self, idx):
        path = self.dataid[idx]
        img =read_img(os.path.join(self.data_root, path))
        img=self.transform(img)
        return img,self.classes.index(self.data[path])

# 初始化设定
model_name="resnet"
num_classes=6
batch_size=48
num_epochs=80
input_size=224
transformtrain=torchvision.transforms.Compose(
            [torchvision.transforms.RandomResizedCrop(224),
             torchvision.transforms.RandomHorizontalFlip(),
             torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值 方差
                 ])
transformtest=torchvision.transforms.Compose(
            [torchvision.transforms.Resize((224,224)),
             torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值 方差
                 ])

# 是否有GPU
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Flag for extracting when False,we finetuning the whole model,when True,we only update the reshaped layer params
feature_extract=True
# 设置哪些层需要梯度
def set_parameter_requires_grad(model,feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad=False

#初始化模型
def initialize_model(model_name,feature_extract,num_classes,use_pretrained=True):
    if model_name=="resnet":
        model_ft=models.resnet34(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft,feature_extract)
        num_ftrs=model_ft.fc.in_features
        model_ft.fc=torch.nn.Linear(num_ftrs,num_classes)
        input_size=224
    else:
        print("model not implemented")
        return None,None
    return model_ft,input_size
def initialize_model1(model_name,model_path,num_classes,use_pretrained=True):
    if model_name=="resnet":
        model_ft=models.resnet34()
        num_ftrs=model_ft.fc.in_features
        model_ft.fc=torch.nn.Linear(num_ftrs,num_classes)
        model_ft.load_state_dict(torch.load(model_path))
        input_size=224
    else:
        print("model not implemented")
        return None,None
    return model_ft,input_size

train_set = myDataset(path,traind,224,classes,transformtrain)
test_set = myDataset(path,vald,224,classes,transformtest)
traindataloader = DataLoader(
train_set, shuffle=True, num_workers=4, batch_size=8)
testdataloader = DataLoader(
test_set, shuffle=True, num_workers=4, batch_size=8)

def train_model(model,loss_fn,optimizer,num_epochs=500):
    best_model_wts=copy.deepcopy(model.state_dict())
    best_acc=0.
    val_acc_history=[]
    for epoch in range(num_epochs):
        for phase in ["train","val"]:
            running_loss=0
            running_corrects=0
            if phase=="train":
                dataloader=traindataloader
                model.train()
            else:
                model.eval()
                dataloader=testdataloader
            for inputs,labels in dataloader:
                inputs,labels=inputs.to(device),labels.to(device)
                with torch.autograd.set_grad_enabled(phase=="train"):
                    outputs=model(inputs)
                    loss=loss_fn(outputs,labels)
                preds=outputs.argmax(dim=1)
                if phase=="train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                running_loss+=loss.item()*inputs.size(0)
                running_corrects+=torch.sum(preds.view(-1)==labels.view(-1)).item()
            epoch_loss=running_loss/len(dataloader.dataset)
            epoch_acc=running_corrects/len(dataloader.dataset)
            print("epoch:{}phase:{}loss:{},acc:{}".format(epoch,phase,epoch_loss,epoch_acc))
            if phase=="val"and epoch_acc>best_acc:
                best_acc=epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase=="val":
                val_acc_history.append(epoch_acc)
    torch.save(best_model_wts,str(best_acc)+".pth")
    return model,val_acc_history
#model_ft,input_size=initialize_model(model_name,feature_extract,num_classes,use_pretrained=True)
model_ft,input_size=initialize_model1(model_name,model_path,num_classes,use_pretrained=True)
model_ft=model_ft.to(device)
optimizer=torch.optim.SGD(filter(lambda p:p.requires_grad,model_ft.parameters()),lr=0.0001,momentum=0.9)
loss_fn=torch.nn.CrossEntropyLoss()
model,val_acc_history=train_model(model_ft,loss_fn,optimizer,num_epochs=num_epochs)

