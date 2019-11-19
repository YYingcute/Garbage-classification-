#!en-coding=utf-8
import os
from PIL import Image
import torch
from torchvision import *
import torchvision
path="dataset-resized"
classes = os.listdir(path)
classes.remove('.DS_Store')
def initialize_model(model_name,model_path,num_classes,use_pretrained=True):
    if model_name=="resnet":
        model_ft=models.resnet101()
        num_ftrs=model_ft.fc.in_features
        model_ft.fc=torch.nn.Linear(num_ftrs,num_classes)
        model_ft.load_state_dict(torch.load(model_path))
        input_size=224
    else:
        print("model not implemented")
        return None,None
    return model_ft,input_size

def predect(image_path):
    model_name="resnet"
    model_path="0.9251170046801872.pth"
    model,_=initialize_model(model_name,model_path=model_path,num_classes=6)
    model.eval()
    transformtest = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((224, 224)),
         torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值 方差
         ])
    input=Image.open(image_path)
    input=transformtest(input)
    input=input.reshape(1,3,224,224)
    output=model(input)
    preds = output.argmax(dim=1)
    print(classes[preds])
image_path="dataset-resized/cardboard/cardboard1.jpg"
predect(image_path)
