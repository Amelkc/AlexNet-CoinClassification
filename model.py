import torch.nn as nn 
from torchvision import models
from torchvision.models import alexnet, AlexNet_Weights



def build_alexnet(num_classes=1000, pretrained=True):
    model = alexnet(weights=AlexNet_Weights.DEFAULT if pretrained else None)
    #model.classifier[6] = nn.Linear(in_features, num_classes)
    model.classifier = nn.Sequential(
        nn.Dropout(),
        nn.Linear(256 * 6 * 6, 2048),  
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(2048, 1024),           
        nn.ReLU(inplace=True),
        nn.Linear(1024, num_classes),  
    )
    return model
