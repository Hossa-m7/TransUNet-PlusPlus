import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

class ConvNeXtBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            weights = ConvNeXt_Tiny_Weights.DEFAULT
            self.convnext = convnext_tiny(weights=weights)
        else:
            self.convnext = convnext_tiny()
        
        self.convnext.classifier = nn.Identity()
        
        self.root_1_2 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        
        self.width = 384 

    def forward(self, x):
        features =[]
        
        feat_1_2 = self.root_1_2(x)
        features.append(feat_1_2)
        
        x = self.convnext.features[0](x) 
        x = self.convnext.features[1](x) 
        features.append(x)
        
        x = self.convnext.features[2](x) 
        x = self.convnext.features[3](x) 
        features.append(x)
        
        x = self.convnext.features[4](x) 
        x = self.convnext.features[5](x) 
        
        return x, features[::-1]
