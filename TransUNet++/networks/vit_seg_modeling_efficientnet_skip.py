import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, efficientnet_b4, EfficientNet_B3_Weights, EfficientNet_B4_Weights

class EfficientNetBackbone(nn.Module):
    def __init__(self, model_type='b3', pretrained=True):
        super().__init__()
        if model_type == 'b3':
            weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
            self.base = efficientnet_b3(weights=weights)
            self.width = 96
            self.skip_channels = [48, 32, 24] 
        elif model_type == 'b4':
            weights = EfficientNet_B4_Weights.DEFAULT if pretrained else None
            self.base = efficientnet_b4(weights=weights)
            self.width = 112
            self.skip_channels = [56, 32, 24] 
        else:
            raise ValueError("Unsupported EfficientNet type. Use 'b3' or 'b4'.")

        self.base.classifier = nn.Identity()

    def forward(self, x):
        features = []
        
        x = self.base.features[0](x)
        x = self.base.features[1](x)
        features.append(x) 
        
        x = self.base.features[2](x)
        features.append(x) 
        
        x = self.base.features[3](x)
        features.append(x) 
        
        x = self.base.features[4](x)
        
        return x, features[::-1]
