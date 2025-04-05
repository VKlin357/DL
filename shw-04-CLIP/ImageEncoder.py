import torch.nn as nn
import timm 

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector.
    """
    def __init__(
        self, model_name="resnet50", pretrained=True, trainable=False
        ):
        """
        We will use standard pretrained ResNet50, and set freeze its parameters.
        Look the documentation of TIMM on how to donwload the model: https://timm.fast.ai/
        """
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        
        if hasattr(self.model, 'fc'):
            in_feats = self.model.fc.in_features
            self.model.fc = nn.Identity()
        elif hasattr(self.model, 'classifier'):
            in_feats = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
        else:
            raise ValueError("РќРµРёР·РІРµСЃС‚РЅР°СЏ Р°СЂС…РёС‚РµРєС‚СѓСЂР°, РЅРµ СѓРґР°Р»РѕСЃСЊ РЅР°Р№С‚Рё fc РёР»Рё classifier.")
        
        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.embedding_dim = in_feats

    def forward(self, x):
        return self.model(x)
