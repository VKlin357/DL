import torch
import torch.nn as nn
from transformers import DistilBertModel

class TextEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", pretrained=True, trainable=False):
        super().__init__()
        """
        Create the model and set its weights frozen. 
        Use Transformers library docs to find out how to do this.
        """
        self.target_token_idx = 0
        self.model = DistilBertModel.from_pretrained(model_name) if pretrained else DistilBertModel()

        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Pass the arguments through the model and make sure to return CLS token embedding
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, self.target_token_idx, :]
        return cls_embedding
