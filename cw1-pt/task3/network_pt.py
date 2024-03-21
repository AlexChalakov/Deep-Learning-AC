# network module
# adapted from: https://pytorch.org/vision/main/models/vision_transformer.html

import torch.nn as nn
import torchvision.models as models

class MyVisionTransformer(nn.Module):
    """
    MyVisionTransformer is a custom class that extends the nn.Module class.
    It loads a pre-trained Vision Transformer model (vit_b_16) from torchvision.models
    and adjusts the classifier to the desired number of classes.

    Further development notes: Customised model creation is possible by modifying the
    Vision Transformer model architecture. For example, the number of layers, hidden units,
    and other hyperparameters can be adjusted to improve the model's performance.
    """
    def __init__(self, num_classes=10):
        super(MyVisionTransformer, self).__init__()
        # Load the pre-trained Vision Transformer model
        self.vit = models.vit_b_16(pretrained=True)
        # Adjust the classifier to the desired number of classes
        self.vit.heads = nn.Linear(self.vit.heads[0].in_features, num_classes)
    
    def forward(self, x):
        # Forward pass through the Vision Transformer model
        return self.vit(x)