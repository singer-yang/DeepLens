"""Perceptual loss function."""

import torch
import torch.nn as nn
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """Perceptual loss based on VGG16 features."""

    def __init__(self, device=None, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        """Initialize perceptual loss.
        
        Args:
            device: Device to put the VGG model on. If None, uses cuda if available.
            weights: Weights for different feature layers.
        """
        super(PerceptualLoss, self).__init__()
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.vgg = models.vgg16(pretrained=True).features.to(device)
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3",
            '29': "relu5_3"
        }
        
        self.weights = weights
        
        for param in self.vgg.parameters():
            param.requires_grad = False
            
    def forward(self, x, y):
        """Calculate perceptual loss.
        
        Args:
            x: Predicted tensor.
            y: Target tensor.
            
        Returns:
            Perceptual loss.
        """
        x_vgg, y_vgg = self._get_features(x), self._get_features(y)
        
        content_loss = 0.0
        for i, (key, value) in enumerate(x_vgg.items()):
            content_loss += self.weights[i] * torch.mean((value - y_vgg[key]) ** 2)
            
        return content_loss
        
    def _get_features(self, x):
        """Extract features from VGG network.
        
        Args:
            x: Input tensor.
            
        Returns:
            Dictionary of feature tensors.
        """
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layer_name_mapping:
                features[self.layer_name_mapping[name]] = x
                
        return features 