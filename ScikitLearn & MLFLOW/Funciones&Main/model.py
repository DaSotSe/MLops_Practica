
import torch
import torch.nn as nn

class MultiModalNet(nn.Module):
    def __init__(
        self,
        n_features: int = 17,
        n_classes: int = 4,
        activation_function=nn.ReLU,
        dropout_p: float = 0.3
    ):
        super(MultiModalNet, self).__init__()

        self.activation = activation_function()

        # Rama de metadatos 
        self.features_branch = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            self.activation,
            nn.Dropout(dropout_p),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            self.activation,
            nn.Dropout(dropout_p),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            self.activation,
            nn.Dropout(dropout_p),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            self.activation
        )

        # Rama de imágenes
        self.image_branch = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            self.activation,
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_p),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            self.activation,
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_p),

            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),  
            nn.BatchNorm1d(128),
            self.activation,
            nn.Dropout(dropout_p),

            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            self.activation
        )

        #Clasificador 
        self.classifier = nn.Sequential(
            nn.Linear(16 + 32, 64),
            nn.BatchNorm1d(64),
            self.activation,
            nn.Dropout(dropout_p),
            nn.Linear(64, n_classes)
        )

    def forward(self, features: torch.Tensor, images: torch.Tensor) -> torch.Tensor:

        x_feat = self.features_branch(features)   
        x_img = self.image_branch(images)         
        x = torch.cat([x_feat, x_img], dim=1)    
        return self.classifier(x)                 