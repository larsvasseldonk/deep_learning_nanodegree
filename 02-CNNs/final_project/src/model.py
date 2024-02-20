import torch
import torch.nn as nn


# Define the CNN architecture
class MyModel2(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            
            nn.Linear(256*7*7, 3000),
            nn.BatchNorm1d(3000),
            nn.ReLU(),
            
            nn.Linear(3000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            
            nn.Linear(1000, num_classes),
        )
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        x = self.classifier(x)
        return x


# define the CNN architecture
class MyModel1(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        
        # CONV BLOCK 1
        # RGB images, so input depth is 3
        # Kernel size: 224 / 4 = 56, so no padding needed
        # Stride is not needed in Conv2d since we perform MaxPooling
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, stride=2), # image size = 224 / 2 = 112 
            nn.LeakyReLU(),
            nn.Dropout2d(p=dropout), # Apply Dropout to randomly zero out entire channels
        )
            
        # CONV BLOCK2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, stride=2), # image size = 112 / 2 = 56
            nn.LeakyReLU(),
            nn.Dropout2d(p=dropout),
        )
            
        # CONV BLOCK3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2), # image size = 56 / 2 = 28
            nn.LeakyReLU(),
            nn.Dropout2d(p=dropout),
        )
        
        # CONV BLOCK4
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, stride=2), # image size = 28 / 2 = 14
            nn.LeakyReLU(),
            nn.Dropout2d(p=dropout),
        )
        
        # CONV BLOCK5
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, stride=2), # image size = 14 / 2 = 7
            nn.ReLU(),
            nn.LeakyReLU(),
            nn.Dropout2d(p=dropout),
        )
            
        # FLATTEN FEATURE MAPS
        self.flatten = nn.Flatten()
            
        # FULLY CONNECTED LAYERS
        self.fc1 = nn.Sequential(
            nn.Linear(4096, 2048), # 15488 / 2 = 7744
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 1024), # 15488 / 2 = 7744
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        
        self.fc3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
            
        # Map to output classes
        self.fc4 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        
        out = self.flatten(out)
        
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        
        return out


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
