
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
    Basic block of the ResNet.
    """

    expansion = 1

    def __init__(self, in_planes, planes, norm_type, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(1, planes) if norm_type == "LN" else nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(1, planes) if norm_type == "LN" else nn.BatchNorm2d(planes)

    def forward(self, x):
        """
        Forward method.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class CustomResNet(nn.Module):
    """
    CustomResNet Architecture as per the assignment 9
    """

    def __init__(self, block, norm_type, num_classes=10):
        super().__init__()

        # PrepLayer
        self.preplayer = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3, stride=1, padding=1,bias= False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        #Layer 1
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3, stride=1, padding=1,bias= False),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Resnet Block 1
        self.resnet_1 = block(128, 128, norm_type=norm_type)


        #Layer 2
        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(128,256,kernel_size=3, stride=1, padding=1,bias= False),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        #Layer 3
        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(256,512,kernel_size=3, stride=1, padding=1,bias= False),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # Resnet Block 2
        self.resnet_2 = block(512, 512, norm_type=norm_type)

        # MaxPool with Kernel size 4
        self.pooling = nn.MaxPool2d(kernel_size=4, stride=4)

        # FC Layer
        self.fc = nn.Linear(512,num_classes)


    def forward(self, x):
        """
        Forward method.
        """
        out = self.preplayer(x)
        
        out = self.conv_layer_1(out)
        
        resnet_1_out = self.resnet_1(out)
        
        out = out + resnet_1_out
        ############################
        out = self.conv_layer_2(out)
        out = self.conv_layer_3(out)
        resnet_2_out = self.resnet_2(out)

        out = out + resnet_2_out
        #############################

        out = self.pooling(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def custom_resnet(norm_type="BN"):
    """
    Custom ResNet18 Model.
    """
    return CustomResNet(BasicBlock, norm_type)
