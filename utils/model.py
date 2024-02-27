import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv
from attention import Attention
from collections import OrderedDict
import torch
import torchxrayvision as xrv
from attention import Attention
from collections import OrderedDict


class FeatureExtractor(nn.Module):
    """
    FeatureExtractor class is responsible for extracting features from input data using DenseNet and ResNet models.
    It supports different attention types and fusion methods for combining the features.
    """

    def __init__(self, attention_type, fusion_method):
        super(FeatureExtractor, self).__init__()
        self.densenet = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.resnet = xrv.models.ResNet(weights="resnet50-res512-all")
        self.fusion_method = fusion_method
        self.densenet.op_threshs = None
        self.resnet.op_threshs = None
        self.densenet.classifier = nn.Identity()
        self.resnet.model.fc = nn.Identity()
        self.densenet.eval()
        self.resnet.eval()
        self.conv_extrapolator = nn.Conv2d(1024, 2048, kernel_size=1)
        if attention_type == "se":
            self.attention_resnet = Attention(attention_type, channel=2048, reduction=8)
            self.attention_densenet = Attention(
                attention_type, channel=1024, reduction=8
            )
        elif attention_type == "eca":
            self.attention_resnet = Attention(attention_type)
            self.attention_densenet = Attention(attention_type)
        elif attention_type == "cbam":
            self.attention_resnet = Attention(attention_type, channel=2048)
            self.attention_densenet = Attention(attention_type, channel=1024)
        elif attention_type == "external":
            self.attention_resnet = Attention(attention_type, d_model=2048)
            self.attention_densenet = Attention(attention_type, d_model=1024)
        elif attention_type == "coordatt":
            self.attention_resnet = Attention(attention_type, inp=2048, oup=2048)
            self.attention_densenet = Attention(attention_type, inp=1024, oup=1024)
        else:
            raise ValueError(f"Unrecognized attention type {attention_type}")

    def fuse_features(self, dense_features, res_features, fusion_method):
        """
        Fuse the dense features and resnet features using the specified fusion method.

        Args:
            dense_features (torch.Tensor): Dense features extracted from the input data.
            res_features (torch.Tensor): Resnet features extracted from the input data.
            fusion_method (str): Fusion method to combine the features. Supported methods are "concat", "add", and "max".

        Returns:
            torch.Tensor: Combined features.

        Raises:
            ValueError: If the fusion method is not recognized.
        """

        dense_features = self.conv_extrapolator(dense_features)
        resnet_features = F.interpolate(
            res_features, size=(7, 7), mode="bilinear", align_corners=False
        )
        if fusion_method == "concat":
            combined_features = torch.cat((dense_features, resnet_features), dim=1)
        elif fusion_method == "add":
            combined_features = dense_features + res_features
        elif fusion_method == "max":
            combined_features = torch.max(dense_features, res_features)
        else:
            raise ValueError(f"Unrecognized fusion method {fusion_method}")

        return combined_features

    def forward(self, x):
        """
        Forward pass of the FeatureExtractor.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Combined features extracted from the input data.
        """
        dense_features = self.densenet.features(x)
        res_features = self.resnet.features2(x)
        dense_features = dense_features * self.attention_densenet(dense_features)
        res_features = res_features * self.attention_resnet(res_features)
        combined_features = self.fuse_features(
            dense_features, res_features, self.fusion_method
        )

        return combined_features


class FusionModel(nn.Module):
    """
    Temporary Architecture for the last model.

    Args:
        attention_type (str): The type of attention mechanism to be used.
        fusion_method (str): The method for fusing features.

    Attributes:
        feature_extractor (FeatureExtractor): The feature extractor module.
        fc (nn.Linear): The fully connected layer.

    """

    def __init__(self, in_channels, num_init_features, attention_type, fusion_method, train_label_shape):
        super(FusionModel, self).__init__()
        self.feature_extractor = FeatureExtractor(attention_type, fusion_method)
        # self.features = nn.Sequential(OrderedDict([
        #     ('conv0', nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
        #     ('norm0', nn.BatchNorm2d(num_init_features)),
        #     ('relu0', nn.ReLU(inplace=True)),
        #     ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        # ]))
        # self.features.add_module('conv1', nn.Conv2d(64, 128, kernel_size=3, padding=1))
        # self.features.add_module('relu1', nn.ReLU())
        # self.features.add_module('conv2', nn.Conv2d(128, 256, kernel_size=3, padding=1))
        # self.features.add_module('relu2', nn.ReLU())
        # self.features.add_module('conv3', nn.Conv2d(512, 1024, kernel_size=3, padding=1))
        # self.features.add_module('relu3', nn.ReLU())
        # self.features.add_module('conv4', nn.Conv2d(1024, 2048, kernel_size=3, padding=1))
        # self.features.add_module('relu4', nn.ReLU())
        if num_init_features == 4096:
            self.conv0 = nn.Conv2d(4096, 2048, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(512, train_label_shape, kernel_size=1)
        self.relu3 = nn.ReLU()
        self.features = nn.Sequential(self.conv1, self.relu1, self.conv2, self.relu2, self.conv3, self.relu3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(train_label_shape, train_label_shape)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.feature_extractor(x)
        if hasattr(self, "conv0"):
            x = self.conv0(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        x = nn.Sigmoid()(x)
        return x


if __name__ == "__main__":
    x = torch.rand(1, 1, 256, 256)
    model = FusionModel("se", "max", 14)
    output = model(x)
    print(output.shape)
