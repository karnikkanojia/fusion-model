import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv
from attention import Attention
import torch
import torchxrayvision as xrv
from attention import Attention
import pytorch_lightning as pl


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

    def __init__(
        self,
        num_init_features,
        attention_type,
        fusion_method,
        train_label_shape,
    ):
        super(FusionModel, self).__init__()
        self.feature_extractor = FeatureExtractor(attention_type, fusion_method)
        if num_init_features == 4096:
            self.conv0 = nn.Conv2d(4096, 2048, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(512, train_label_shape, kernel_size=1)
        self.relu3 = nn.ReLU()
        self.features = nn.Sequential(
            self.conv1, self.relu1, self.conv2, self.relu2, self.conv3, self.relu3
        )
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
        x = nn.Softmax(dim=1)(x)
        return x


class FusionModelLightning(pl.LightningModule):
    def __init__(self, num_init_features, attention_type, fusion_method, train_label_shape, learning_rate=1e-3):
        super(FusionModelLightning, self).__init__()
        self.save_hyperparameters()
        
        # FeatureExtractor internal setup
        self.feature_extractor = FeatureExtractor(attention_type, fusion_method)
        if num_init_features == 4096:
            self.conv0 = torch.nn.Conv2d(4096, 2048, kernel_size=3, padding=1)
        self.conv1 = torch.nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(512, train_label_shape, kernel_size=1)
        self.relu3 = torch.nn.ReLU()
        self.features = torch.nn.Sequential(
            self.conv1, self.relu1, self.conv2, self.relu2, self.conv3, self.relu3
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = torch.nn.Flatten()
        self.classifier = torch.nn.Linear(train_label_shape, train_label_shape)
        self.learning_rate = learning_rate

    def forward(self, x):
        x = self.feature_extractor(x)
        if hasattr(self, "conv0"):
            x = self.conv0(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        x = F.softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        idx, y, x = batch.values()
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    # Optionally, add validation and test steps
    def validation_step(self, batch, batch_idx):
        idx, y, x = batch.values()
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        idx, y, x = batch.values()
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)


if __name__ == "__main__":
    x = torch.rand(2, 1, 256, 256)
    model = FusionModel(2048, "se", "max", 14)
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    output = model(x)
    print(output.shape)
    print(output)
