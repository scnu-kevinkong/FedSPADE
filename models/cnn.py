from torch import nn
import torch.nn.functional as F
import torch
"""
Small CNN Architectures taken from
https://github.com/JianXu95/FedPAC/blob/main/models/cnn.py
"""

# File: models/cifarnet_taylor.py (or modify your existing models.py)
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.taylor_mlp_layer import TaylorMLP_Layer # Assuming TaylorMLP_Layer is in models/

class CIFARNetTaylor(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, 
                 D_h_taylor=128, N_taylor=8, taylor_activation="gelu",
                 # Optional: for initializing TaylorMLP_Layer from a conceptual standard MLP
                 initialize_taylor_from_conceptual_mlp=False,
                 conceptual_W1_init=None, conceptual_b1_init=None,
                 conceptual_W2_init=None, conceptual_c2_init=None,
                 conceptual_z0_init=None):
        super(CIFARNetTaylor, self).__init__()
        self.input_shape = (in_channels, 32, 32)
        self.conv1 = nn.Conv2d(in_channels, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.flat_size = 64 * 3 * 3 
        
        self.linear = nn.Linear(self.flat_size, 128) # Feature extractor part

        self.fc_taylor = TaylorMLP_Layer(
            in_features=128, 
            out_features=num_classes,
            D_h_taylor=D_h_taylor,
            N_taylor=N_taylor,
            activation_name=taylor_activation
        )
        
        if initialize_taylor_from_conceptual_mlp:
            # This part assumes you provide the conceptual weights for the two-layer MLP
            # that the TaylorMLP_Layer represents.
            if conceptual_W1_init is not None: self.fc_taylor.W1.data.copy_(conceptual_W1_init)
            if conceptual_b1_init is not None: self.fc_taylor.b1.data.copy_(conceptual_b1_init)
            if conceptual_z0_init is not None: self.fc_taylor.z0.data.copy_(conceptual_z0_init)
            
            if conceptual_W2_init is not None and conceptual_c2_init is not None:
                self.fc_taylor.initialize_theta_from_conceptual_mlp(
                    conceptual_W2_init, conceptual_c2_init, taylor_activation, N_taylor
                )
            else:
                print("Warning: Conceptual W2/c2 not provided for Taylor Theta initialization. Theta uses random init.")
            
        self.D = 128 
        self.cls = num_classes
        
        # Parameters to be aggregated in FedTaylorCFL
        self.aggregated_param_keys = [k for k, _ in self.named_parameters()] # Send all by default

    def forward(self, x, return_feat=False):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = x.view(-1, self.flat_size)
        feat = F.leaky_relu(self.linear(x))
        out = self.fc_taylor(feat)
        
        if return_feat:
            return feat, out
        return out

class CIFARNet(nn.Module):
    """
    This version of CIFARNet is architecturally identical to CIFARNet_EDL
    for a fair comparison in federated learning experiments. It includes
    BatchNorm and Dropout layers.
    """
    def __init__(self, num_classes=10, in_channels=3, dropout_rate=0.2):
        super(CIFARNet, self).__init__()
        self.num_classes = num_classes
        self.input_shape = (in_channels, 32, 32)
        self.D = 128  # Feature dimension

        # Convolutional Block 1
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(dropout_rate)

        # Convolutional Block 2
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(dropout_rate)

        # Convolutional Block 3
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(dropout_rate)

        # Calculate flat_size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 32, 32)
            x_calc = self._forward_features(dummy_input)
            self.flat_size = x_calc.view(-1).shape[0]

        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flat_size, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout_fc1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(256, self.D) # Feature extractor output
        self.bn_fc2 = nn.BatchNorm1d(self.D)
        self.dropout_fc2 = nn.Dropout(dropout_rate)
        
        # Classifier Head
        self.fc = nn.Linear(self.D, num_classes)
        
        # Define keys for the classifier layer for federated learning
        self.classifier_weight_keys = ['fc.weight', 'fc.bias']
        self.cls = num_classes

    def _forward_features(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        return x

    def forward(self, x, return_feat=False):
        # Feature extraction
        x = self._forward_features(x)
        x = x.view(-1, self.flat_size)

        # Fully connected layers
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc1(x)
        
        features = F.relu(self.bn_fc2(self.fc2(x)))
        features = self.dropout_fc2(features)
        
        # Classifier
        out = self.fc(features)

        if return_feat:
            return features, out
        return out

# class CIFARNet(nn.Module):
#     def __init__(self, num_classes=10, in_channels=3):
#         super(CIFARNet, self).__init__()
#         self.input_shape = (in_channels, 32, 32)  # CIFAR默认尺寸
#         self.conv1 = nn.Conv2d(in_channels, 16, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(16, 32, 5, padding=1)
#         self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
#         self.flat_size = 64 * 3 * 3
#         self.linear = nn.Linear(self.flat_size, 128)
#         self.fc = nn.Linear(128, num_classes)
#         self.D = 128
#         self.cls = num_classes
#         # Define keys for the classifier layer
#         self.classifier_weight_keys = ['fc.weight', 'fc.bias']

#     def forward(self, x, return_feat=False):
#         x = self.pool(F.leaky_relu(self.conv1(x)))
#         x = self.pool(F.leaky_relu(self.conv2(x)))
#         x = self.pool(F.leaky_relu(self.conv3(x)))
#         x = x.view(-1, self.flat_size)
#         x = F.leaky_relu(self.linear(x))
#         out = self.fc(x)
#         if return_feat:
#             return x, out
#         return out

class EMNISTNet(nn.Module):
    def __init__(self, num_classes=62, in_channels=1):
        super(EMNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=1)
        self.flat_size = 32 * 5 * 5
        self.linear = nn.Linear(self.flat_size, 128)
        self.fc = nn.Linear(128, num_classes)
        self.D = 128
        self.cls = num_classes

    def forward(self, x, return_feat=False):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = x.view(-1, self.flat_size)
        x = F.leaky_relu(self.linear(x))
        out = self.fc(x)
        if return_feat:
            return x, out
        return out


class ImageNet(nn.Module):
    # 添加 input_size 参数，用于指定输入图片的尺寸
    def __init__(self, num_classes=10, in_channels=3, input_size=64): # Default to 64 for Tiny-ImageNet
        super(ImageNet, self).__init__()
        # 更新 input_shape 以反映实际输入的尺寸
        self.input_shape = (in_channels, input_size, input_size)

        # 定义卷积和池化层 (这部分网络结构不变)
        self.conv1 = nn.Conv2d(in_channels, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        # --- 动态计算 self.flat_size ---
        # 创建一个只包含特征提取部分 (conv + pool) 的临时 Sequential 模型
        # 需要确保这里的层和顺序与 forward 方法中的一致，包括激活函数
        self.features_temp = nn.Sequential(
            self.conv1,
            nn.LeakyReLU(), # 对应 forward 中的 F.leaky_relu
            self.pool,
            self.conv2,
            nn.LeakyReLU(),
            self.pool,
            self.conv3,
            nn.LeakyReLU(),
            self.pool
        )

        # 创建一个虚拟的输入张量，用于计算展平后的尺寸
        # 尺寸为 [批量大小=1, 通道数, 高, 宽]
        dummy_input = torch.randn(1, in_channels, input_size, input_size)

        # 将虚拟输入通过特征提取层
        dummy_output = self.features_temp(dummy_input)

        # 计算展平后的元素个数 (排除批量大小维度)
        self.flat_size = torch.flatten(dummy_output, 1).size(1)

        # --- 使用计算出的 self.flat_size 定义全连接层 ---
        self.linear = nn.Linear(self.flat_size, 128)
        self.fc = nn.Linear(128, num_classes)

        self.D = 128
        self.cls = num_classes
        # Define keys for the classifier layer
        self.classifier_weight_keys = ['fc.weight', 'fc.bias']

    def forward(self, x, return_feat=False):
        # 直接使用 self.features_temp 来处理特征提取部分，更简洁
        # 或者保持原样使用单独的层调用，但确保与 __init__ 中的顺序一致
        # 这里保持原样，但逻辑上等同于 self.features_temp(x)
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))

        # 使用计算出的 self.flat_size 进行展平操作
        x = x.view(-1, self.flat_size)

        # 传递给全连接层
        x = F.leaky_relu(self.linear(x))
        out = self.fc(x)

        if return_feat:
            return x, out
        return out
