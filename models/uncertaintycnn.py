import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFARNet_EDL(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, dropout_rate=0.2):
        super(CIFARNet_EDL, self).__init__()
        self.num_classes = num_classes
        self.input_shape = (in_channels, 32, 32)  # CIFAR default size
        self.D = 128  # Feature dimension, required by server_base.py
        self.temperature = 1.0
        
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(dropout_rate)
        
        # Calculate flat_size dynamically based on dummy input
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 32, 32)
            x_calc = self._forward_features(dummy_input)
            self.flat_size = x_calc.view(-1).shape[0]

        self.fc1 = nn.Linear(self.flat_size, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout_fc = nn.Dropout(dropout_rate)
        
        # Output from this layer will be used to compute evidence
        self.fc_evidence = nn.Linear(128, num_classes) 
        
        # To identify parameters of the feature extractor vs. classifier head if needed for partial aggregation later
        self.feature_extractor_params = [
            p_name for p_name, _ in self.named_parameters() 
            if 'fc_evidence' not in p_name
        ]
        self.classifier_head_params = ['fc_evidence.weight', 'fc_evidence.bias']

    def _forward_features(self, x):
        # 第一个卷积块
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # 第二个卷积块
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # 第三个卷积块
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        return x

    def get_evidence(self, x_logits):
        """
        Transforms logits to evidence with improved scaling.
        Using softplus with temperature scaling for better calibration.
        """
        # 使用温度缩放改善校准
        return F.softplus(x_logits / self.temperature)

    def forward(self, x, return_feat_alpha=False):
        x = self._forward_features(x)
        x = x.reshape(-1, self.flat_size)  # Use reshape instead of view for better compatibility
        
        # 全连接层
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        
        features = F.relu(self.bn_fc2(self.fc2(x)))
        features = self.dropout_fc(features)

        evidence_logits = self.fc_evidence(features)

        if return_feat_alpha:
            evidence = self.get_evidence(evidence_logits)
            alpha = evidence + 1 
            return features, evidence_logits, alpha
        return evidence_logits  # Return raw logits for evidence computation in loss

    def predict(self, x):
        evidence_logits = self.forward(x, return_feat_alpha=False)
        evidence = self.get_evidence(evidence_logits)
        alpha = evidence + 1
        probs = alpha / torch.sum(alpha, dim=1, keepdim=True)  # Mean of Dirichlet
        return probs, alpha

# class CIFARNet_EDL(nn.Module):
#     """
#     增强版EDL CNN模型，支持多头不确定性估计和特征提取
#     """
#     def __init__(self, num_classes=10, in_channels=3):
#         super(CIFARNet_EDL, self).__init__()
#         self.num_classes = num_classes
        
#         # 特征提取器
#         self.features = nn.Sequential(
#             nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(64),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(64),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout(0.2),
            
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(128),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(128),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout(0.3),
            
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(256),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(256),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout(0.4),
#         )
        
#         # 分类器
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(256 * 4 * 4, 512),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm1d(512),
#             nn.Dropout(0.5),
#             nn.Linear(512, num_classes)
#         )
#         self.D = 512
#         # 初始化权重
#         self._initialize_weights()
    
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
    
#     def forward(self, x, return_features=False):
#         # 提取特征
#         features = self.features(x)
#         features_flat = torch.flatten(features, 1)
        
#         # 分类
#         logits = self.classifier(features_flat)
        
#         if return_features:
#             return logits, features_flat
#         else:
#             return logits
    
#     def get_evidence(self, logits):
#         """获取evidence"""
#         return F.softplus(logits)

class EMNISTNet_EDL(nn.Module):
    """
    EMNISTNet with Evidential Deep Learning (EDL) capabilities.
    This architecture is adapted for the EMNIST dataset (28x28 grayscale images)
    and incorporates best practices like BatchNorm and Dropout for improved performance
    and uncertainty quantification, making it suitable for high-quality research submissions.
    """
    def __init__(self, num_classes=62, in_channels=1, dropout_rate=0.2):
        super(EMNISTNet_EDL, self).__init__()
        self.num_classes = num_classes
        self.input_shape = (in_channels, 28, 28)  # EMNIST default size
        self.D = 128  # Feature dimension

        # Convolutional Block 1
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(dropout_rate)

        # Convolutional Block 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(dropout_rate)

        # Calculate flat_size dynamically based on a dummy input
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.input_shape)
            x_calc = self._forward_features(dummy_input)
            self.flat_size = x_calc.view(-1).shape[0]

        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flat_size, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout_fc1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(256, self.D) # To extract features of dimension D
        self.bn_fc2 = nn.BatchNorm1d(self.D)
        self.dropout_fc2 = nn.Dropout(dropout_rate)
        
        # Classifier Head (Evidence Computer)
        self.fc_evidence = nn.Linear(self.D, num_classes)

        # Parameter groups for potential partial aggregation in federated learning
        self.feature_extractor_params = [
            p_name for p_name, _ in self.named_parameters()
            if 'fc_evidence' not in p_name
        ]
        self.classifier_head_params = ['fc_evidence.weight', 'fc_evidence.bias']

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
        
        return x

    def get_evidence(self, x_logits):
        """
        Transforms logits to evidence with improved scaling.
        Using softplus with temperature scaling for better calibration.
        """
        return F.softplus(x_logits)

    def forward(self, x, return_feat_alpha=False):
        # Feature extraction
        x = self._forward_features(x)
        x = x.view(-1, self.flat_size)
        
        # Fully connected layers
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc1(x)
        
        features = F.relu(self.bn_fc2(self.fc2(x)))
        features = self.dropout_fc2(features)
        
        # Get logits for evidence
        evidence_logits = self.fc_evidence(features)

        if return_feat_alpha:
            evidence = self.get_evidence(evidence_logits)
            alpha = evidence + 1
            return features, evidence_logits, alpha
            
        return evidence_logits

    def predict(self, x):
        """
        Performs a prediction and returns class probabilities and alpha values.
        """
        evidence_logits = self.forward(x, return_feat_alpha=False)
        evidence = self.get_evidence(evidence_logits)
        alpha = evidence + 1
        # Calculate probabilities from the mean of the Dirichlet distribution
        probs = alpha / torch.sum(alpha, dim=1, keepdim=True)
        return probs, alpha

class ImageNet_EDL(nn.Module):
    """
    ImageNet-style CNN with Evidential Deep Learning (EDL) capabilities.
    This architecture is designed to be flexible for various image sizes (e.g., Tiny-ImageNet 64x64)
    by dynamically computing the flattened layer size. It incorporates robust features like
    BatchNorm and Dropout, making it suitable for publication-level research.
    """
    def __init__(self, num_classes=10, in_channels=3, input_size=64, dropout_rate=0.25):
        super(ImageNet_EDL, self).__init__()
        self.num_classes = num_classes
        self.input_shape = (in_channels, input_size, input_size)
        self.D = 128  # Feature dimension

        # Convolutional Block 1
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        # Convolutional Block 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        # Convolutional Block 3
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(dropout_rate)

        # Calculate flat_size dynamically based on a dummy input
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.input_shape)
            x_calc = self._forward_features(dummy_input)
            self.flat_size = x_calc.view(-1).shape[0]

        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flat_size, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout_fc1 = nn.Dropout(0.5) # Higher dropout for larger FC layer
        
        self.fc2 = nn.Linear(512, self.D) # To extract features of dimension D
        self.bn_fc2 = nn.BatchNorm1d(self.D)
        self.dropout_fc2 = nn.Dropout(0.5)

        # Classifier Head (Evidence Computer)
        self.fc_evidence = nn.Linear(self.D, num_classes)

        # Parameter groups for potential partial aggregation
        self.feature_extractor_params = [
            p_name for p_name, _ in self.named_parameters()
            if 'fc_evidence' not in p_name
        ]
        self.classifier_head_params = ['fc_evidence.weight', 'fc_evidence.bias']

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

    def get_evidence(self, x_logits):
        """
        Transforms logits to evidence.
        Using softplus to ensure evidence is positive.
        """
        return F.softplus(x_logits)

    def forward(self, x, return_feat_alpha=False):
        # Feature extraction
        x = self._forward_features(x)
        x = x.view(-1, self.flat_size)
        
        # Fully connected layers
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc1(x)
        
        features = F.relu(self.bn_fc2(self.fc2(x)))
        features = self.dropout_fc2(features)
        
        # Get logits for evidence
        evidence_logits = self.fc_evidence(features)

        if return_feat_alpha:
            evidence = self.get_evidence(evidence_logits)
            alpha = evidence + 1
            return features, evidence_logits, alpha
            
        return evidence_logits

    def predict(self, x):
        """
        Performs a prediction and returns class probabilities and alpha values.
        """
        evidence_logits = self.forward(x, return_feat_alpha=False)
        evidence = self.get_evidence(evidence_logits)
        alpha = evidence + 1
        # Calculate probabilities from the mean of the Dirichlet distribution
        probs = alpha / torch.sum(alpha, dim=1, keepdim=True)
        return probs, alpha
