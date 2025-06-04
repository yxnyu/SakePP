import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiplySigmoid(nn.Module):
    """Multiplicative Sigmoid activation function

        该模块实现了一种乘性Sigmoid激活函数,主要包含以下组件:
        1. 乘性因子: 控制Sigmoid的缩放
        2. Sigmoid激活函数: 将输入值缩放到[0,1]区间
        3. 乘法操作: 将Sigmoid输出与一个预定义的乘性因子相乘
    """
    def __init__(self, factor: float = 2.0):
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.factor * torch.sigmoid(x)


class GradientScale(nn.Module):
    """Gradient Scaling Module - Training and Validation Consistency

    This module is used to maintain consistent gradient scaling during training and validation,
    mainly containing the following components:
    1. Scaling factor: Controls the scaling ratio of gradients
    2. Scaling operation: Multiplies the input value by a predefined scaling factor
    """
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        
    def forward(self, x):
        return x * self.scale  # Scale regardless of training or validation


class FeatureRecalibration(nn.Module):
    """Feature Recalibration Module

    This module is used for feature recalibration, mainly containing the following components:
    1. Average pooling layer: Compresses input features to 1D
    2. Fully connected layer: Reduces feature dimensions
    3. Activation function: ReLU
    4. Fully connected layer: Restores feature dimensions
    5. Sigmoid activation function: Scales output values to [0,1] range
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = channels
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x, mask=None):
        # The shape of input x should be (batch_size, num_nodes, channels)
        batch_size, num_nodes, channels = x.shape
        
        if mask is not None:
            # Apply mask
            x = x * mask.unsqueeze(-1)
        
        # Convert dimensions to (batch_size * num_nodes, channels, 1)
        x_reshaped = x.view(-1, channels, 1)
        
        # Calculate channel attention
        y = self.avg_pool(x_reshaped).squeeze(-1)  # (batch_size * num_nodes, channels)
        y = self.fc(y)  # (batch_size * num_nodes, channels)
        
        # Apply attention
        y = y.view(batch_size, num_nodes, channels)
        
        # If there is a mask, ensure the values in the masked region are 0
        if mask is not None:
            y = y * mask.unsqueeze(-1)
            
        return x * y


class BalancedFusion(nn.Module):
    """Balanced Feature Fusion Module

    This module is used for balanced feature fusion, mainly containing the following components:
    1. Sake gate: Process padding data
    2. Dgn gate: Process padding data
    3. Sigmoid activation function: Scales output values to [0,1] range
    """
    def __init__(self, dim, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        # Modify the gate network to process padding data
        self.sake_gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        self.dgn_gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, sake_feat, dgn_feat, mask=None):
        # sake_feat, dgn_feat shape: [batch_size, max_num_nodes, dim]
        batch_size, max_num_nodes, dim = sake_feat.shape

        # Calculate attention weights
        sake_weight = self.sake_gate(sake_feat)  # [batch_size, max_num_nodes, 1]
        dgn_weight = self.dgn_gate(dgn_feat)    # [batch_size, max_num_nodes, 1]

        # Connect weights and apply softmax
        weights = torch.cat([sake_weight, dgn_weight], dim=-1)  # [batch_size, max_num_nodes, 2]
        weights = F.softmax(weights / self.temperature, dim=-1)

        # Apply mask (if provided)
        if mask is not None:
            weights = weights * mask.unsqueeze(-1).float()

        # Fusion features
        fused = (weights[..., 0].unsqueeze(-1) * sake_feat + 
                weights[..., 1].unsqueeze(-1) * dgn_feat)

        return fused


class AdaptiveFusion(nn.Module):
    """Adaptive Feature Fusion Module

    This module is used for adaptive feature fusion, mainly containing the following components:
    1. Layer normalization: Normalize input features
    2. Linear transformation: Convert input features to query, key, and value
    3. Multi-head attention: Calculate attention weights
    4. Linear transformation: Convert output features to final dimension
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Create Q, K, V transformations for each feature stream
        self.to_q1 = nn.Linear(dim, dim)
        self.to_k1 = nn.Linear(dim, dim)
        self.to_v1 = nn.Linear(dim, dim)

        self.to_q2 = nn.Linear(dim, dim)
        self.to_k2 = nn.Linear(dim, dim)
        self.to_v2 = nn.Linear(dim, dim)

        self.to_out = nn.Linear(dim * 2, dim)

    def forward(self, x1, x2, mask=None):
        """
        x1, x2: [batch_size, max_num_nodes, dim]
        mask: [batch_size, max_num_nodes]
        """
        batch_size, max_num_nodes, _ = x1.shape

        # Process input features with layer normalization
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)

        # Attention of the first feature stream
        q1 = self.to_q1(x1).view(batch_size, max_num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        k1 = self.to_k1(x1).view(batch_size, max_num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        v1 = self.to_v1(x1).view(batch_size, max_num_nodes, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention of the second feature stream
        q2 = self.to_q2(x2).view(batch_size, max_num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        k2 = self.to_k2(x2).view(batch_size, max_num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        v2 = self.to_v2(x2).view(batch_size, max_num_nodes, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate cross-stream attention
        attn1 = torch.matmul(q1, k2.transpose(-2, -1)) * self.scale  # Stream 1 attention to Stream 2
        attn2 = torch.matmul(q2, k1.transpose(-2, -1)) * self.scale  # Stream 2 attention to Stream 1

        if mask is not None:
            # [batch_size, 1, 1, max_num_nodes]
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn1 = attn1.masked_fill(~mask, float('-inf'))
            attn2 = attn2.masked_fill(~mask, float('-inf'))

        attn1 = F.softmax(attn1, dim=-1)
        attn2 = F.softmax(attn2, dim=-1)

        # Calculate attention output
        out1 = torch.matmul(attn1, v2)  # Stream 1 attention to Stream 2 features
        out2 = torch.matmul(attn2, v1)  # Stream 2 attention to Stream 1 features

        # Reshape dimensions
        out1 = out1.transpose(1, 2).contiguous().view(batch_size, max_num_nodes, self.dim)
        out2 = out2.transpose(1, 2).contiguous().view(batch_size, max_num_nodes, self.dim)

        # Connect and fuse two feature streams
        out = torch.cat([out1, out2], dim=-1)
        out = self.to_out(out)  # Fusion to original dimension

        return out