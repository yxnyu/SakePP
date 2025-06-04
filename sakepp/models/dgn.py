import dgl
import torch.nn as nn
from dgl.nn import DGNConv
import torch
from .modules import (
    GradientScale, FeatureRecalibration, 
    BalancedFusion, AdaptiveFusion )
from typing import List, Optional, Tuple
import torch.nn.functional as F


class DGN(nn.Module):
    """Implementation of Directional Graph Networks

    This model is implemented based on DGL's DGNConv layer and contains the following components:
    1. Input projection layer: Projects input node features to hidden dimension
    2. Multiple DGN convolutions: Uses various aggregators and scalers for graph convolution
    3. Inter-layer processing: Normalization, activation and dropout after each layer
    4. Output projection layer: Maps final features to target dimension

    Mainly used for processing graph representations of protein structures, capable of 
    capturing spatial directional information
    """
    def __init__(
        self,
        node_feat_size,
        edge_feat_size,
        hidden_size,
        out_size,
        num_layers=4,
        dropout_rate=0.15
        ):
        super(DGN, self).__init__()
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(node_feat_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Define aggregators and scalers
        aggregators = ['mean', 'max', 'sum', 'dir1-av', 'dir1-dx']
        scalers = ['identity', 'amplification', 'attenuation']

        # DGN layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = DGNConv(
                in_size=hidden_size,
                out_size=hidden_size,
                aggregators=aggregators,
                scalers=scalers,
                delta=1.0,
                edge_feat_size=edge_feat_size,
                dropout=dropout_rate
            )
            self.layers.append(layer)

        # Inter-layer processing
        self.intermediate = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ) for _ in range(num_layers)
        ])

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, out_size)
        )

        # Residual connection scaling factor
        self.residual_scale = nn.Parameter(torch.ones(num_layers))

        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, g, node_features, edge_feat):
        """Forward propagation"""
        # Move graph and features to same device
        device = node_features.device
        g = g.to(device)
        edge_feat = edge_feat.to(device)

        # Input feature processing
        h = self.input_projection(node_features)

        # Calculate Laplacian eigenvectors
        laplacian_pe = dgl.lap_pe(g, k=3, padding=True)
        g.ndata['eig'] = laplacian_pe.to(device)
        eig_vec = g.ndata['eig']

        # Pass through DGN layers
        for i, (layer, inter) in enumerate(zip(self.layers, self.intermediate)):
            # Save residual connection
            identity = h

            # Layer processing
            h_new = layer(g, h, edge_feat=edge_feat, eig_vec=eig_vec)
            h_new = inter(h_new)

            # Residual connection
            residual_weight = F.sigmoid(self.residual_scale[i])
            h = h_new + residual_weight * identity

        # Output processing
        return self.output_projection(h)


class DGNLinearFusion(nn.Module):
    """Fusion model of DGN and linear network

    This model fuses the features of DGN and linear network, containing the following main components:
    - Linear network path: Multi-layer linear transformations process input features
    - DGN path: Use DGN to process graph structured data
    - Feature fusion: Use both balanced fusion and adaptive fusion
    - Feature recalibration: Recalibrate features from both paths
    - Feature enhancement: Enhance feature representations of both paths
    - Gradient scaling: Control the gradient ratio of both paths

    Parameters:
        input_size (int): Input feature dimension, default 26
        hidden_size (int): Hidden layer dimension, default 128
        dropout_rate (float): Dropout rate, default 0.2
        out_size (int): Output dimension, default 256
        output_layer_sizes (List[int]): Output layer dimension list, default None
        dgn_hidden_size (int): DGN hidden layer dimension, default 128
        dgn_num_layers (int): DGN layer number, default 6
        node_feat_size (int): Node feature dimension, default 26
        edge_feat_size (int): Edge feature dimension, default 17
        fusion_temperature (float): Fusion temperature parameter, default 0.5
    """
    def __init__(
        self,
        input_size: int = 26,
        hidden_size: int = 128,
        dropout_rate: float = 0.2,
        out_size: int = 256,
        output_layer_sizes: Optional[List[int]] = None,
        dgn_hidden_size: int = 128,
        dgn_num_layers: int = 6,
        node_feat_size: int = 26,
        edge_feat_size: int = 17,
        fusion_temperature: float = 0.5
    ):
        super(DGNLinearFusion, self).__init__()

        self.feature_dim = dgn_hidden_size
        self.dropout_rate = dropout_rate

        # Gradient scaler
        self.linear_grad_scale = GradientScale(0.5)
        self.dgn_grad_scale = GradientScale(2.0)

        # Linear network path
        self.linear_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, self.feature_dim),
            nn.LayerNorm(self.feature_dim)
        )

        # DGN component
        self.dgn = DGN(
            node_feat_size=node_feat_size,
            edge_feat_size=edge_feat_size,
            hidden_size=dgn_hidden_size,
            out_size=self.feature_dim,
            num_layers=dgn_num_layers,
            dropout_rate=dropout_rate
        )

        # Feature recalibration
        self.linear_recalibration = FeatureRecalibration(self.feature_dim, reduction=4)
        self.dgn_recalibration = FeatureRecalibration(self.feature_dim, reduction=4)

        # Feature enhancement
        self.linear_enhancement = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        self.dgn_enhancement = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        # Feature alignment layer
        self.linear_align = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.dgn_align = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Feature fusion component
        self.balanced_fusion = BalancedFusion(
            dim=self.feature_dim,
            temperature=fusion_temperature
        )
        self.adaptive_fusion = AdaptiveFusion(
            dim=self.feature_dim,
            num_heads=4
        )

        # Post-processing component
        self.combined_processing = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim)
        )

        # Output projection layer
        if output_layer_sizes is None:
            output_layer_sizes = [out_size, out_size // 2]

        output_layers = []
        input_dim = self.feature_dim
        for output_dim in output_layer_sizes:
            output_layers.extend([
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = output_dim
        output_layers.append(nn.Linear(input_dim, 1))
        self.output_projection = nn.Sequential(*output_layers)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, a=0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, batched_graph, h, x, v, pairlist, edge_attr):
        """
        Forward propagation
        Args:
            batched_graph: Batch processed graph
            h: Node features
            x: Node coordinates
            v: Node velocities
            pairlist: Edge connection
            edge_attr: Edge features
        """
        device = h.device
        batch_size = batched_graph.batch_size
        batch_num_nodes = batched_graph.batch_num_nodes()
        max_num_nodes = max(batch_num_nodes)

        # Create mask
        node_mask = torch.zeros((batch_size, max_num_nodes), device=device, dtype=torch.bool)
        start_idx = 0
        for i, num_nodes in enumerate(batch_num_nodes):
            node_mask[i, :num_nodes] = True

        # Linear network path
        h_linear = self.linear_network(h)
        h_linear_padded = torch.zeros(batch_size, max_num_nodes, self.feature_dim, device=device)
        start_idx = 0
        for i, num_nodes in enumerate(batch_num_nodes):
            h_linear_padded[i, :num_nodes] = h_linear[start_idx:start_idx + num_nodes]
            start_idx += num_nodes

        h_linear = self.linear_grad_scale(h_linear_padded)
        h_linear = self.linear_enhancement(h_linear)
        h_linear = self.linear_recalibration(h_linear, node_mask)

        # DGN path
        h_dgn = self.dgn(batched_graph, h, edge_attr)  # Use DGN to process
        h_dgn_padded = torch.zeros_like(h_linear_padded)
        start_idx = 0
        for i, num_nodes in enumerate(batch_num_nodes):
            h_dgn_padded[i, :num_nodes] = h_dgn[start_idx:start_idx + num_nodes]
            start_idx += num_nodes

        h_dgn = self.dgn_grad_scale(h_dgn_padded)
        h_dgn = self.dgn_enhancement(h_dgn)
        h_dgn = self.dgn_recalibration(h_dgn, node_mask)

        # Feature alignment and fusion
        h_linear_aligned = self.linear_align(h_linear)
        h_dgn_aligned = self.dgn_align(h_dgn)

        h_balanced = self.balanced_fusion(h_linear_aligned, h_dgn_aligned, node_mask)
        h_adaptive = self.adaptive_fusion(h_linear_aligned, h_dgn_aligned, node_mask)
        h_combined = h_balanced + h_adaptive

        h_final = self.combined_processing(h_combined)
        h_final = h_final[node_mask].view(-1, self.feature_dim)

        # Update graph features
        batched_graph.ndata['h'] = h_final
        graph_repr = dgl.mean_nodes(batched_graph, 'h')
        output = self.output_projection(graph_repr)

        return output