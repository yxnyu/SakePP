import torch
import torch.nn as nn
import torch.nn.functional as F
from .sakeLayer import SAKEInteractionLayer
from .dgn import DGN
from .modules import (
    GradientScale, FeatureRecalibration,
    BalancedFusion, AdaptiveFusion)
from typing import List, Optional
import dgl


class SAKEPP(nn.Module):
    """SAKE++: A protein structure prediction model based on graph neural networks
    Innovative integration and extension of SAKE (Spatial Attention Protein Structure Prediction) and DGN (Directional Graph Neural Network)
    Main architecture includes:

    Core components:
    1. Feature encoder:
        - Input projection module: Map atomic feature vectors to a high-dimensional representation space
        - Position encoder: Capture spatial position information
    
    2. Dual-path feature extraction:
        - SAKE path: Use spatial attention mechanism to model long-range dependencies between atoms
        - DGN path: Extract local structural features through directional graph convolution
    
    3. Feature fusion and optimization:
        - Multi-level feature fusion: Combine balanced fusion and adaptive fusion strategies
        - Feature recalibration module: Dynamically adjust the weights of dual-path features
        - Residual enhancement mechanism: Maintain gradient flow and enhance feature expression
    
    4. Output processing:
        - Gradient balancer: Adaptively adjust the backward propagation of dual paths
        - Feature mapping layer: Project fused features to the target dimension space

    The model achieves precise modeling and prediction of protein structures through the above architecture.
    """
    def __init__(
        self,
        nr_atom_basis: int = 26,
        nr_edge_basis: int = 32,
        nr_edge_basis_hidden: int = 64,
        nr_atom_basis_hidden: int = 64,
        nr_atom_basis_spatial_hidden: int = 32,
        nr_atom_basis_spatial: int = 32,
        nr_atom_basis_velocity: int = 16,
        nr_coefficients: int = 8,
        nr_heads: int = 8,
        activation: nn.Module = nn.SiLU(),
        maximum_interaction_radius: float = 10.0,
        number_of_radial_basis_functions: int = 64,
        epsilon: float = 1e-6,
        scale_factor: float = 1.0,
        extra_node_features: int = 0,
        num_layers: int = 4,
        dropout_rate: float = 0.2,
        out_size: int = 256,
        input_size: int = 26,
        output_layer_sizes: Optional[List[int]] = None,
        dgn_hidden_size: int = 128,
        dgn_num_layers: int = 6,
        fusion_temperature: float = 0.5  # Add fusion temperature parameter
    ):
        super(SAKEPP, self).__init__()

        self.nr_atom_basis = nr_atom_basis
        self.feature_dim = dgn_hidden_size
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        # Initialization layer
        self.initial_linear = nn.Linear(input_size, nr_atom_basis)
        self.initial_activation = activation
        self.initial_norm = nn.LayerNorm(nr_atom_basis)
        self.initial_dropout = nn.Dropout(dropout_rate)

        # Gradient scaler
        self.sake_grad_scale = GradientScale(0.5)
        self.dgn_grad_scale = GradientScale(2)

        # SAKE layer component
        self.sake_layers = nn.ModuleList()
        self.sake_norms = nn.ModuleList()
        self.additional_linear_layers = nn.ModuleList()
        self.sake_residual_weights = nn.ParameterList()

        # Feature recalibration
        self.sake_recalibration = FeatureRecalibration(nr_atom_basis, reduction=4)
        self.dgn_recalibration = FeatureRecalibration(nr_atom_basis, reduction=4)

        for _ in range(num_layers):
            self.sake_layers.append(
                SAKEInteractionLayer(
                    nr_atom_basis=nr_atom_basis,
                    nr_edge_basis=nr_edge_basis,
                    nr_edge_basis_hidden=nr_edge_basis_hidden,
                    nr_atom_basis_hidden=nr_atom_basis_hidden,
                    nr_atom_basis_spatial_hidden=nr_atom_basis_spatial_hidden,
                    nr_atom_basis_spatial=nr_atom_basis_spatial,
                    nr_atom_basis_velocity=nr_atom_basis_velocity,
                    nr_coefficients=nr_coefficients,
                    nr_heads=nr_heads,
                    activation=activation,
                    maximum_interaction_radius=maximum_interaction_radius,
                    number_of_radial_basis_functions=number_of_radial_basis_functions,
                    epsilon=epsilon,
                    scale_factor=scale_factor,
                    extra_node_features=extra_node_features
                )
            )
            self.sake_norms.append(nn.LayerNorm(nr_atom_basis))
            self.additional_linear_layers.append(
                nn.Sequential(
                    nn.Linear(nr_atom_basis, nr_atom_basis_hidden),
                    activation,
                    nn.LayerNorm(nr_atom_basis_hidden),
                    nn.Dropout(dropout_rate),
                    nn.Linear(nr_atom_basis_hidden, nr_atom_basis)
                )
            )
            self.sake_residual_weights.append(nn.Parameter(torch.ones(1)))

        # DGN component
        self.dgn = DGN(
            node_feat_size=input_size,
            edge_feat_size=17,
            hidden_size=dgn_hidden_size,
            out_size=nr_atom_basis,
            num_layers=dgn_num_layers,
            dropout_rate=dropout_rate
        )

        # Feature alignment layer
        self.sake_align = nn.Sequential(
            nn.Linear(nr_atom_basis, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.dgn_align = nn.Sequential(
            nn.Linear(nr_atom_basis, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Feature fusion component
        self.balanced_fusion = BalancedFusion(
            dim=self.feature_dim,  # Pass in feature dimension
            temperature=fusion_temperature  # Pass in temperature parameter
        )
        self.adaptive_fusion = AdaptiveFusion(
            dim=self.feature_dim,
            num_heads=4
        )

        # Feature enhancement layer
        self.sake_enhancement = nn.Sequential(
            nn.Linear(nr_atom_basis, nr_atom_basis),
            nn.LayerNorm(nr_atom_basis),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(nr_atom_basis, nr_atom_basis)
        )

        self.dgn_enhancement = nn.Sequential(
            nn.Linear(nr_atom_basis, nr_atom_basis),
            nn.LayerNorm(nr_atom_basis),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(nr_atom_basis, nr_atom_basis)
        )

        # Post-processing component
        self.combined_processing = nn.Sequential(
            nn.Linear(self.feature_dim, nr_atom_basis),
            activation,
            nn.LayerNorm(nr_atom_basis),
            nn.Dropout(dropout_rate),
            nn.Linear(nr_atom_basis, nr_atom_basis),
            nn.LayerNorm(nr_atom_basis)
        )

        # Output projection layer
        if output_layer_sizes is None:
            output_layer_sizes = [out_size, out_size // 2]

        output_layers = []
        input_dim = nr_atom_basis
        for output_dim in output_layer_sizes:
            output_layers.extend([
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim),
                activation,
                nn.Dropout(dropout_rate)
            ])
            input_dim = output_dim
        output_layers.append(nn.Linear(input_dim, 1))
        self.output_projection = nn.Sequential(*output_layers)

        # Training progress tracking
        self.register_buffer('training_step', torch.tensor(0))
        self.progressive_weight = nn.Parameter(torch.ones(1))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, a=0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def get_progressive_weight(self):
        """Get progressive weight
        
        Returns a smooth progressive weight based on the training step,
        used to maintain consistent feature fusion during training and validation.
        """
        step = self.training_step.item()
        return torch.sigmoid(
            self.progressive_weight * (step / 1000)
            )  # Remove training judgment

    def forward(self, batched_graph, h, x, v, pairlist, edge_attr):
        device = h.device
        batch_size = batched_graph.batch_size
        batch_num_nodes = batched_graph.batch_num_nodes()
        max_num_nodes = max(batch_num_nodes)
    
        # Create mask
        node_mask = torch.zeros((batch_size, max_num_nodes), device=device, dtype=torch.bool)
        start_idx = 0
        for i, num_nodes in enumerate(batch_num_nodes):
            node_mask[i, :num_nodes] = True
    
        # SAKE path
        h_sake = self.initial_linear(h)
        h_sake = self.initial_activation(h_sake)
        h_sake = self.initial_norm(h_sake)
        h_sake = self.initial_dropout(h_sake)
    
        h_sake_padded = torch.zeros(batch_size, max_num_nodes, self.nr_atom_basis, device=device)
        start_idx = 0
        for i, num_nodes in enumerate(batch_num_nodes):
            h_sake_padded[i, :num_nodes] = h_sake[start_idx:start_idx + num_nodes]
            start_idx += num_nodes
    
        sake_intermediates = []
        prog_weight = self.get_progressive_weight()  # Use progressive weight uniformly
    
        # SAKE layer processing
        for idx, (layer, norm, additional_layer) in enumerate(zip(
            self.sake_layers, self.sake_norms, self.additional_linear_layers)):
            
            identity = h_sake_padded
            h_sake_flat = h_sake_padded[node_mask].view(-1, self.nr_atom_basis)
            h_res, x_new, v_new = layer(h_sake_flat, x, v, pairlist)
            
            h_res_padded = torch.zeros_like(h_sake_padded)
            start_idx = 0
            for i, num_nodes in enumerate(batch_num_nodes):
                h_res_padded[i, :num_nodes] = h_res[start_idx:start_idx + num_nodes]
                start_idx += num_nodes
            
            h_res_padded = additional_layer(h_res_padded)
            residual_weight = F.softplus(self.sake_residual_weights[idx])
            h_sake_padded = prog_weight * h_res_padded + (1 - prog_weight) * identity * residual_weight
            h_sake_padded = norm(h_sake_padded)
            h_sake_padded = self.sake_recalibration(h_sake_padded, node_mask)
            
            sake_intermediates.append(h_sake_padded)
            x, v = x_new, v_new

        # Apply gradient scaling uniformly
        h_sake_multi = torch.stack(sake_intermediates, dim=0).mean(dim=0)
        h_sake = self.sake_grad_scale(h_sake_multi)  # Always apply scaling
        h_sake = self.sake_enhancement(h_sake)

        # DGN path - Apply scaling uniformly
        h_dgn = self.dgn(batched_graph, h, edge_attr)
        h_dgn_padded = torch.zeros_like(h_sake_padded)
        start_idx = 0
        for i, num_nodes in enumerate(batch_num_nodes):
            h_dgn_padded[i, :num_nodes] = h_dgn[start_idx:start_idx + num_nodes]
            start_idx += num_nodes

        h_dgn = self.dgn_grad_scale(h_dgn_padded)  # Always apply scaling
        h_dgn = self.dgn_enhancement(h_dgn)
        h_dgn = self.dgn_recalibration(h_dgn, node_mask)

        # Feature alignment and fusion - Maintain consistent behavior
        h_sake_aligned = self.sake_align(h_sake)
        h_dgn_aligned = self.dgn_align(h_dgn)

        h_balanced = self.balanced_fusion(h_sake_aligned, h_dgn_aligned, node_mask)
        h_adaptive = self.adaptive_fusion(h_sake_aligned, h_dgn_aligned, node_mask)
        h_combined = h_balanced + h_adaptive

        h_final = self.combined_processing(h_combined)
        h_final = h_final[node_mask].view(-1, self.nr_atom_basis)

        # Update graph features
        batched_graph.ndata['h'] = h_final
        graph_repr = dgl.mean_nodes(batched_graph, 'h')
        output = self.output_projection(graph_repr)

        # Update training step - Only update during training
        if self.training:
            self.training_step += 1
            
        return output

    def reset_parameters(self):
        def _reset(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        
        self.apply(_reset)
        self.training_step.zero_()
        nn.init.ones_(self.progressive_weight)
