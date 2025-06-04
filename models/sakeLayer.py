from torch import nn
from .modules import MultiplySigmoid
from modelforge.potential.utils import DenseWithCustomDist, scatter_softmax
from modelforge.potential.representation import PhysNetRadialBasisFunction
import torch
from typing import Optional


class SAKEInteractionLayer(nn.Module):
    """SAKE Interaction Layer

    This layer is used to process interatomic interactions in protein structures, containing the following main components:
    1. Radial Basis Function Module: Calculate radial basis functions for interatomic distances
    2. Node MLP: Process node features
    3. Post-normalization MLP: Process combined features
    """
    def __init__(
        self, 
        nr_atom_basis: int,
        nr_edge_basis: int,
        nr_edge_basis_hidden: int,
        nr_atom_basis_hidden: int,
        nr_atom_basis_spatial_hidden: int,
        nr_atom_basis_spatial: int,
        nr_atom_basis_velocity: int,
        nr_coefficients: int,
        nr_heads: int,
        activation: nn.Module,
        maximum_interaction_radius: float,
        number_of_radial_basis_functions: int,
        epsilon: float,
        scale_factor: float,
        extra_node_features: int = 0
    ):
        super().__init__()
        self.nr_atom_basis = nr_atom_basis
        self.nr_edge_basis = nr_edge_basis
        self.nr_edge_basis_hidden = nr_edge_basis_hidden
        self.nr_atom_basis_hidden = nr_atom_basis_hidden
        self.nr_atom_basis_spatial_hidden = nr_atom_basis_spatial_hidden
        self.nr_atom_basis_spatial = nr_atom_basis_spatial
        self.nr_atom_basis_velocity = nr_atom_basis_velocity
        self.nr_coefficients = nr_coefficients
        self.nr_heads = nr_heads
        self.epsilon = epsilon

        # Radial Basis Function
        self.radial_symmetry_function_module = PhysNetRadialBasisFunction(
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            max_distance=maximum_interaction_radius,
            dtype=torch.float32,
        )

        # Node MLP
        self.node_mlp = nn.Sequential(
            DenseWithCustomDist(
                self.nr_atom_basis
                + self.nr_heads * self.nr_edge_basis
                + self.nr_atom_basis_spatial,
                self.nr_atom_basis_hidden,
                activation_function=activation,
            ),
            nn.LayerNorm(self.nr_atom_basis_hidden),
            DenseWithCustomDist(
                self.nr_atom_basis_hidden,
                self.nr_atom_basis,
                activation_function=activation,
            ),
            nn.LayerNorm(self.nr_atom_basis)
        )

        # Post-normalization MLP
        self.post_norm_mlp = nn.Sequential(
            DenseWithCustomDist(
                self.nr_coefficients,
                self.nr_atom_basis_spatial_hidden,
                activation_function=activation,
            ),
            nn.LayerNorm(self.nr_atom_basis_spatial_hidden),
            DenseWithCustomDist(
                self.nr_atom_basis_spatial_hidden,
                self.nr_atom_basis_spatial,
                activation_function=activation,
            ),
            nn.LayerNorm(self.nr_atom_basis_spatial)
        )

        # Edge MLP
        self.edge_mlp_in = nn.Linear(
            self.nr_atom_basis * 2,
            number_of_radial_basis_functions
        )

        self.edge_mlp_out = nn.Sequential(
            DenseWithCustomDist(
                self.nr_atom_basis * 2 + number_of_radial_basis_functions + 1,
                self.nr_edge_basis_hidden,
                activation_function=activation,
            ),
            nn.LayerNorm(self.nr_edge_basis_hidden),
            nn.Linear(nr_edge_basis_hidden, nr_edge_basis),
            nn.LayerNorm(nr_edge_basis)
        )

        # Semantic attention MLP, used to calculate the attention weights of edge features
        self.semantic_attention_mlp = DenseWithCustomDist(
            self.nr_edge_basis,
            self.nr_heads,
            activation_function=nn.CELU(alpha=2.0)
        )

        # Velocity MLP - a multi-layer perceptron used to predict atomic velocities
        # Contains three main components:
        # 1. Feature transformation layer: Convert atomic features to velocity feature space
        # 2. Layer normalization: Standardize feature distribution
        # 3. Output layer: Use multiplicative Sigmoid activation function to predict velocity
        self.velocity_mlp = nn.Sequential(
            DenseWithCustomDist(
                self.nr_atom_basis,
                self.nr_atom_basis_velocity,
                activation_function=activation,
            ),
            nn.LayerNorm(self.nr_atom_basis_velocity),
            DenseWithCustomDist(
                self.nr_atom_basis_velocity,
                1,
                activation_function=MultiplySigmoid(factor=2.0),
                bias=False,
            )
        )

        # mixing MLPs
        self.x_mixing_mlp = nn.Sequential(
            DenseWithCustomDist(
                self.nr_heads * self.nr_edge_basis,
                self.nr_coefficients,
                bias=False,
                activation_function=nn.Tanh(),
            ),
            nn.LayerNorm(self.nr_coefficients)
        )

        self.v_mixing_mlp = DenseWithCustomDist(
            self.nr_coefficients,
            1,
            bias=False
        )

        self.scale_factor_in_nanometer = scale_factor

    def update_edge(self, h_i_by_pair, h_j_by_pair, d_ij):
        """Update edge features"""
        h_ij_cat = torch.cat([h_i_by_pair, h_j_by_pair], dim=-1)
        h_ij_filtered = self.radial_symmetry_function_module(
            d_ij.unsqueeze(-1)
        ).squeeze(-2) * self.edge_mlp_in(h_ij_cat)

        return self.edge_mlp_out(
            torch.cat([
                h_ij_cat,
                h_ij_filtered,
                d_ij.unsqueeze(-1) / self.scale_factor_in_nanometer
            ], dim=-1)
        )

    def update_node(self, h, h_i_semantic, h_i_spatial):
        """Update node features"""
        return h + self.node_mlp(torch.cat([h, h_i_semantic, h_i_spatial], dim=-1))

    def update_velocity(self, v, h, combinations, idx_i):
        """Update velocity"""
        v_ij = self.v_mixing_mlp(combinations.transpose(-1, -2)).squeeze(-1)
        expanded_idx_i = idx_i.view(-1, 1).expand_as(v_ij)
        dv = torch.zeros_like(v).scatter_reduce(
            0, expanded_idx_i, v_ij, "mean", include_self=False
        )
        return self.velocity_mlp(h) * v + dv

    def get_combinations(self, h_ij_semantic, dir_ij):
        """Get combined features"""
        return torch.einsum("px,pc->pcx", dir_ij, self.x_mixing_mlp(h_ij_semantic))

    def get_spatial_attention(self, combinations: torch.Tensor, idx_i: torch.Tensor, nr_atoms: int):
        """Get spatial attention"""
        expanded_idx_i = idx_i.view(-1, 1, 1).expand_as(combinations)
        out_shape = (nr_atoms, self.nr_coefficients, combinations.shape[-1])
        zeros = torch.zeros(
            out_shape, dtype=combinations.dtype, device=combinations.device
        )
        combinations_mean = zeros.scatter_reduce(
            0, expanded_idx_i, combinations, "mean", include_self=False
        )
        combinations_norm_square = (combinations_mean**2).sum(dim=-1)
        return self.post_norm_mlp(combinations_norm_square)

    def aggregate(self, h_ij_semantic: torch.Tensor, idx_i: torch.Tensor, nr_atoms: int):
        """Feature aggregation"""
        expanded_idx_i = idx_i.view(-1, 1).expand_as(h_ij_semantic)
        out_shape = (nr_atoms, self.nr_heads * self.nr_edge_basis)
        zeros = torch.zeros(
            out_shape, dtype=h_ij_semantic.dtype, device=h_ij_semantic.device
        )
        return zeros.scatter_add(0, expanded_idx_i, h_ij_semantic)

    def get_semantic_attention(
            self, h_ij_edge: torch.Tensor, idx_i: torch.Tensor,
            idx_j: torch.Tensor, nr_atoms: int):
        """Get semantic attention"""
        h_ij_att_weights = self.semantic_attention_mlp(h_ij_edge) - (
            torch.eq(idx_i, idx_j) * 1e5
        ).unsqueeze(-1)
        expanded_idx_i = idx_i.view(-1, 1).expand_as(h_ij_att_weights)
        combined_ij_att = scatter_softmax(
            h_ij_att_weights,
            expanded_idx_i,
            dim=0,
            dim_size=nr_atoms,
        )
        return torch.reshape(
            torch.einsum("pf,ph->pfh", h_ij_edge, combined_ij_att),
            (len(idx_i), self.nr_edge_basis * self.nr_heads),
        )

    def forward(
            self, h: torch.Tensor, x: torch.Tensor, v: torch.Tensor,
            pairlist: torch.Tensor,
            extra_features: Optional[torch.Tensor] = None):
        """Forward propagation"""
        if extra_features is not None:
            h = torch.cat([h, extra_features], dim=-1)

        if self.node_mlp[0].in_features != h.shape[-1]:
            self.node_mlp[0] = DenseWithCustomDist(
                h.shape[-1] + self.nr_heads * self.nr_edge_basis + self.nr_atom_basis_spatial,
                self.nr_atom_basis_hidden,
                activation_function=self.node_mlp[0].activation_function
            ).to(h.device)

        idx_i, idx_j = pairlist.unbind(0)
        nr_of_atoms_in_all_systems = int(x.size(dim=0))
        r_ij = x[idx_j] - x[idx_i]
        d_ij = torch.sqrt((r_ij**2).sum(dim=1) + self.epsilon)
        dir_ij = r_ij / (d_ij.unsqueeze(-1) + self.epsilon)

        h_ij_edge = self.update_edge(h[idx_j], h[idx_i], d_ij)
        h_ij_semantic = self.get_semantic_attention(
            h_ij_edge, idx_i, idx_j, nr_of_atoms_in_all_systems
        )
        h_i_semantic = self.aggregate(h_ij_semantic, idx_i, nr_of_atoms_in_all_systems)
        combinations = self.get_combinations(h_ij_semantic, dir_ij)
        h_i_spatial = self.get_spatial_attention(
            combinations, idx_i, nr_of_atoms_in_all_systems
        )
        h_updated = self.update_node(h, h_i_semantic, h_i_spatial)
        v_updated = self.update_velocity(v, h_updated, combinations, idx_i)
        x_updated = x + v_updated

        return h_updated, x_updated, v_updated