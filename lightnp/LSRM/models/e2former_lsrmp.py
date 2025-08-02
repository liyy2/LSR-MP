# E2Former adaptation for LSR-MP framework
# Consistent with the equiformer adaptation approach using force blocks

import sys
import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter_mean, scatter
import math
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops

# Add E2Former repository to path
sys.path.append('/nfs/roberts/project/pi_mg269/yl2428/E2Former')

# Import E2Former components
from src.E2Former import E2FormerBackbone, eSCN_ForceBlockV2
from .output_net import OutputNet
from .torchmdnet.models.utils import vec_layernorm, max_min_norm, norm
from ..utils import conditional_grad


class E2FormerDataMapper:
    """
    Maps LSR-MP data format to E2Former data format.
    
    LSR-MP uses PyTorch Geometric format with:
    - atomic_numbers: atomic numbers [num_nodes] (actual atomic numbers, e.g., 1 for H, 6 for C)
    - pos: positions [num_nodes, 3] 
    - batch: batch indices [num_nodes]
    - labels: group labels [num_nodes]
    
    E2Former expects batched format with:
    - pos: positions [batch_size, max_nodes, 3]
    - atomic_numbers: atomic numbers [num_nodes] (then gets batched internally)
    - cell: unit cell [batch_size, 3, 3] 
    - pbc: periodic boundary conditions [batch_size, 3]
    - batch: batch indices [num_nodes]
    - ptr: batch pointer [batch_size + 1]
    """
    
    @staticmethod
    def map_lsrmp_to_e2former(pos: Tensor, atomic_numbers: Tensor, batch: Tensor) -> Data:
        """
        Convert LSR-MP input format to E2Former format.
        
        Args:
            pos: Node positions [num_nodes, 3]
            atomic_numbers: Atomic numbers [num_nodes] (actual atomic numbers, e.g., 1 for H, 6 for C)
            batch: Batch indices [num_nodes]
                
        Returns:
            E2Former format data 
        """
        device = pos.device
        num_graphs = batch.max().item() + 1
        
        # Subtract center of mass from positions (following GotenNet approach)
        pos_centered = pos.clone()
        com = scatter_mean(pos_centered, batch, dim=0)  # [num_graphs, 3]
        pos_centered = pos_centered - com[batch]  # Subtract CoM for each atom
        
        # Create batch pointers for E2Former
        ptr = torch.zeros(num_graphs + 1, dtype=torch.long, device=device)
        for i in range(num_graphs):
            ptr[i + 1] = ptr[i] + (batch == i).sum()
        
        # Handle cell and pbc - provide defaults for molecular systems
        # Default unit cell (large box for molecules)
        cell = torch.eye(3, device=device).unsqueeze(0).repeat(num_graphs, 1, 1) * 100.0
        
        # Default: no periodic boundary conditions for QM9/MD17
        pbc = torch.zeros((num_graphs, 3), dtype=torch.bool, device=device)
        
        # Create E2Former data format
        e2former_data = Data(
            pos=pos_centered,  # Center-of-mass centered positions [num_nodes, 3]
            atomic_numbers=atomic_numbers,  # Proper atomic numbers [num_nodes]
            cell=cell,  # [num_graphs, 3, 3]
            pbc=pbc,    # [num_graphs, 3]
            batch=batch,  # [num_nodes]
            ptr=ptr     # [num_graphs + 1]
        )
        
        return e2former_data


class E2Former(nn.Module):
    """
    E2Former model adapted for LSR-MP framework.
    
    This class provides an LSR-MP-compatible interface for the E2Former model,
    handling data format conversion and providing the expected output format.
    This is the vanilla E2Former model without long-short range message passing.
    """
    
    def __init__(self, 
                 regress_forces=True,
                 hidden_channels=128,
                 num_layers=6,
                 num_rbf=50,
                 rbf_type="expnorm",
                 trainable_rbf=True,
                 neighbor_embedding=True,
                 short_cutoff_upper=10,
                 long_cutoff_upper=10,
                 mean=None,
                 std=None,
                 atom_ref=None,
                 max_z=100,
                 group_center='center_of_mass',
                 tf_writer=None,
                 radius=5.0,
                 **kwargs):
        """
        Initialize E2Former for LSR-MP.
        
        Args:
            regress_forces: Whether to predict forces
            hidden_channels: Number of hidden channels
            num_layers: Number of layers
            radius: Cutoff radius
            mean: Task mean for normalization
            std: Task std for normalization
            **kwargs: Additional arguments
        """
        super().__init__()
        
        # Store LSR-MP-compatible parameters
        self.regress_forces = regress_forces
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.group_center = group_center
        self.tf_writer = tf_writer
        self.t = 0
        
        # Set E2Former configuration based on LSR-MP requirements
        e2former_config = {
            # Global configs  
            'regress_forces': regress_forces,
            'use_fp16_backbone': False,
            'use_compile': False,
            'encoder_embed_dim': hidden_channels,
            'hidden_size': hidden_channels,
            'encoder': 'dit',
            
            # PBC and radius configs
            'pbc_max_radius': radius,
            'max_radius': radius,
            'max_neighbors': 50,
            'pbc_expanded_num_cell_per_direction': 1,
            
            # DIT config
            'ffn_embedding_dim': hidden_channels * 2,
            'num_attention_heads': 8,
            'dropout': 0.0,
            'num_encoder_layers': num_layers,
            
            # Backbone config (adapted for LSR-MP)
            'irreps_node_embedding': f"{hidden_channels}x0e+{hidden_channels}x1e+{hidden_channels}x2e",
            'num_layers': num_layers,
            'basis_type': 'gaussiansmear',
            'number_of_basis': num_rbf,
            'num_attn_heads': 8,
            'attn_scalar_head': 8,
            'irreps_head': f"{hidden_channels//8}x0e+{hidden_channels//8}x1e+{hidden_channels//8}x2e",
            'rescale_degree': False,
            'nonlinear_message': False,
            'norm_layer': 'layer_norm_sh',
            'alpha_drop': 0.1,
            'proj_drop': 0.0,
            'out_drop': 0.0,
            'drop_path_rate': 0.1,
            'tp_type': 'dot_alpha_small',
            'attn_type': 'all-order',
            'edge_embedtype': 'default',
            'attn_biastype': 'share',
            'ffn_type': 's2',
            'add_rope': False,
            'time_embed': False,
            'sparse_attn': False,
            'dynamic_sparse_attn_threthod': 1000,
            'force_head': 'direct'
        }
        
        # Update with any user-provided kwargs
        e2former_config.update(kwargs)
        
        # Initialize E2Former backbone
        self.backbone = E2FormerBackbone(**e2former_config)
        self.data_mapper = E2FormerDataMapper()
        
        # Initialize LSR-MP-style output modules
        # Energy prediction using LSR-MP OutputNet (scalar features only)
        self.out_energy = OutputNet(hidden_channels, act='silu', dipole=False, 
                                  mean=mean, std=std, atomref=atom_ref, scale=None)
        
        # Force prediction using eSCN force block (vec features)
        # This provides vec features that are used as force predictions
        self.force_head = eSCN_ForceBlockV2(self.backbone, num_layers=4)
        
        # Normalization layers for output
        self.out_norm_scalar = nn.LayerNorm(hidden_channels)
        self.out_norm_vec = lambda x: vec_layernorm(x, max_min_norm)
        
    @conditional_grad(torch.enable_grad())
    def forward(self, data, *args, **kwargs):
        """
        Forward pass through E2Former model using LSR-MP-style output.
        
        Args:
            data: LSR-MP data object with pos, atomic_numbers, batch, labels, etc.
            
        Returns:
            Dict with energy and forces (if regress_forces=True):
                - energy: [batch_size] energy predictions
                - forces: [num_nodes, 3] force predictions from vec features
        """
        if self.regress_forces:
            data.pos.requires_grad_(True)
        
        # Remove self loops and extract basic info
        data.edge_index = remove_self_loops(data.edge_index)[0]
        z = data.atomic_numbers.long()
        pos = data.pos
        batch = data.batch
        
        if z.dim() == 2:  # if z of shape num_atoms x 1
            z = z.squeeze()  # squeeze to num_atoms
            
        # Convert LSR-MP format to E2Former format
        e2former_data = self.data_mapper.map_lsrmp_to_e2former(pos, z, batch)
        
        # Forward through E2Former backbone 
        results = self.backbone(e2former_data)
        
        # Get node features from E2Former
        node_scalar = results['node_features']  # [num_nodes, irreps_dim] - scalar features
        node_vec = results['node_vec_features']  # [num_nodes, 3, vec_dim] - vector features
        
        # Get forces directly from E2Former force block if needed
        if self.regress_forces:
            force_results = self.force_head.forward(e2former_data, results)
            forces = force_results['forces']  # [num_nodes, 3] - direct force output
        else:
            forces = None
        
        # Apply normalization
        node_scalar = self.out_norm_scalar(node_scalar)
        node_vec = self.out_norm_vec(node_vec)
        
        # Energy prediction using LSR-MP OutputNet with existing data
        energy = self.out_energy(node_scalar, node_vec, data)
        
        # Return results in LSR-MP format
        if self.regress_forces:
            return {"energy": energy, "forces": forces}
        else:
            return {'energy': energy}


