import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.utils import remove_self_loops
from .long_short_interact_modules import LongShortIneractModel_dis_direct_vector2_drop
from .utils import get_distance
from .torchmdnet.models.torchmd_norm import  EquivariantMultiHeadAttention
from .torchmdnet.models.utils import ExpNormalSmearing,GaussianSmearing,NeighborEmbedding, vec_layernorm, max_min_norm, norm
from .output_net import OutputNet
from ..utils import conditional_grad


class Node_Edge_Fea_Init(nn.Module):
    def __init__(self,
                 max_z = 100,
                 rbf_type="expnorm",
                 num_rbf = 50,
                 trainable_rbf = True,
                 hidden_channels = 128,
                 cutoff_lower = 0,
                 cutoff_upper = 5,
                 neighbor_embedding = True):
        super().__init__()
        self.embedding = nn.Embedding(max_z, hidden_channels)
        
        if rbf_type == "expnorm":
            rbf = ExpNormalSmearing
        elif rbf_type == "":
            rbf = GaussianSmearing
        else:
            assert(False)
        self.distance_encoder=rbf(cutoff_lower=cutoff_lower, cutoff_upper=cutoff_upper, num_rbf=num_rbf, trainable=trainable_rbf)
        self.rbf_linear = nn.Linear(num_rbf,hidden_channels)
        if neighbor_embedding:
            self.neighbor_embedding = NeighborEmbedding(hidden_channels, num_rbf, cutoff_lower, cutoff_upper, max_z)
        else:
            self.neighbor_embedding = None
            
    def forward(self,z,pos,edge_index):
        #z means atoms-ID: H1, C6, N7, O8,
        ### this part is for node short term neighbor.
        node_embedding = self.embedding(z)
        node_vec = torch.zeros(node_embedding.size(0), 3, node_embedding.size(1), device=node_embedding.device)
        edge_index, edge_weight, edge_vec = get_distance(pos,pos,edge_index)
        edge_attr = self.distance_encoder(edge_weight)
        # mask = edge_index[0] != edge_index[1]
        # edge_vec[mask] = edge_vec[mask]  / (torch.norm(edge_vec[mask], dim=1).unsqueeze(1)+1e-5)
        edge_vec = edge_vec  / norm(edge_vec, keepdim=True) 
        if self.neighbor_embedding is not None:
            node_embedding = self.neighbor_embedding(z, node_embedding, edge_index, edge_weight, edge_attr)
        edge_attr = self.rbf_linear(edge_attr)
        return node_embedding, node_vec, edge_index, edge_weight, edge_attr, edge_vec

class Edge_Feat_Init(nn.Module):
    def __init__(self,
                rbf_type="expnorm",
                num_rbf = 50,
                trainable_rbf = True,
                hidden_channels = 128,
                cutoff_lower = 0,
                cutoff_upper = 5):
    
        super().__init__()
        if rbf_type == "expnorm":
            rbf = ExpNormalSmearing
        elif rbf_type == "":
            rbf = GaussianSmearing
        else:
            assert(False)
        self.distance_encoder=rbf(cutoff_lower=cutoff_lower, cutoff_upper=cutoff_upper, num_rbf=num_rbf, trainable=trainable_rbf)
        self.rbf_linear = nn.Linear(num_rbf,hidden_channels)

    def forward(self, pos, edge_index):
        edge_index, edge_weight, edge_vec = get_distance(pos,pos,edge_index)
        edge_attr = self.distance_encoder(edge_weight)
        edge_vec = edge_vec  / norm(edge_vec, keepdim=True)
        edge_attr = self.rbf_linear(edge_attr)
        return edge_index, edge_weight, edge_attr, edge_vec
    

class Bipartite_Edge_Feat_Init(nn.Module):
    def __init__(self,
                rbf_type="expnorm",
                num_rbf = 50,
                trainable_rbf = True,
                hidden_channels = 128,
                cutoff_lower = 0,
                cutoff_upper = 10):
    
        super().__init__()
        if rbf_type == "expnorm":
            rbf = ExpNormalSmearing
        elif rbf_type == "":
            rbf = GaussianSmearing
        else:
            assert(False)
        self.distance_encoder=rbf(cutoff_lower=cutoff_lower, cutoff_upper=cutoff_upper, num_rbf=num_rbf, trainable=trainable_rbf)
        self.rbf_linear = nn.Linear(num_rbf,hidden_channels)

    def forward(self, edge_index, node_pos, group_pos, *args, **kwargs):
        edge_vec = node_pos[edge_index[0]] - group_pos[edge_index[1]]
        edge_weight = norm(edge_vec, dim=1)
        edge_vec = edge_vec / edge_weight.unsqueeze(1)
        edge_attr = self.distance_encoder(edge_weight)
        edge_attr = self.rbf_linear(edge_attr)
        return edge_index, edge_weight, edge_attr, edge_vec   
    



class Visnorm_shared_LSRMNorm2_2branchSerial(nn.Module):
    def __init__(self,regress_forces = True,
                 hidden_channels=128,
                 num_layers=6,
                 num_rbf=50,
                 rbf_type="expnorm",
                 trainable_rbf=True,
                 neighbor_embedding=True,
                 short_cutoff_upper=10,
                 long_cutoff_upper=10,
                 mean = None,
                 std = None,
                 atom_ref = None,
                 max_z=100,
                 group_center='center_of_mass',
                 tf_writer = None,
                 **kwargs):
        super().__init__()
        # self.embedding_long = nn.Embedding(max_z, hidden_channels)
        # self.logger = {f'{i}': {'dx':[], 'vec': } for i in range(1, num_layers + 1)}
        self.hidden_channels = hidden_channels
        self.regress_forces = regress_forces
        self.num_layers = num_layers
        self.group_center = group_center
        self.tf_writer = tf_writer
        self.t = 0
        
        self.node_fea_init = Node_Edge_Fea_Init(
                                    max_z = max_z,
                                    rbf_type=rbf_type,
                                    num_rbf = num_rbf,
                                    trainable_rbf = trainable_rbf,
                                    hidden_channels = hidden_channels,
                                    cutoff_lower = 0,
                                    cutoff_upper = short_cutoff_upper,
                                    neighbor_embedding = neighbor_embedding)
        self.mlp_node_fea = nn.Linear(hidden_channels,2*hidden_channels)
        self.mlp_node_vec_fea = nn.Linear(hidden_channels,2*hidden_channels, bias = False)
        
        self.edge_fea_init = Edge_Feat_Init(rbf_type = rbf_type,
                num_rbf = num_rbf,
                trainable_rbf = trainable_rbf,
                hidden_channels = hidden_channels,
                cutoff_lower = 0,
                cutoff_upper = short_cutoff_upper)
        
        self.bipartite_edge_fea_init = Bipartite_Edge_Feat_Init(rbf_type = rbf_type,
                num_rbf = num_rbf,
                trainable_rbf = trainable_rbf,
                hidden_channels = hidden_channels,
                cutoff_lower = 0,
                cutoff_upper = long_cutoff_upper)


        self.long_cutoff_upper = long_cutoff_upper
        
        self.visnet_att0 = nn.ModuleList()
        self.longshortinteract_models = nn.ModuleList()
        for _ in range(self.num_layers):
            self.visnet_att0.append(EquivariantMultiHeadAttention(
                                        hidden_channels,
                                        distance_influence = "both",
                                        num_heads = 8,
                                        activation = "silu",
                                        attn_activation = "silu",
                                        cutoff_lower = 0,
                                        cutoff_upper = short_cutoff_upper,
                                        last_layer=False,
                                    ))

        config = kwargs["config"]
        self.long_num_layers = config["long_num_layers"]
        for i in range(self.long_num_layers):
            self.longshortinteract_models.append(LongShortIneractModel_dis_direct_vector2_drop(hidden_channels, num_gaussians=50, 
                                                                                             cutoff=self.long_cutoff_upper,norm=True, max_group_num = 3,act = "silu",num_heads=8,
                                                                                             p = config["dropout"]))


            # self.group_embed_abn.append(All_Batch_Norm(normalized_shape = hidden_channels,L2_norm_dim = None))
            # self.node_embed_abn.append(All_Batch_Norm(normalized_shape = hidden_channels,L2_norm_dim = None))
            # self.node_vec_embed_abn.append(All_Batch_Norm(normalized_shape = hidden_channels,L2_norm_dim = 1))
            # self.edge_embed_abn.append(All_Batch_Norm(normalized_shape = hidden_channels,L2_norm_dim = None))
        self.out_norm1 = nn.LayerNorm(hidden_channels)
        self.out_norm2 = nn.LayerNorm(hidden_channels)
        self.out_energy = OutputNet(hidden_channels*2, act = 'silu', dipole = False, mean = mean, std = std, atomref = atom_ref, scale = None)
        
    @conditional_grad(torch.enable_grad())
    def forward(self,
                data,
                *args,
                **kwargs
                ):
        
        '''
        data.grouping_graph # Grouping graph (intra group complete graph; inter group disconnected)
        data.interaction_graph #Bipartite graph, [0] node, [1] group
        '''
        # if self.debug:
        # torch.autograd.set_detect_anomaly(True)
        if self.regress_forces:
            data.pos.requires_grad_(True)
        
        data.edge_index =  remove_self_loops(data.edge_index)[0]
        # data.grouping_graph = remove_self_loops(data.grouping_graph)[0]
        z = data.atomic_numbers.long()
        pos = data.pos
        labels = data.labels
        atomic_numbers = data.atomic_numbers

        # node related feature, node-node distance
        if z.dim() == 2:  # if z of shape num_atoms x 1
            z = z.squeeze()  # squeeze to num_atoms
        device = pos.device
        #group related feature
        if self.group_center == 'geometric':
            group_pos = scatter(pos, data.labels, reduce='mean', dim=0)
        elif self.group_center == 'center_of_mass':
            group_pos = scatter(pos * atomic_numbers, labels, reduce='sum', dim=0) /scatter(atomic_numbers,labels, reduce='sum', dim=0)
        else:
            assert(False)
        node_id,group_id = data.interaction_graph[0],data.interaction_graph[1]
        node_group_dis = torch.sqrt(torch.sum((pos[node_id]-group_pos[group_id])**2,dim = 1))
        data.interaction_graph = data.interaction_graph[:,node_group_dis<=self.long_cutoff_upper]
        group_embedding = None
        group_vec = torch.zeros((group_pos.shape[0],3,self.hidden_channels),device = device)
        node_embedding, node_vec, edge_index_short, edge_weight_short, edge_attr_short, edge_vec_short = self.node_fea_init(z,pos,data.edge_index)
        edge_index_bipartite, edge_weight_bipartite, edge_attr_bipartite, edge_vec_bipartite = self.bipartite_edge_fea_init(data.interaction_graph, pos, group_pos)
        node_embedding_short,node_embedding_long= torch.split(self.mlp_node_fea(node_embedding), self.hidden_channels, dim=-1)
        node_vec_short, node_vec_long= torch.split(self.mlp_node_vec_fea(node_vec), self.hidden_channels, dim=-1)
        for idx  in range(self.num_layers):
            # short term local neighbor
            delta_node_embedding_short, delta_node_vec_short, dedge_attr_short = self.visnet_att0[idx](node_embedding_short, node_vec_short, edge_index_short, edge_weight_short, edge_attr_short, edge_vec_short)
            node_embedding_short = node_embedding_short + delta_node_embedding_short
            node_vec_short = node_vec_short + delta_node_vec_short
            edge_attr_short = edge_attr_short + dedge_attr_short
        if self.long_num_layers!=0:
            node_embedding_long = node_embedding_short
            node_vec_long = node_vec_short
        else:
            node_embedding_long = node_embedding_long*0
            node_vec_long = node_vec_long*0
        for idx  in range(self.long_num_layers):        
            group_embedding = scatter(node_embedding_long, labels, dim=0, reduce = 'mean')
            # group_embedding = self.group_embed_ln[idx](group_embedding)
            group_vec = scatter(node_vec_long, labels, dim=0, reduce = 'mean')
            # Vector Scalar Interaction
            # node group interaction 
            # node_embedding0, group_embedding
            delta_node_embedding_long, delta_node_vec_long = self.longshortinteract_models[idx](edge_index = edge_index_bipartite, 
                                                                                     node_embedding = node_embedding_long, node_pos = pos,node_vec = node_vec_long,
                                                                                     group_embedding=group_embedding, group_pos = group_pos,
                                                                                     group_vec = group_vec, edge_attr = edge_attr_bipartite,
                                                                                     edge_weight = edge_weight_bipartite, edge_vec = edge_vec_bipartite)
            
            node_embedding_long = node_embedding_long + delta_node_embedding_long
            node_vec_long = node_vec_long + delta_node_vec_long
            
        node_embedding_short = self.out_norm1(node_embedding_short)
        node_vec_short = vec_layernorm(node_vec_short, max_min_norm)
        node_embedding_long = self.out_norm2(node_embedding_long)
        node_vec_long = vec_layernorm(node_vec_long, max_min_norm)
        
        node_embedding_short = torch.cat([node_embedding_short,node_embedding_long],dim = -1)
        node_vec_short = torch.cat([node_vec_short,node_vec_long],dim = -1)

        # node_embedding_short = self.out_norm(node_embedding_short)
        energy = self.out_energy(node_embedding_short,node_vec_short,data)
            
        if self.regress_forces:
            forces = -1 * (
                torch.autograd.grad(
                    energy,
                    data.pos,
                    grad_outputs=torch.ones_like(energy),
                    create_graph=True if self.training else False,
                    retain_graph=True if self.training else False
                )[0]
            )
            if torch.any(torch.isnan(energy)):
                assert(False)
            if torch.any(torch.isnan(forces)):
                assert(False)
            return {"energy": energy, "forces": forces}
        else:
            return {'energy': energy}
