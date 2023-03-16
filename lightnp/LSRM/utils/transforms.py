from torch_cluster.radius import radius_graph
from torch_scatter import scatter
import torch

def convert_to_neighbor(r=3.0):
    '''
    This function is used to convert the broadcasted graph to radius graph.
    '''
    def _convert_to_neighbor(data):
        data = data.clone()
        data.edge_index = radius_graph(data.pos, r, max_num_neighbors=32)
        return data
    return _convert_to_neighbor


def reconstruct_group_with_threshold(threshold=1000000, group_center = 'center_of_mass'):
    '''
    This function is used to reconstruct the node/group interaction with threshold.
    '''
    def _convert_to_neighbor(data):
        data = data.clone()
        pos = data.pos # (N, 3)
        if group_center == 'geometric':
            group_pos = scatter(pos, data.labels, reduce='mean', dim=0)
        elif group_center == 'center_of_mass':
            group_pos = scatter(pos * data.atomic_numbers, data.labels, reduce='sum', dim=0) / scatter(data.atomic_numbers, data.labels, reduce='sum', dim=0)
        dist = (pos[:, None, :] - group_pos[None, :, :] + 1e-6).norm(dim=-1)
        mask = dist < threshold # (N, M)
        data.interaction_graph = torch.stack(torch.where(mask), dim=0) # (2, E)
        return data
    
    return _convert_to_neighbor