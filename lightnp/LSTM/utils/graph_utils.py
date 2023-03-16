from tqdm import tqdm
from torch_scatter import scatter
def filter_padding_edges(g):
    r'''
    Filter out padding edges
    @param g: a PyG graph'''

    g.edge_index = g.edge_index[:, g.edge_index[1] >= 0] # filter out padding edges

import logging
def get_max_dis(loader, group_center = 'center_of_mass'):
    """
    Get the maximum distance between two atoms in the loader
    """
    logging.info("get max distance")
    max_dis_short = 0
    min_dis_short = 100
    max_dis_long = 0
    min_dis_long = 100

    for data in tqdm(loader, total = len(loader)):
        pos = data.pos
        edge_index = data.edge_index
        nodes = data.interaction_graph[0]
        groups = data.interaction_graph[1]

        
        if group_center == 'geometric':
            group_center_pos = scatter(pos, data.labels, reduce='mean', dim=0)
        elif group_center == 'center_of_mass':
            group_center_pos = scatter(pos * data.atomic_numbers, 
            data.labels, reduce='sum', dim=0
            ) / scatter(data.atomic_numbers, 
            data.labels, reduce='sum', dim=0)
        else:
            raise NotImplementedError
        short_edge_weight = (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)
        long_edge_weight = (pos[nodes] - group_center_pos[groups]).norm(dim=-1)
        curr_max_dis_short = short_edge_weight.max().item()
        curr_min_dis_short = short_edge_weight.min().item()
        curr_max_dis_long = long_edge_weight.max().item()
        curr_min_dis_long = long_edge_weight.min().item()
        if curr_max_dis_short > max_dis_short:
            max_dis_short = curr_max_dis_short
        if curr_min_dis_short < min_dis_short:
            min_dis_short = curr_min_dis_short
        if curr_max_dis_long > max_dis_long:
            max_dis_long = curr_max_dis_long
        if curr_min_dis_long < min_dis_long:
            min_dis_long = curr_min_dis_long
    return max_dis_short, min_dis_short, max_dis_long, min_dis_long