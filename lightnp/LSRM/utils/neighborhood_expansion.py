import numpy as np
import torch 
from torch_geometric.utils import coalesce, to_undirected



def get_most_frequent_val_frequency(labels):
    r'''
    Get the frequency of the most frequent val in a given numpy array
    @params labels: a 1d numpy array
    '''

    return np.bincount(labels).max()


def build_group_matrix(labels):
    r'''
    Given a labels, return the Group matrix of size num_groups(labels) \times (max_frequency of each group)
    Each row is a group, each column is a node, indicating the existence of the node in the group    
    @param labels: The label tensor
    Return: a num_groups(labels) \times max_frequency
    '''
    num_labels = torch.unique(labels).shape[0]
    max_len = get_most_frequent_val_frequency(labels.numpy())
    group_matrix = torch.zeros(num_labels, max_len).fill_(-1)
    for i in range(num_labels):
        curr = torch.where( labels == i )[0]
        fill_length = len(curr)
        group_matrix[i, :fill_length] = curr
    return group_matrix.long()




def add_edges_based_on_interaction_matrix(g, labels, interaction_matrix, undirected = False):
    r'''
    Add edges based on Node-group interaction matrix. If a node interacts with a group, then add directional edges from the central node to all the nodes within group
    @param g: The original graph
    @param labels: The results of the clustering label
    Return: Edge list of the new graph
    '''
    idx = torch.where(interaction_matrix)
    node = idx[0]
    groups = idx[1] 
    group_matrix = build_group_matrix(labels)
    source = group_matrix[groups]
    target = node.unsqueeze(1).expand_as(source)
    added_edges = torch.stack([source, target], dim = 0).reshape(2, -1)
    added_edges = added_edges[:, added_edges[0] >= 0]
    curr_edges = g.edge_index
    edges = coalesce(torch.cat([curr_edges, added_edges], dim = -1))
    if undirected:
        edges = to_undirected(edges)
    return edges
    
def neighborhood_expansion(g, labels):
    r'''
    Perform Neighborhood Expansion on the graph:
    If one of the neighbor falls into other long term group, automatically 
    expand the short term neighbors to include that Long term group
    @param g: The graph to be expanded
    Return: Expanded Edge List and the LONG-TERM (NOT SHORT TERM!!!) node group interaction matrix
    '''
    assert not g.is_directed(), " Error! Radius graph G should be an undirected Graph"
    edge_index_labeled_group = labels[g.edge_index] # Get the labels of (s,t) -> (s_label, t_label)
    num_nodes = g.num_nodes
    num_labels = g.num_labels
    filters = edge_index_labeled_group[0] != edge_index_labeled_group[1] # s_label != t_label means connections are in different groups
    edges_crossing_group = g.edge_index[:, filters]
    edges_crossing_group_label = labels[edges_crossing_group]
    short_term_interaction_matrix = torch.zeros(num_nodes, num_labels)
    # source -> target_label
    short_term_interaction_matrix[edges_crossing_group[0], edges_crossing_group_label[1]] = 1
    new_edges =  add_edges_based_on_interaction_matrix(g, labels, short_term_interaction_matrix)
    long_term_interaction_matrix = 1 - short_term_interaction_matrix
    # Remove self interaction
    long_term_interaction_matrix[torch.arange(num_nodes), labels[torch.arange(num_nodes)]] = 0
    return new_edges, long_term_interaction_matrix

def adj2edge_index(adj):
    r'''
    Convert the adjacency matrix to edge index
    '''
    edge_index = torch.where(adj)
    return torch.stack(edge_index)

def build_neighborhood_n_interaction(g):
    r'''
    Build the interation 
    '''
    labels = g.labels
    new_edges, interaction_matrix = neighborhood_expansion(g, labels)
    interaction_edge_idx = adj2edge_index(interaction_matrix)
    g.edge_index = new_edges
    g.interaction_graph = interaction_edge_idx
    return g





    