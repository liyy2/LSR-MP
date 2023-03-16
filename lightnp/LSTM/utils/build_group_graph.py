import torch
from torch_geometric.utils import subgraph,remove_self_loops

def label_to_graph(g, labels):
    r'''
    Convert a graph g with clustering label to a disjoint graph where the components are connected by labels
    @parms g: a PyG graph
    @parms label: a Tensor of the clustering label 
    Return: The edge list of the induced label grouping graph
    '''
    num_labels = torch.unique(labels).shape[0]
    sub_g_list = []
    for i in range(num_labels):
        curr = torch.where(labels == i)
        sub_g, _ = subgraph(curr[0], g.edge_index)
        sub_g_list.append(sub_g)
    edge_list = torch.cat(sub_g_list, dim = 1)
    return edge_list




def create_complete_graph(nodes):
    r'''
    Given a list of nodes, return the complete graph induced by those nodes
    @params curr: a numpy array of nodes
    Return: the edge index of the induced graph
    '''
    num_nodes = len(nodes)
    edges = torch.zeros(2, num_nodes, num_nodes)
    edges_src = nodes.unsqueeze(1).expand(num_nodes, num_nodes)
    edges_target = nodes.unsqueeze(0).expand(num_nodes, num_nodes)
    edges[0] = edges_src
    edges[1] = edges_target
    # ret = remove_self_loops(edges.reshape(2, -1))[0]
    # assert ret.shape[1] == (num_nodes * (num_nodes - 1)), f'Error, wrong # of graph edges. Want {num_nodes * (num_nodes - 1)}, but get {ret.shape[1]}'
    return edges.reshape(2, -1)

def label_to_complete_graph(labels):
    r'''
    Convert a graph g with clustering label to a disjoint graph where the components are connected by labels
    @parms g: a PyG graph
    @parms label: a Tensor of the clustering label 
    Return: The edge list of the induced label grouping graph
    '''
    num_labels = torch.unique(labels).shape[0]
    sub_g_list = []
    for i in range(num_labels):
        curr = torch.where(labels == i)
        sub_g  = create_complete_graph(curr[0])
        sub_g_list.append(sub_g)
    edge_list = torch.cat(sub_g_list, dim = 1)
    return edge_list

def build_grouping_graph(g):
    r'''
    Build the grouping graph of a graph g
    @params g: a PyG graph
    Return: The edge list of the grouping graph
    '''
    labels = g.labels
    g.grouping_graph = label_to_complete_graph(labels)

if __name__ == '__main__':
    print(label_to_complete_graph(labels= torch.LongTensor([1, 2 ,3 ,0 ,2 ,3, 1])))