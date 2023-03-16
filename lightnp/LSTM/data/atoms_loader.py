import torch
import numpy as np
from torch_geometric.data import Data
from ase.units import Bohr

def collate_fn(unit = 1, with_force = True):
    def _collate_fn(list_of_data):
        '''
        collate function for pytorch geometric data objects
        Data object should have the following keys:
        num_nodes: number of nodes in the graph
        num_labels: number of labels in the graph
        x: Node features torch.Tensor of shape (num_nodes, num_features)
        pos: Node features torch.Tensor of shape (num_nodes, 3)
        y: output labels torch.Tensor of shape (num_targets)
        force: output labels torch.Tensor of shape (num_nodes * 3)
        edge_index: torch.LongTensor of shape (2, num_edges). This graph should be neighborhood expanded.
        grouping_graph: torch.LongTensor of shape (2, num_edges). This graph is used to group the nodes. 
            Intergroup connection are not allowed. Intragroup connection are constructed b/w every possible nodes (aka. Complete).
        interaction_graph: torch.LongTensor of shape (2, num_edges). This graph is used to construct the interaction graph of size num_nodes * num_groups.
        label: Labelling of the nodes for each group.
        '''
        atomic_numbers = torch.cat([d['atomic_numbers'].reshape(-1,1) for d in list_of_data],dim = 0)
        pos = torch.cat([d['pos'] for d in list_of_data]).float()
        if isinstance(list_of_data[0]['energy'],np.float64) or isinstance(list_of_data[0]['energy'],np.float32):
            energy = np.stack([d['energy'] for d in list_of_data]).reshape(-1,1)
            energy = torch.from_numpy(energy).float()
        else:
            energy = torch.cat([d['energy'] for d in list_of_data]).reshape(-1,1)
        energy = energy / unit
        if with_force:
            forces = torch.cat([d['forces'] for d in list_of_data])
            forces = forces /unit
        else:
            forces = None


        # concatenate graph inside data object to get a single graph (block concatenation)
        edge_index = torch.cat([d['edge_index'] for d in list_of_data], dim = -1)
        # grouping_graph = torch.cat([d['grouping_graph'] for d in list_of_data], dim = -1)
        interaction_graph = torch.cat([d['interaction_graph'] for d in list_of_data], dim = -1)
        # indices are indicator varible indicating which batch does the current edge belongs to
        edgex_index_indices = torch.cat([torch.zeros(d['edge_index'].shape[1], dtype=torch.long).fill_(i) for i, d in enumerate(list_of_data)])
        # group_graph_indices = torch.cat([torch.zeros(d['grouping_graph'].shape[1], dtype=torch.long).fill_(i) for i, d in enumerate(list_of_data)])
        interaction_graph_indices = torch.cat([torch.zeros(d['interaction_graph'].shape[1], dtype=torch.long).fill_(i) for i, d in enumerate(list_of_data)])
        # concatenate the labels
        labels = torch.cat([d['labels'] for d in list_of_data])
        num_nodes = sum([d['num_nodes'] for d in list_of_data])
        max_num_nodes = max(d['num_nodes'] for d in list_of_data)
        num_labels = sum([d['num_labels'] for d in list_of_data])
        max_num_labels = max(d['num_labels'] for d in list_of_data)

        batch = torch.cat([torch.zeros(d['num_nodes'], dtype=torch.long).fill_(i) for i, d in enumerate(list_of_data)])
        node_idx_mapping_source = torch.cat([torch.arange(d['num_nodes'], dtype=torch.long) for d in list_of_data])
        node_idx_mapping_target = torch.arange(num_nodes, dtype=torch.long)

        # remap the labels
        label_batch = torch.cat([torch.zeros(d['num_labels'], dtype=torch.long).fill_(i) for i, d in enumerate(list_of_data)])
        label_idx_remapped_source = torch.cat([torch.arange(d['num_labels'], dtype=torch.long) for d in list_of_data])
        label_idx_remapped_target = torch.arange(num_labels, dtype=torch.long)
        labels = mapping_function(labels, batch, label_idx_remapped_source, label_idx_remapped_target, label_batch, max_num_labels)
        # remap the short term graph
        edge_index = mapping_function(edge_index, edgex_index_indices, node_idx_mapping_source, node_idx_mapping_target, batch, max_num_nodes)
        # remap the grouping graph
        # grouping_graph = mapping_function(grouping_graph, group_graph_indices, node_idx_mapping_source, node_idx_mapping_target, batch, max_num_nodes)
        # remap the interaction graph
        interaction_graph_src = mapping_function(interaction_graph[0], interaction_graph_indices, node_idx_mapping_source, node_idx_mapping_target, batch, max_num_nodes)
        interaction_graph_tgt = mapping_function(interaction_graph[1], interaction_graph_indices, label_idx_remapped_source, label_idx_remapped_target, label_batch, max_num_labels)
        interaction_graph = torch.stack([interaction_graph_src, interaction_graph_tgt], dim=0)
        return Data(atomic_numbers=atomic_numbers, pos=pos, energy=energy, forces = forces, edge_index=edge_index, 
                    # grouping_graph=grouping_graph, 
                    interaction_graph=interaction_graph, 
                    labels=labels, num_nodes=num_nodes, num_labels=num_labels, batch = batch, label_batch = label_batch)
    
    return _collate_fn

def remap_values(remapping, x):
    index = torch.bucketize(x.ravel(), remapping[0])
    return remapping[1][index].reshape(x.shape)


def mapping_function(to_be_remapped, indice, remap_source, remap_target, remap_batch, max_num_nodes):
    r'''
    This function perform remapping.
    to_be_remapped: The edge index to be remapped.
    indice: The indicator variable indicating which batch the edge belongs to.
    remap_source: The domain of the mapping function
    remap_target: The range (induced value) of the mapping function
    remap_batch: The batch of the mapping function
    max_num_nodes: The maximum number of nodes in the dataset.
    ######################################################################################################################
    Below is an illustration of the mapping function.
    remap_source + remap_batch -> remap target (Predefined mapping function)
    to_be_remapped + to_be_remapped_indices -> mapped_target (Mapping function application)
    ######################################################################################################################
    '''

    hash_1 =  indice * (max_num_nodes + 1) + to_be_remapped
    hash_2 = remap_batch * (max_num_nodes + 1) + remap_source
    remapping = hash_2, remap_target
    return remap_values(remapping, hash_1)





