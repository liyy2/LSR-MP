import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from ..utils import build_neighborhood_n_interaction, build_label, build_grouping_graph, filter_padding_edges
from torch_geometric.transforms.radius_graph import RadiusGraph
from multiprocessing import Pool
import multiprocessing as mp
import os

class PyGWrapper(Dataset):
    def __init__(self, wrapped_dataset, otf_graph = False, r = 3):
        self.wrapped_dataset = wrapped_dataset
        self.otf_graph = otf_graph
        self.r = r
        self.memo = {}
        # self.process_all_data()
    
    def __len__(self):
        return len(self.wrapped_dataset)
    
    def neighbor_list_2_edge_index(self, neighbor_list):
        nb_atoms, nb_neighbors = neighbor_list.shape
        idx = torch.arange(nb_atoms)
        repeated_idx = idx.repeat(nb_neighbors, 1).transpose(1, 0)
        edge_idx = torch.cat([repeated_idx, neighbor_list]).reshape(2, -1)
        return edge_idx

    def process_data(self, index, memo):
        item = self.wrapped_dataset[index]
        data = Data()
        data.atomic_numbers = item['_atomic_numbers'].reshape(-1, 1)
        data.energy = item[self.wrapped_dataset.energy].unsqueeze(1)
        data.forces = item[self.wrapped_dataset.forces]
        data.num_nodes = data.atomic_numbers.shape[0]
        data.pos = item['_positions']
        if self.otf_graph:
            neighbor_finder = RadiusGraph(r = self.r)
            data = neighbor_finder(data)
        else:
            data.edge_index = self.neighbor_list_2_edge_index(item['_neighbors'])
            filter_padding_edges(data)
        build_label(data)
        build_grouping_graph(data)
        build_neighborhood_n_interaction(data)
        memo[index] = data
        return data

    def process_all_data(self):
        manager = mp.Manager()
        memo = manager.dict()
        pool = mp.Pool(os.cpu_count())
        for i in range(len(self.wrapped_dataset)):
            pool.apply_async(self.process_data, args=(self, i, memo))
        pool.close()
        pool.join()
        self.memo = memo


    def __getitem__(self, index):
        if index in self.memo:
            return self.memo[index]
        self.memo[index] = self.process_data(index, self.memo)
        return self.memo[index]
        # data = self.process_data(index)
        # return data