from calendar import c
from graph import rdkit_grouping
from schnetpack.datasets import MD17
import os
import numpy as np
# import hashlib
from collections import defaultdict
from tqdm import trange
import torch

class Atom:
    def __init__(self, x) -> None:
        self.x = x
    
    def __lt__(self, other):
        idx = {6: 0, 1: 1, 8: 2, 7: 3, 16: 4, 15: 5, 9: 6, 17: 7, 14: 8, 18: 9,}
        idx1 = idx[self.x]
        idx2 = idx[other.x]
        # if self.x == 8 and other.x == 1:
        #     return True
        # elif  self.x == 1 and other.x == 8:
        #     return False
        return idx1 < idx2
        
def count_atom_types(atomic_numbers):
    atom_types = defaultdict(int)
    for atom in atomic_numbers:
        atom_types[atom] += 1
    return atom_types

def dict2str(d):
    dict = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}
    return ''.join(['{}{}'.format(dict[int(k)], v if v >=2 else '') for k, v in sorted(d.items(), key = lambda x : Atom(x[0]))])


def grouping2label(groups) -> torch.Tensor:
    '''
    Take the groups and implement the label associated with the grouping
    '''
    num_atoms = sum([len(group) for group in groups])
    label = torch.zeros(num_atoms).long()
    for i in range(len(groups)):
        label[groups[i]] = i    
    return label

def rdkit_label_builder(g, min_group_size, charge = 0) -> None:
    '''
    Given a graph g, build the label for the associated graph using rdkit fragmentation method
    '''
    grouping = rdkit_grouping(g.atomic_numbers.squeeze().numpy(), g.pos.numpy(), min_group_size = min_group_size, charge=charge)[0]
    g.labels = grouping2label(grouping)
    g.num_labels = len(grouping)




if __name__ == '__main__':
    ################### TESTING #################
    datapath = "/home/zhangjia/v-yunyangli/dataset"
    dataset = MD17(os.path.join(datapath, f"md17_aspirin.db"),
                    molecule = 'aspirin', load_only =["energy","forces"])

    output = defaultdict(int)
    # problem_idx = [1000]
    for i in trange(0, len(dataset)):
        atom_dict = {'C':0, 'O': 0, 'H': 0}
        example = dataset[i]
        grouping = rdkit_grouping(example['_atomic_numbers'].numpy(), example['_positions'].numpy(), min_group_size=4)[0]
        labels = grouping2label(grouping)
        # print(labels)
        atom_ids = [example['_atomic_numbers'].numpy()[np.array(group)] for group in grouping]
        # print(atom_ids)
        atom_types = [dict2str(count_atom_types(atoms)) for atoms in atom_ids]
        # grouping_size = [len(group) for group in grouping]
        # grouping_size.sort()
        
        hash_str = ';'.join(atom_types)
        output[hash_str] += 1
        # if hash_str == 'C6H2O;C2H3O;CHO2;H;H':
        #     problem_idx.append(i)

    print(output)

# print(problem_idx)
# for i in problem_idx:
#     dict = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}
#     for number, pos in zip(dataset[i]['_atomic_numbers'].numpy(), dataset[i]['_positions'].numpy()):
#         print(f'{dict[number]}    {pos[0]:9.5f}    {pos[1]:9.5f}    {pos[2]:9.5f}')
#     print('\n')



# print(output)






    