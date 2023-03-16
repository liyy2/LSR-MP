import torch
from torch.utils.data.dataset import Subset
from torch import randperm
from torch._utils import _accumulate
from collections import defaultdict
import numpy as np

def atomic_numbers_2_formula(atom_numbers):
    values, counts = torch.unique(atom_numbers, return_counts=True)
    return ''.join([f'{v}{c}' for v, c in zip(values.tolist(), counts.tolist())])


def inductive_random_split(dataset, lengths, generator=torch.Generator().manual_seed(42)):
    '''
    This function performs inductive random split on a atomic dataset.
    @param dataset: the dataset to be split
    @param lengths: the lengths of the split, expected to sum up to the length of the dataset
    @param generator: the random generator to be used
    It will first group the dataset by the atomic formula, and then perform shuffle split on the groups. Then it will shuffle the indices of the split.    '''
    
    if sum(lengths) != len(dataset):
    # type: ignore [arg-type]
        raise ValueError ("Sum of input lengths does not equal the length of the input dataset!")
    
    formulas_to_counts = defaultdict(int)
    formula_to_indices = defaultdict(list)
    for index, mol in enumerate(dataset):
        formulas_to_counts[atomic_numbers_2_formula(mol.atomic_numbers)] += 1
        formula_to_indices[atomic_numbers_2_formula(mol.atomic_numbers)].append(index)
    
    # Shuffle the formulas
    indices = randperm(len(formulas_to_counts.keys()), generator=generator).numpy()
    all_keys = np.array(list(formulas_to_counts.keys()))
    shuffled_keys = all_keys[indices] 
    
    cum_sum_lengths = list(_accumulate(lengths)) # []
    cum_sum_keys = torch.cumsum(torch.tensor([formulas_to_counts[key] for key in shuffled_keys]), dim=0)
    
    output_indices = []
    
    for index in range(len(lengths)):
        output_indices.append([])
        mask_1 = (cum_sum_keys <= cum_sum_lengths[index])
        mask_2 = (cum_sum_keys > cum_sum_lengths[index - 1] if index > 0 else True)
        mask = mask_1 * mask_2
        output_indices[index] = np.concatenate([formula_to_indices[key] for key in shuffled_keys[mask]])
    
    # shuffle the output indices 
    # for index in range(len(lengths)):
    #     output_indices[index] = randperm(len(output_indices[index]), generator=generator).numpy()
   
        
    return [Subset(dataset, indices) for indices in output_indices]

def split_combined_dataset(dataset, lengths, train_prop, generator=torch.Generator().manual_seed(42)):
    '''
    This function split a dataset based on its sample list. analagous to stratified split   '''
    sample_list = np.array(dataset.sample_list)
    cum_sum = np.cumsum(sample_list)
    prev = 0
    out_train = []
    out_val = []
    for i in range(len(sample_list)):
        sample_list = torch.arange(prev, cum_sum[i])
        out_subset = Subset(dataset, sample_list)
        sub_train, sub_val, _ = torch.utils.data.random_split(out_subset, [int(lengths[i] * train_prop), 
                                                                           lengths[i] - int(lengths[i] * train_prop), 
                                                                           len(out_subset) - lengths[i]], 
                                                              generator = generator)
        out_train.append(sub_train)
        out_val.append(sub_val)
        prev = cum_sum[i]
        
    return torch.utils.data.ConcatDataset(out_train), torch.utils.data.ConcatDataset(out_val)