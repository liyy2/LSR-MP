import sys
import time
# from cmath import inf
from distutils.log import info
from functools import cmp_to_key
# from tokenize import Double
from pyscf import gto, scf
import numpy as np
import networkx as nx
from lightnp.utils import MolGraph 
# from cudft.cuda_factory import CUDAFactory


def process_mol(xyz_file):
    atom_types = []
    atom_cords = []
    with open(xyz_file, 'r') as f:
        for line in f:
            line = line.split()
            atom_types.append(line[0])
            atom_cords.append([float(i) for i in line[1:]])
    atom_cords = np.array(atom_cords, dtype=np.float64)
    atom_dist_matrix =np.sqrt(np.sum((atom_cords.reshape(-1, 1, 3) - atom_cords.reshape(1, -1, 3))**2, axis=-1))

    return atom_types, atom_cords, atom_dist_matrix


def find_edge_sup(atom_dist_matrix, groups, num_limit=2):
    '''For each atom, find the top `num_limit` nearest ignored edges'''
    natom = len(atom_dist_matrix)
    atom2group = [0]*natom

    #Build mapping from atom to groups
    for idx, g in enumerate(groups):
        for a in g:
            atom2group[a] = idx

    edge_groups = []
    sort_idx = np.argsort(atom_dist_matrix, axis=-1)
    for i in range(natom):
        for j in sort_idx[i,1:num_limit+1]:
            if j>i and atom2group[i] != atom2group[j]: # i and j are not in the same groups
                edge_groups.append([i,j])
    return edge_groups

def count_overlap(natom, groups):
    '''Count how many times each atom is used'''
    ol_cnt = [0] * natom
    for g in groups:
        for a in g:
            ol_cnt[a] += 1
    return ol_cnt

def build_graph_with_distance(atom_dist_matrix, distance_cutoff=2.0):
    G = nx.Graph()
    natom = len(atom_dist_matrix)
    sorted_idx = np.argsort(atom_dist_matrix)
    for i in range(natom):
        for j in sorted_idx[i, 1:]:
            if atom_dist_matrix[i, j] <= distance_cutoff:
                G.add_edge(i, j)
            else:
                break
    assert(nx.is_connected(G))
    return G

def build_graph_from_xyz(xyz_filename):
    mg = MolGraph()
    mg.read_xyz(xyz_filename)
    G = mg.to_networkx_graph(mg)
    return G

def rdmol_to_nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                #    is_aromatic=atom.GetIsAromatic(),
                   atom_symbol=atom.GetSymbol())
        
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                #    bond_type=bond.GetBondType(),
                )
        
    return G

def build_graph(atom_types, atom_cords, method="molecule", distance_cutoff=2.0):
    """
    Building graph from atom types and atom coordinates.

    Args:
        atom_types (list): list of atom types.
        atom_cords (list): list of atom coordinates.
        method (str): `molecule` or `distance`. For the molecule method, the graph will be built 
                      according to real chemical bond, while for the distance method, the graph 
                      will be build according to the `distance_cutoff`.
        distance_cutoff (float): the distance cutoff of edges for the `distance` method. 
    """
    G = nx.Graph()
    if method == "molecule":
        mg = MolGraph(atom_types, atom_cords)
        G = mg.to_networkx_graph()
    elif method == "distance":
        atom_types, atom_cords = np.array(atom_types), np.array(atom_cords)
        atom_dist_matrix =np.sqrt(np.sum((atom_cords.reshape(-1, 1, 3) - atom_cords.reshape(1, -1, 3))**2, axis=-1))
        G = build_graph_with_distance(atom_dist_matrix, distance_cutoff)
    else:
        raise NotImplementedError

    return G

def BFS(G, src, num_limit=None, mask={}):
    '''Conduct BFS on `G` from `src` node. Up to `num_limit` nodes are visited.'''
    if num_limit is None or num_limit <= 0:
        num_limit = len(G.nodes)
    # flags = {idx: 1 for idx in range(len(mask)) if mask[idx]}
    # flags[src] = 1
    if mask is None:
        mask = {}

    ls = [src]
    mask[src] = 1
    idx = 0
    while len(ls) < num_limit and idx < len(ls):
        for n in G.neighbors(ls[idx]):
            if mask.get(n, 0):
                continue
            ls.append(n)
            mask[n] = 1
            if len(ls) >= num_limit:
                break
        idx += 1
    
    return ls

# def sample_traj(traj_list, p=None):
#     n = len(traj_list)
#     if p is None:
#         # TODO: Find some hyper-parameters with more theoretical guarantee.
#         p = np.power([1.05]*n, range(n))
#         p /= np.sum(p)

#     idx = np.random.choice(n, p=p)
#     return idx

# def sample_node(traj, max, p=None):
#     n = min(len(traj), max)
#     if p is None:
#         # TODO: Find some hyper-parameters with more theoretical guarantee.
#         p = np.power([1.05]*n, range(n))
#         p /= np.sum(p)

#     idx = np.random.choice(n, p=p)
#     return idx

def BFS_with_mask(G, mask):
    visited = dict(mask)
    traj_list = []
    rand_list = list(G.nodes)
    np.random.shuffle(rand_list)
    for node in rand_list:
        # print(len(G.nodes()), node, len(visited))
        if visited.get(node, 0):
            continue
        traj = BFS(G, node, num_limit=len(G.nodes), mask=visited)
        traj_list.append(traj)

    return traj_list

def generate_mask(n, visit_cnt, limits):
    '''
    At most `limits` nodes among `n` nodes are masked with probability proportional to `visit_cnt`.
    '''
    mask_cnt = np.random.randint(limits)
    cnt = np.array(visit_cnt)
    cnt[cnt>300] = 300
    p = np.power(10.0, cnt)
    p /= p.sum()
    sample = np.random.choice(n, mask_cnt, p = p)
    mask = {idx:0 for idx in range(n)}
    for node in sample:
        mask[node] = 1

    return mask

def find_possible_subgraph(G, subgraph_size, graph_num_limits=100):
    '''
    Step 1. Random generate a mask for the BFS trajectory. \n
    Step 2. Perform BFS with the mask to find all trajectories greater than the `subgraph_size`. \n
    Step 3. Truncate the trajectory to construct a subgraph with specified size and insert in to `subgraph_list` if haven't seen before. \n
    '''
    # assert(nx.is_connected(G))
    node_list = list(G.nodes)
    if len(node_list) <= subgraph_size:
        return [node_list]

    subgraph_list = []
    try_cnt = 0
    mask = {idx: 0 for idx in range(len(G.nodes))}
    visited_cnt = [0] * len(G.nodes)
    while len(subgraph_list) < graph_num_limits and try_cnt < graph_num_limits * 3:
        try_cnt += 1
        traj_list = BFS_with_mask(G, mask)
        for traj in traj_list:
            if len(traj) < subgraph_size:
                continue
            traj = sorted(traj[:subgraph_size])
            if traj not in subgraph_list:
                subgraph_list.append(traj)
                for node in traj:
                    visited_cnt[node] += 1

        mask = generate_mask(len(G.nodes), visited_cnt, (len(G.nodes)-subgraph_size)//2)

    return subgraph_list

# def find_possible_subgraph(G, subgraph_size, graph_num_limits=1000):
#     '''
#     Step 1. Construct a BFS trajectory from arbitrary node. \n
#     Step 2. Flip a random bit to remove a node `n` from the trajectory. \n
#     Step 3. Complete BFS without node `n`. \n
#     Remark: The random bit should be carefully designed to obtain as diverse subgraph as possible. \n
#     '''
#     assert(nx.is_connected(G))
#     node_list = list(G.nodes)
#     if len(node_list) <= subgraph_size:
#         return [node_list]

#     # Step 1
#     bfs_traj = BFS(G, len(node_list))
#     traj_list = [bfs_traj]
#     mask_list = [[0]*len(bfs_traj)]
#     subgraph_list = [bfs_traj[:subgraph_size]]
#     try_cnt = 0
#     while len(subgraph_list) < graph_num_limits and try_cnt < graph_num_limits * 1.5:
#         try_cnt += 1

#         # Step 2
#         traj_idx = sample_traj(traj_list)
#         node_idx = sample_node(traj_list[traj_idx], max=subgraph_size)

#         # Step 3
#         new_mask = list(mask_list[traj_idx])
#         new_mask[node_idx] = 1
#         new_traj = BFS_with_mask(G, mask_list[traj_idx])
#         if len(new_traj) < subgraph_size:
#             continue
#         # Note that we haven't handle repeated traj here.
#         traj_list.append(new_traj)
#         mask_list.append(new_mask)
#         subgraph_list.append(new_traj[:subgraph_size])

#     return subgraph_list

def find_possible_fragmentation(G, num_limit, frag_count):
    pass

def compare_groups(g1, g2):
    l1, l2 = len(g1), len(g2)
    if l1 != l2:
        return l1 - l2
    if g1 == g2:
        return 0
    if g1 < g2:
        return -1
    else:
        return 1

def find_min_group_id(G, node_list, groups, atom2group):
    '''Find the smallest group associated with the points in the `node_list`'''
    gidx = -1
    gsize = 1e10
    for n in node_list:
        for nb in list(G[n]):
            if nb in node_list:
                continue
            tmp_idx = atom2group[nb]
            tmp_size = len(groups[tmp_idx])
            if tmp_size < gsize:
                gidx = tmp_idx
                gsize = tmp_size

    return gidx
        
def balanced_graph_grouping(G, num_limit=6, shuffle=True):
    '''
    Build groups according to `G`, with the sizes of groups as close to the `num_limit` as possible. \n
    When `shuffle` is `True`, a random fragmentation starting point.
    '''
    GC = G.copy()
    natom = G.number_of_nodes()
    groups = []
    atom2group = [-1]*natom
    iso_nodes = []
    while GC.number_of_nodes() > 0:
        min_degree = natom + 1
        min_node = -1

        # Search for isolated nodes and non-isolated node with minimum degree
        all_nodes = list(GC.nodes)
        if shuffle:
            np.random.shuffle(all_nodes)
        for n in all_nodes:
            if GC.degree[n] == 0: # Find an isolate node
                iso_nodes.append(n)
            elif GC.degree[n] < min_degree:
                min_degree = GC.degree[n]
                min_node = n
        # Isolated nodes are removed from the diagram and processed later
        GC.remove_nodes_from(iso_nodes)

        # The node with the lowest degree was not found
        if min_degree == natom + 1:
            break

        # Starting from the node with the smallest degree, groups are established through breadth-first search
        ls = BFS(GC, min_node, num_limit)
        for i in ls:
            atom2group[i] = len(groups)
        groups.append(ls)
        GC.remove_nodes_from(ls)

    # Uniformly handle isolated nodes
    for n in iso_nodes:
        gidx = find_min_group_id(G, [n], groups, atom2group)
        if gidx == -1:
            continue
        groups[gidx].append(n)
        atom2group[n] = gidx

    # For smaller groups, they will be merged into adjacent groups as much as possible. 
    group_len = [len(g) for g in groups]
    group_sort_idx = np.argsort(group_len)
    for idx in group_sort_idx:
        g = groups[idx]
        if len(g) > num_limit*0.6:
            continue
        gidx = find_min_group_id(G, g, groups, atom2group)
        if gidx == -1 or len(groups[gidx]) > num_limit + 2: # The smallest nearest group is already too large to be merged
            continue
        groups[gidx].extend(g)
        for n in g:
            atom2group[n] = gidx
        groups[idx] = []

    # Remove invalid groups
    groups = [g for g in groups if g]
    return groups

def cord2xyz(atom_types, atom_cords):
    xyz = ""
    for i in range(len(atom_cords)):
        xyz += f"{atom_types[i]} {' '.join([str(j) for j in atom_cords[i]])}\n"
    return xyz


def sort_groups(groups):
    for g in groups:
        g.sort()
    groups.sort(key=cmp_to_key(compare_groups))
    return groups

# ================================== Mol fragmentation based RDKit ========================================
from xyz2mol import int_atom, read_xyz_file, xyz2mol
from rdkit import Chem
from rdkit.Chem.BRICS import BRICSDecompose

def build_rd_mol(atoms, atom_coords, charge=0):
    '''By default, returns only one RDKit mol instance'''
    if isinstance(atoms[0], str):
        atoms = [int_atom(a) for a in atoms]
    return xyz2mol(atoms, atom_coords, charge)[0]

def get_rd_fragments(rd_mol, min_group_size=5):
    # Method 1
    # frag_mols = BRICSDecompose(rd_mol, minFragmentSize=min_group_size, returnMols=True)

    # Method 2
    frag_mol = Chem.FragmentOnBRICSBonds(rd_mol)
    frag_mols = Chem.GetMolFrags(frag_mol, asMols=True)

    return frag_mols


def rdkit_grouping(atoms, atom_coords, charge=0, min_group_size=6):
    coords2idx = {}
    for i, coord in enumerate(atom_coords):
        coord_str = f"x={coord[0]:.6f},y={coord[1]:.6f},z={coord[2]:.6f}"
        # print(coord_str)
        coords2idx[coord_str] = i
    rd_mol = build_rd_mol(atoms, atom_coords, charge)
    # for mol in Chem.GetMolFrags(rd_mol, asMols=True):
    #     print(Chem.rdmolfiles.MolToXYZBlock(mol))
    rd_mol_frags = get_rd_fragments(rd_mol)
    # for mol in rd_mol_frags:
    #     print(Chem.rdmolfiles.MolToXYZBlock(mol))

    atom2group = [-1]*len(atoms)
    group_break_bond = []
    groups = []
    for mol in rd_mol_frags:
        g = []
        b = []
        c = mol.GetConformer()
        for idx, a in enumerate(mol.GetAtoms()):
            pos = c.GetAtomPosition(idx)
            coord_str = f"x={pos.x:.6f},y={pos.y:.6f},z={pos.z:.6f}"
            atom_idx = coords2idx[coord_str]
            if a.GetSymbol() == "*":
                b.append(atom_idx)    
                continue
            g.append(atom_idx)
            atom2group[atom_idx] = len(groups)
        group_break_bond.append(b)
        groups.append(g)
    
    for idx, g in enumerate(groups):
        if len(g) < min_group_size:
            for bond_atom_idx in group_break_bond[idx]:
                neighbor_idx = atom2group[bond_atom_idx]
                neighbor_group = groups[neighbor_idx]
                neighbor_bond = group_break_bond[neighbor_idx]
                if neighbor_group and len(neighbor_group) < 3*min_group_size:
                    neighbor_group.extend(g)
                    for a in g:
                        atom2group[a] = neighbor_idx
                    groups[idx] = []
                    group_break_bond[neighbor_idx] = list(set(group_break_bond[idx]).union(neighbor_bond).difference(neighbor_group))
                    group_break_bond[idx] = []
                    break

    return [g for g in groups if g], [gb for gb in group_break_bond if gb]