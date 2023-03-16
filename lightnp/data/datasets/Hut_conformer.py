from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Recap,BRICS
import numpy
# from ase import Atoms

def parseRDKitMol(mol, addH, group=None):
    n_atom = mol.GetNumAtoms()
    atomic_numbers = []
    atom_positions = []
    clean_group = []
    for idx in range(n_atom):
        atm = mol.GetAtomWithIdx(idx)
        num = atm.GetAtomicNum()
        if num != 0:
            atomic_numbers.append(num)
            x,y,z = mol.GetConformer().GetAtomPosition(idx)
            atom_positions.append([x,y,z])
            if group is not None:
                clean_group.append(group[idx])
        else:
            if addH:
                if sum(atomic_numbers)%2 != 0:
                    atomic_numbers.append(1)
                    x,y,z = mol.GetConformer().GetAtomPosition(idx)
                    atom_positions.append([x,y,z])
            else:
                raise ValueError("fake atom unprocessed")
    return clean_group, numpy.array(atomic_numbers), numpy.array(atom_positions)

def convert_to_ase_atoms(mol):
        pass