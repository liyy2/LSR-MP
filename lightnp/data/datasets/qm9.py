import logging
import os
import re
# import shutil
from torch_scatter import scatter
from tqdm import tqdm
import tarfile
import tempfile
from urllib import request as request

import torch
import torch
import numpy as np
from ase.io.extxyz import read_xyz
from ase.units import Debye, Bohr, Hartree, eV
from torch.utils.data import DataLoader
# 1 hartree to eV = 27.2114 
import lightnp as ltnp
from lightnp.data.atoms_dataset import AtomsData
from lightnp.data.statistic import StatisticsAccumulator
from lightnp.utils import folder_IFnotexist_create

## 1hartree = 27.2107 eV = 627.503 kcal/mol

__all__ = ["QM9_Dataset","get_statistics","get_atomref"]


class QM9_Dataset(AtomsData):
    """QM9 benchmark database for organic molecules.

    The QM9 database contains small organic molecules with up to nine non-hydrogen atoms
    from including C, O, N, F. This class adds convenient functions to download QM9 from
    figshare and load the data into pytorch.

    Args:
        dbpath (str): path to directory containing database.
        download (bool, optional): enable downloading if database does not exists.
        subset (list, optional): Deprecated! Do not use! Subsets are created with
            AtomsDataSubset class.
        propertity_load (list, optional): reduced set of properties to be loaded
        collect_triples (bool, optional): Set to True if angular features are needed.
        remove_uncharacterized (bool, optional): remove uncharacterized molecules.
        environment_provider (ltnp.environment.BaseEnvironmentProvider): define how
            neighborhood is calculated
            (default=ltnp.environment.SimpleEnvironmentProvider).

    References:
        .. [#qm9_1] https://ndownloader.figshare.com/files/3195404

    """

    # properties
    A = "rotational_constant_A"
    B = "rotational_constant_B"
    C = "rotational_constant_C"
    mu = "dipole_moment"
    alpha = "isotropic_polarizability"
    homo = "homo"
    lumo = "lumo"
    gap = "gap"
    r2 = "electronic_spatial_extent"
    zpve = "zpve"
    U0 = "energy_U0"
    U = "energy_U"
    H = "enthalpy_H"
    G = "free_energy"
    Cv = "heat_capacity"
    tot_energy = "tot_energy"
    G_energy = "group_energy"
    G_rho = "group_rho"
    edge_energy = "edge_energy"
    edge_rho = "edge_rho"
    G_matrix = "coarse_graph_matrix"
    tot_rho = "total_rho"
    tot_dms = "total_dms"
    G_dms = "group_dms"
    G_atom_types = "group_atom_types"
    G_atom_coords = "group_atom_cords"

    reference = {zpve: 0, U0: 1, U: 2, H: 3, G: 4, Cv: 5}

    def __init__(
        self,
        dbpath,
        download=False,
        propertity_load=None,
        collect_triples=False,
        remove_uncharacterized=False,
        neighbor_provider=ltnp.neighbor_function.Allatoms_NeighborProvider(),
        **kwargs
    ):
        super().__init__(
            dbpath=dbpath,
            propertity_load=propertity_load,
            neighbor_provider=neighbor_provider,
            collect_triples=collect_triples
        )

        
        self.remove_uncharacterized = remove_uncharacterized

        # this part is used for data downloading .
        self.all_properties = [
            QM9_Dataset.A,
            QM9_Dataset.B,
            QM9_Dataset.C,
            QM9_Dataset.mu,
            QM9_Dataset.alpha,
            QM9_Dataset.homo,
            QM9_Dataset.lumo,
            QM9_Dataset.gap,
            QM9_Dataset.r2,
            QM9_Dataset.zpve,
            QM9_Dataset.U0,
            QM9_Dataset.U,
            QM9_Dataset.H,
            QM9_Dataset.G,
            QM9_Dataset.Cv,
            QM9_Dataset.tot_energy,
            QM9_Dataset.G_energy,
            QM9_Dataset.G_rho,
            QM9_Dataset.edge_energy,
            QM9_Dataset.edge_rho,
            QM9_Dataset.G_matrix
        ]
                
        # this part is used for data downloading .
        self.all_# this part is used for data downloading .
        self.all_units = [
            1.0,
            1.0,
            1.0,
            Debye,
            Bohr ** 3,
            Hartree,
            Hartree,
            Hartree,
            Bohr ** 2,
            Hartree,
            Hartree,
            Hartree,
            Hartree,
            Hartree,
            1.0,
            # Hartree,# tot_energy
            1.0,
            1.0, # tot dms
            Hartree,
            1.0, # group_rho
            1.0,
            1.0, # group atom_types
            1.0,
            # Hartree,
            # 1.0, 
            1.0,
        ]

                
        self.divide_by_atoms = {
            QM9_Dataset.mu: True,
            QM9_Dataset.alpha: True,
            QM9_Dataset.homo: False,
            QM9_Dataset.lumo: False,
            QM9_Dataset.gap: False,
            QM9_Dataset.r2: True,
            QM9_Dataset.zpve: True,
            QM9_Dataset.U0: True,
            QM9_Dataset.U: True,
            QM9_Dataset.H: True,
            QM9_Dataset.G: True,
            QM9_Dataset.Cv: True,
            # ANI1.energy: True,
            # MD17.energy: True,
            # MaterialsProject.EformationPerAtom: False,
            # MaterialsProject.EPerAtom: False,
            # MaterialsProject.BandGap: False,
            # MaterialsProject.TotalMagnetization: True,
            # OrganicMaterialsDatabase.BandGap: False,
        }

        

        self.atomref = self.get_atomref(propertity_load)


        
        if download and not os.path.exists(self.dbpath):
            self.download()
    
    
    def download(self,):
        folder_IFnotexist_create(self.dbpath)
        self._download()
        
    def _download(self):
        if self.remove_uncharacterized:
            evilmols = self._load_evilmols()
        else:
            evilmols = None

        self._load_data(evilmols)

        atref, labels = self._load_atomrefs()
        self.set_metadata({"atomrefs": atref.tolist(), "atref_labels": labels})

    def _load_atomrefs(self):
        logging.info("Downloading GDB-9 atom references...")
        at_url = "https://ndownloader.figshare.com/files/3195395"
        tmpdir = tempfile.mkdtemp("gdb9")
        tmp_path = os.path.join(tmpdir, "atomrefs.txt")

        request.urlretrieve(at_url, tmp_path)
        logging.info("Done.")

        atref = np.zeros((100, 6))
        labels = [QM9_Dataset_Dataset.zpve, QM9_Dataset_Dataset.U0, QM9_Dataset_Dataset.U, QM9_Dataset_Dataset.H, QM9_Dataset_Dataset.G, QM9_Dataset_Dataset.Cv]
        with open(tmp_path) as f:
            lines = f.readlines()
            for z, l in zip([1, 6, 7, 8, 9], lines[5:10]):
                atref[z, 0] = float(l.split()[1])
                atref[z, 1] = float(l.split()[2]) * Hartree / eV
                atref[z, 2] = float(l.split()[3]) * Hartree / eV
                atref[z, 3] = float(l.split()[4]) * Hartree / eV
                atref[z, 4] = float(l.split()[5]) * Hartree / eV
                atref[z, 5] = float(l.split()[6])
        return atref, labels

    def _load_evilmols(self):
        logging.info("Downloading list of uncharacterized molecules...")
        at_url = "https://ndownloader.figshare.com/files/3195404"
        tmpdir = tempfile.mkdtemp("gdb9")
        tmp_path = os.path.join(tmpdir, "uncharacterized.txt")

        request.urlretrieve(at_url, tmp_path)
        logging.info("Done.")

        evilmols = []
        with open(tmp_path) as f:
            lines = f.readlines()
            for line in lines[9:-1]:
                evilmols.append(int(line.split()[0]))
        return np.array(evilmols)

    def _load_data(self, evilmols=None):
        logging.info("Downloading GDB-9 data...")
        tmpdir = tempfile.mkdtemp("gdb9")
        tar_path = os.path.join(tmpdir, "gdb9.tar.gz")
        raw_path = os.path.join(tmpdir, "gdb9_xyz")
        url = "https://ndownloader.figshare.com/files/3195389"

        request.urlretrieve(url, tar_path)
        logging.info("Done.")

        logging.info("Extracting files...")
        tar = tarfile.open(tar_path)
        tar.extractall(raw_path)
        tar.close()
        logging.info("Done.")

        logging.info("Parse xyz files...")
        ordered_files = sorted(
            os.listdir(raw_path), key=lambda x: (int(re.sub("\D", "", x)), x)
        )

        all_atoms = []
        all_properties = []

        irange = np.arange(len(ordered_files), dtype=np.int)
        if evilmols is not None:
            irange = np.setdiff1d(irange, evilmols - 1)

        for i in irange:
            xyzfile = os.path.join(raw_path, ordered_files[i])

            if (i + 1) % 10000 == 0:
                logging.info("Parsed: {:6d} / 133885".format(i + 1))
            properties = {}
            tmp = os.path.join(tmpdir, "tmp.xyz")

            with open(xyzfile, "r") as f:
                lines = f.readlines()
                l = lines[1].split()[2:]
                for pn, p in zip(self.all_properties, l):
                    properties[pn] = np.array([float(p) * self.all_units[pn]])
                with open(tmp, "wt") as fout:
                    for line in lines:
                        fout.write(line.replace("*^", "e"))

            with open(tmp, "r") as f:
                ats = list(read_xyz(f, 0))[0]
            all_atoms.append(ats)
            all_properties.append(properties)

        logging.info("Write atoms to db...")
        self.add_systems(all_atoms, all_properties)
        logging.info("Done.")

        shutil.rmtree(tmpdir)

        return True
    
    
def get_statistics(dataset,loader, prop_name, prop_divide_by_atoms, atomref=None):
    """
    Compute mean and variance of a property. Uses the incremental Welford
    algorithm implemented in StatisticsAccumulator

    Args:
        prop_name (str):  gather/compile statistic for given property. 
        prop_divide_by_atoms (True or False): divide by the number of atoms if True.
        prop_atomref: Use reference values for single atoms.
                                        e.g. COOH   Energy(COOH)-reference(C)-2*reference(O)-reference(H)

    Returns:
        mean: Mean value
        stddev: Standard deviation

    """
    
    # divide_by_atoms = dataset.divide_by_atoms
    # atomref = dataset.atomref
    # print('When statistic, use_atomref dict: ',use_atomref)
    
    ## just for statistic
    if loader is None:
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    statistics = StatisticsAccumulator(batch=True)
    if atomref is not None and isinstance(atomref,np.ndarray):
        atomref = torch.from_numpy(atomref).float()
        
    with torch.no_grad():
        print("statistics will be calculated...")
        for data in tqdm(loader, total = len(loader)):
            property_value = data[prop_name]
            
            # use atom as reference value!
            if atomref is not None:
                z = data["atomic_numbers"]
                p0 = atomref[z].reshape(-1,1)
                p0 = scatter(p0, data['batch'], dim=0).reshape(-1,1)
                property_value -= p0
            if prop_divide_by_atoms:
                try:
                    mask = torch.sum(data["_atom_mask"], dim=1, keepdim=True).view(
                        [-1, 1] + [1] * (property_value.dim() - 2)
                    )
                    property_value /= mask
                except:
                    counter = torch.ones_like(data['batch'])
                    mask = scatter(counter, data['batch'], dim=0)
                    # if mask.dim() == 1:
                    #     mask = mask.unsqueeze(1)
                    property_value = property_value.reshape(-1)/mask.reshape(-1)
                    # property_value = property_value.unsqueeze(1)

                
                
            statistics.add_sample(property_value)


    return statistics.get_mean(), statistics.get_stddev()



def get_atomref(dataset, prop_name, data_len = None, atomic_number_max = 60):
    '''
    prop_name: "energy"
    '''
    data_len = len(dataset) if data_len is None else data_len
    r = torch.zeros(data_len, atomic_number_max+1)
    energy = torch.zeros(data_len)
    for i in range(data_len):
        data = dataset[i]
        counter = scatter(torch.ones(data.atomic_numbers.shape[0]),data.atomic_numbers.reshape(-1),dim = 0)
        energy[i] = data[prop_name].item()
        r[i,:len(counter)] = counter

    mask = torch.sum(r, dim = 0) > 0
    r = r[:, mask]
    legal_atomref = torch.linalg.lstsq(r, energy).solution
    print('in this dataset, available legal atomic numer is {}. its atomref is {}'.format(torch.where(mask)[0],legal_atomref))
    atomref = torch.zeros(atomic_number_max+1)
    atomref[mask] = legal_atomref
    return atomref