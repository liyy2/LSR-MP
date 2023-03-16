"""
This module contains all functionalities required to load atomistic data,
generate batches and compute statistics. It makes use of the ASE database
for atoms [#ase2]_.

References
----------
.. [#ase2] Larsen, Mortensen, Blomqvist, Castelli, Christensen, Dułak, Friis,
   Groves, Hammer, Hargus:
   The atomic simulation environment -- a Python library for working with atoms.
   Journal of Physics: Condensed Matter, 9, 27. 2017.
"""

import logging
import os
import warnings
import bisect

import torch
import numpy as np
from ase.db import connect
from torch.utils.data import Dataset, ConcatDataset, Subset
from tqdm import tqdm
from volumentations import *

import lightnp as ltnp
from lightnp.propertity import Properties
from lightnp.neighbor_function import NeighborProvider,\
                                        Allatoms_NeighborProvider,\
                                        AseNeighborProvider,\
                                        TorchNeighborProvider
from lightnp.utils.torch_utils import torchify_dict,numpyfy_dict


logger = logging.getLogger(__name__)

__all__ = [
    "AtomsData",
    "AtomsDataSubset",
    "ConcatAtomsData",
    # "AtomsConverter",
    "get_center_of_mass",
    "get_center_of_geometry",
]


def get_center_of_mass(atoms):
    """
    Computes center of mass.

    Args:
        atoms (ase.Atoms): atoms object of molecule

    Returns:
        center of mass
    """
    masses = atoms.get_masses()
    return np.dot(masses, atoms.arrays["positions"]) / masses.sum()


def get_center_of_geometry(atoms):
    """
    Computes center of geometry.

    Args:
        atoms (ase.Atoms): atoms object of molecule

    Returns:
        center of geometry
    """
    return atoms.arrays["positions"].mean(0)


class AtomsDataError(Exception):
    pass


class AtomsData(Dataset):
    """
    PyTorch dataset for atomistic data. The raw data is stored in the specified
    ASE database. Use together with lightnp.data.AtomsLoader to feed data
    to your model.

    Args:
        dbpath (str): path to directory containing database.
        subset (list, optional): Deprecated! Do not use! Subsets are created with
            AtomsDataSubset class.
        available_properties (list, optional): complete set of physical properties
            that are contained in the database.
        propertity_load (list, optional): reduced set of properties to be loaded
        environment_provider (ltnp.environment.BaseEnvironmentProvider): define how
            neighborhood is calculated
            (default=ltnp.environment.SimpleEnvironmentProvider).
        collect_triples (bool, optional): Set to True if angular features are needed.
        centering_function (callable or None): Function for calculating center of
            molecule (center of mass/geometry/...). Center will be subtracted from
            positions.
    """

    ENCODING = "utf-8"

    def __init__(
        self,
        dbpath,
        propertity_load=None,
        neighbor_provider=Allatoms_NeighborProvider(),
        collect_triples=False,
        centering_function=get_center_of_mass,
    ):
        # checks
        if not dbpath.endswith(".db"):
            raise AtomsDataError(
                "Invalid dbpath! Please make sure to add the file extension '.db' to "
                "your dbpath."
            )

        # database
        self.dbpath = dbpath

        # check if database is deprecated:
        if self._is_deprecated():
            self._deprecation_update()

        self.propertity_load = propertity_load
        self.available_properties = self._get_available_properties()
        print("current dataset available_properties is {}".format(self.available_properties))

        self.divide_by_atoms = None
        self.pooling_mode = None
        # environment
        self.neighbor_provider = neighbor_provider
        self.collect_triples = collect_triples
        self.centering_function = centering_function


    def download(self):
        """
        Wrapper function for the download method.
        """
        if os.path.exists(self.dbpath):
            logger.info(
                "The dataset has already been downloaded and stored "
                "at {}".format(self.dbpath)
            )
        else:
            logger.info("Starting download")
            folder = os.path.dirname(os.path.abspath(self.dbpath))
            if not os.path.exists(folder):
                os.makedirs(folder)
            self._download()

    def _download(self):
        """
        To be implemented in deriving classes.
        """
        raise NotImplementedError
    
    def _is_deprecated(self):
        """
        Check if database is deprecated.

        Returns:
            (bool): True if ase db is deprecated.
        """
        # check if db exists
        if not os.path.exists(self.dbpath):
            return False

        # get properties of first atom
        with connect(self.dbpath) as conn:
            data = conn.get(1).data

        # check byte style deprecation
        if True in [pname.startswith("_dtype_") for pname in data.keys()]:
            return True
        # fallback for properties stored directly in the row
        if True in [type(val) != np.ndarray for val in data.values()]:
            return True

        return False

    def _deprecation_update(self):
        """
        Update deprecated database to a valid ase database.
        """
        warnings.warn(
            "The database is deprecated and will be updated automatically. "
            "The old database is moved to {}.deprecated!".format(self.dbpath)
        )

        # read old database
        (
            atoms_list,
            properties_list,
            key_value_pairs_list,
        ) = ltnp.utils.read_deprecated_database(self.dbpath)
        metadata = self.get_metadata()

        # move old database
        os.rename(self.dbpath, self.dbpath + ".deprecated")

        # write updated database
        self.set_metadata(metadata=metadata)
        with connect(self.dbpath) as conn:
            for atoms, properties, key_value_pairs in tqdm(
                zip(atoms_list, properties_list, key_value_pairs_list),
                "Updating new database",
                total=len(atoms_list),
            ):
                conn.write(
                    atoms,
                    data=numpyfy_dict(properties),
                    key_value_pairs=key_value_pairs,
                )
    
    
    # metadata
    def get_metadata(self, key=None):
        """
        Returns an entry from the metadata dictionary of the ASE db.

        Args:
            key: Name of metadata entry. Return full dict if `None`.

        Returns:
            value: Value of metadata entry or full metadata dict, if key is `None`.

        """
        with connect(self.dbpath) as conn:
            if key is None:
                return conn.metadata
            if key in conn.metadata.keys():
                return conn.metadata[key]
        return None

    def set_metadata(self, metadata=None, **kwargs):
        """
        Sets the metadata dictionary of the ASE db.

        Args:
            metadata (dict): dictionary of metadata for the ASE db
            kwargs: further key-value pairs for convenience
        """

        # merge all metadata
        if metadata is not None:
            kwargs.update(metadata)

        with connect(self.dbpath) as conn:
            conn.metadata = kwargs

    def update_metadata(self, data):
        with connect(self.dbpath) as conn:
            metadata = conn.metadata
        metadata.update(data)
        self.set_metadata(metadata)

    def get_atomref(self, properties):
        """
        Return multiple single atom reference values as a dictionary.

        Args:
            properties (list or str): Desired properties for which the atomrefs are
                calculated.

        Returns:
            dict: atomic references
        """
        if type(properties) is not list:
            properties = [properties]
        return {p: self._get_atomref(p) for p in properties}
    
    
    # add systems
    def add_system(self, atoms, properties=dict(), key_value_pairs=dict()):
        """
        Add atoms data to the dataset.

        Args:
            atoms (ase.Atoms): system composition and geometry
            **properties: properties as key-value pairs. Keys have to match the
                `available_properties` of the dataset.

        """
        with connect(self.dbpath) as conn:
            self._add_system(conn, atoms, properties, key_value_pairs)

    def add_systems(self, atoms_list, property_list=None, key_value_pairs_list=None):
        """
        Add atoms data to the dataset.

        Args:
            atoms_list (list of ase.Atoms): system composition and geometry
            property_list (list): Properties as list of key-value pairs in the same
                order as corresponding list of `atoms`.
                Keys have to match the `available_properties` of the dataset.

        """
        # build empty dicts if property/kv_pairs list is None
        if property_list is None:
            property_list = [dict() for _ in range(len(atoms_list))]
        if key_value_pairs_list is None:
            key_value_pairs_list = [dict() for _ in range(len(atoms_list))]

        # write systems to database
        with connect(self.dbpath) as conn:
            for at, prop, kv_pair in zip(
                atoms_list, property_list, key_value_pairs_list
            ):
                self._add_system(conn, at, prop, kv_pair)



    def __add__(self, other):
        return ConcatAtomsData([self, other])

    # private methods
    def _add_system(self, conn, atoms, properties=dict(), key_value_pairs=dict()):
        """
        Write systems to the database. Floats, ints and np.ndarrays without dimension are transformed to np.ndarrays with dimension 1.

        """
        data = {}
        # add available properties to database
        for pname in self.available_properties:
            try:
                data[pname] = properties[pname]
            except:
                raise AtomsDataError("Required property missing:" + pname)

        # transform to np.ndarray
        data = numpyfy_dict(data)

        conn.write(atoms, data=data, key_value_pairs=key_value_pairs)

    def _get_atomref(self, property):
        """
        Returns single atom reference values for specified `property`.

        Args:
            property (str): property name

        Returns:
            list: list of atomrefs
        """
        labels = self.get_metadata("atref_labels")
        if labels is None:
            return None

        col = [i for i, l in enumerate(labels) if l == property]
        assert len(col) <= 1

        if len(col) == 1:
            col = col[0]
            atomref = np.array(self.get_metadata("atomrefs"))[:, col : col + 1]
        else:
            atomref = None

        return atomref

    def _get_available_properties(self):
        """
        Get available properties from argument or database.

        Returns:
            (list): all properties of the dataset
        """
        # read database properties
        if os.path.exists(self.dbpath) and len(self) != 0:
            with connect(self.dbpath) as conn:
                atmsrw = conn.get(1)
                db_properties = list(atmsrw.data.keys())
        else:
            db_properties = None

        # use the provided list
        # if properties is not None:
        #     if db_properties is None or set(db_properties) == set(properties) or set(set(db_properties) & set(properties)) == set(properties)  :
        #         return properties

        #     # raise error if available properties do not match database
        #     raise AtomsDataError(
        #         "The available_properties {} do not match the "
        #         "properties in the database {}!".format(properties, db_properties)
        #     )

        # return database properties
        if db_properties is not None:
            return db_properties

        raise AtomsDataError(
            "Please define available_properties or set db_path to an existing database!"
        )

    
    # get atoms and properties
    def get_properties(self, idx, propertity_load=None):
        """
        Return property dictionary at given index.

        Args:
            idx (int): data index
            propertity_load (sequence or None): subset of available properties to load

        Returns:
            at (ase.Atoms): atoms object
            properties (dict): dictionary with molecular properties

        """
        # use all available properties if nothing is specified
        if propertity_load is None:
            propertity_load = self.available_properties

        # read from ase-database
        with connect(self.dbpath) as conn:
            row = conn.get(idx + 1)
        at = row.toatoms()

        # extract properties
        properties = {}
        for pname in propertity_load:
            properties[pname] = row.data[pname]

        # extract/calculate structure
        properties = _convert_atoms(
            at,
            neighbor_provider=self.neighbor_provider,
            collect_triples=self.collect_triples,
            centering_function=self.centering_function,
            output=properties,
        )

        return at, properties

    def get_atoms(self, idx):
        """
        Return atoms of provided index.

        Args:
            idx (int): atoms index

        Returns:
            ase.Atoms: atoms data

        """
        with connect(self.dbpath) as conn:
            row = conn.get(idx + 1)
        at = row.toatoms()
        return at
    
    
    def __getitem__(self, idx):
        _, properties = self.get_properties(idx, self.propertity_load)
        properties["_idx"] = np.array([idx], dtype=np.int)

        return torchify_dict(properties)
    
    
    # __functions__
    def __len__(self):
        with connect(self.dbpath) as conn:
            return conn.count()

    


class ConcatAtomsData(ConcatDataset):
    r"""
    Dataset as a concatenation of multiple atomistic datasets.
    Args:
        datasets (sequence): list of datasets to be concatenated
    """

    def __init__(self, datasets):
        # checks
        # for dataset in datasets:
        #     if not any(
        #         [
        #             isinstance(dataset, dataset_class)
        #             for dataset_class in [AtomsData, AtomsDataSubset, ConcatDataset]
        #         ]
        #     ):
        #         raise AtomsDataError(
        #             "{} is not an instance of AtomsData, AtomsDataSubset or "
        #             "ConcatAtomsData!".format(dataset)
        #         )
        super(ConcatAtomsData, self).__init__(datasets)
        self.propertity_load = datasets[0].propertity_load

        #use datasets[0] as reference
        for key,val in vars(datasets[0]).items():
            # self.(eval(key)) = val
            setattr(self, key, val)

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        
        _, properties =self.datasets[dataset_idx].get_properties(sample_idx,self._propertity_load)
        properties["_idx"] = np.array([idx], dtype=np.int)

        return torchify_dict(properties)
    

    


class AtomsDataSubset():#Subset(dataset, indices)
    r"""
    Subset of an atomistic dataset at specified indices.
    Arguments:
        dataset (torch.utils.data.Dataset): atomistic dataset
        indices (sequence): subset indices
    """

    def __init__(self, dataset, indices, is_aug):
        
        super().__init__()
        self.dataset = dataset
        self.indices = indices
        self._propertity_load = None
        self.is_aug = is_aug
        
        for key,val in vars(dataset).items():
            # self.(eval(key)) = val
            setattr(self, key, val)
            

    def __getitem__(self, idx):
        _, properties = self.dataset.get_properties(self.indices[idx], self.propertity_load)
        properties["_idx"] = np.array([idx], dtype=np.int)
        if self.is_aug:
            properties[self.is_aug['property']] = \
                self.get_augmentation(properties[self.is_aug['property']], *self.is_aug['args'])
        return torchify_dict(properties)

    def __len__(self):
        return len(self.indices)

    def get_augmentation(self, train_batch, patch_size):
        #print(train_batch.shape)
        aug_func = Compose([
            Rotate((-15, 15), (0, 0), (0, 0), p=0.5),
            RandomCropFromBorders(crop_value=0.1, p=0.5),
            # ElasticTransform((0, 0.25), interpolation=2, p=0.1),
            # Resize(patch_size, interpolation=1, resize_type=0, always_apply=True, p=1.0),
            Flip(0, p=0.5),
            Flip(1, p=0.5),
            Flip(2, p=0.5),
            RandomRotate90((1, 2), p=0.5),
            # GaussianNoise(var_limit=(0, 5), p=0.2),
            # RandomGamma(gamma_limit=(0.5, 1.5), p=0.2),
            PadIfNeeded((patch_size)),
        ], p=1.0)
        resolution = patch_size[0]
        batch_size = train_batch.shape[0]
        #print(train_batch.shape)
        img = train_batch.reshape(-1, 4, resolution**3)[:, 0, :].reshape(-1,resolution,resolution,resolution)
        output = []
        for i in range(len(img)):
            data = {'image': img[i]}
            output = aug_func(**data)['image'] if i==0 else np.concatenate((output,aug_func(**data)['image']), axis=0)
        train_batch = train_batch.reshape(-1, 4, resolution**3).copy()
        train_batch[:, 0, :] = output.reshape(-1,resolution**3)
        train_batch = train_batch.reshape(batch_size, -1)
        output = train_batch
        #print(output.shape)
        return output


def _convert_atoms(
    atoms,
    neighbor_provider=Allatoms_NeighborProvider(),
    collect_triples=False,
    centering_function=None,
    output=None,
):
    """
    Helper function to convert ASE atoms object to SchNetPack input format.

    Args:
        atoms (ase.Atoms): Atoms object of molecule
        environment_provider (callable): Neighbor list provider.
        collect_triples (bool, optional): Set to True if angular features are needed.
        centering_function (callable or None): Function for calculating center of
            molecule (center of mass/geometry/...). Center will be subtracted from
            positions.
        output (dict): Destination for converted atoms, if not None

    Returns:
        dict of torch.Tensor: Properties including neighbor lists and masks
            reformated into SchNetPack input format.

    """
    if output is None:
        inputs = {}
    else:
        inputs = output

    # Elemental composition
    inputs[Properties.Z] = atoms.numbers.astype(np.int)
    positions = atoms.positions.astype(np.float32)
    if centering_function:
        positions -= centering_function(atoms)
    inputs[Properties.R] = positions

    # get atom environment
    nbh_idx, offsets = neighbor_provider.get_neighbor(atoms)

    # Get neighbors and neighbor mask
    inputs[Properties.neighbors] = nbh_idx.astype(np.int)

    # Get cells
    inputs[Properties.cell] = np.array(atoms.cell.array, dtype=np.float32)
    inputs[Properties.cell_offset] = offsets.astype(np.float32)

    # If requested get neighbor lists for triples
    if collect_triples:
        nbh_idx_j, nbh_idx_k, offset_idx_j, offset_idx_k = collect_atom_triples(nbh_idx)
        inputs[Properties.neighbor_pairs_j] = nbh_idx_j.astype(np.int)
        inputs[Properties.neighbor_pairs_k] = nbh_idx_k.astype(np.int)

        inputs[Properties.neighbor_offsets_j] = offset_idx_j.astype(np.int)
        inputs[Properties.neighbor_offsets_k] = offset_idx_k.astype(np.int)

    return inputs




# class AtomsConverter:
#     """
#     Convert ASE atoms object to an input suitable for the SchNetPack
#     ML models.

#     Args:
#         environment_provider (callable): Neighbor list provider.
#         collect_triples (bool, optional): Set to True if angular features are needed.
#         device (str): Device for computation (default='cpu')
#     """

#     def __init__(
#         self,
#         neighbor_provider=Allatoms_NeighborProvider(),
#         collect_triples=False,
#         device=torch.device("cpu"),
#     ):
#         self.neighbor_provider = neighbor_provider
#         self.collect_triples = collect_triples

#         # Get device
#         self.device = device

#     def __call__(self, atoms):
#         """
#         Args:
#             atoms (ase.Atoms): Atoms object of molecule

#         Returns:
#             dict of torch.Tensor: Properties including neighbor lists and masks
#                 reformated into SchNetPack input format.
#         """
#         inputs = _convert_atoms(atoms, self.neighbor_provider, self.collect_triples)
#         inputs = torchify_dict(inputs)

#         # Calculate masks
#         inputs[Properties.atom_mask] = torch.ones_like(inputs[Properties.Z]).float()
#         mask = inputs[Properties.neighbors] >= 0
#         inputs[Properties.neighbor_mask] = mask.float()
#         inputs[Properties.neighbors] = (
#             inputs[Properties.neighbors] * inputs[Properties.neighbor_mask].long()
#         )

#         if self.collect_triples:
#             mask_triples = torch.ones_like(inputs[Properties.neighbor_pairs_j])
#             mask_triples[inputs[Properties.neighbor_pairs_j] < 0] = 0
#             mask_triples[inputs[Properties.neighbor_pairs_k] < 0] = 0
#             inputs[Properties.neighbor_pairs_mask] = mask_triples.float()

#         # Add batch dimension and move to CPU/GPU
#         for key, value in inputs.items():
#             inputs[key] = value.unsqueeze(0).to(self.device)

#         return inputs
