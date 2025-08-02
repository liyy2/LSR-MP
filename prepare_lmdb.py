# import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import Value
import torch
import os
import pickle
from schnetpack.datasets import *
import lmdb
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.transforms.radius_graph import RadiusGraph
from lightnp.LSRM.utils import build_neighborhood_n_interaction, build_label, build_grouping_graph,build_label_two
from rdkit_label_builder import rdkit_label_builder
import argparse


class MD22(DownloadableAtomsData):
    """
    MD22 benchmark data set for molecular dynamics of small molecules
    containing molecular forces.

    Args:
        dbpath (str): path to database
        molecule (str or None): Name of molecule to load into database. Allowed are:
                            aspirin
                            benzene
                            ethanol
                            malonaldehyde
                            naphthalene
                            salicylic_acid
                            toluene
                            uracil
        subset (list, optional): Deprecated! Do not use! Subsets are created with
            AtomsDataSubset class.
        download (bool): set true if dataset should be downloaded
            (default: True)
        collect_triples (bool): set true if triples for angular functions
            should be computed (default: False)
        load_only (list, optional): reduced set of properties to be loaded
        environment_provider (spk.environment.BaseEnvironmentProvider): define how
            neighborhood is calculated
            (default=spk.environment.SimpleEnvironmentProvider).


    See: http://quantum-machine.org/datasets/
    """

    energy = "energy"
    forces = "forces"

    datasets_dict = dict(
        aspirin="aspirin_dft.npz",
        # aspirin_ccsd='aspirin_ccsd.zip',
        azobenzene="azobenzene_dft.npz",
        benzene="benzene_dft.npz",
        ethanol="ethanol_dft.npz",
        # ethanol_ccsdt='ethanol_ccsd_t.zip',
        malonaldehyde="malonaldehyde_dft.npz",
        # malonaldehyde_ccsdt='malonaldehyde_ccsd_t.zip',
        naphthalene="naphthalene_dft.npz",
        paracetamol="paracetamol_dft.npz",
        salicylic_acid="salicylic_dft.npz",
        toluene="toluene_dft.npz",
        # toluene_ccsdt='toluene_ccsd_t.zip',
        uracil="uracil_dft.npz",
        AT_AT_CG_CG = "md22_AT-AT-CG-CG.npz",
        AT_AT = "md22_AT-AT.npz",
        stachyose = "md22_stachyose.npz",
        DHA = "md22_DHA.npz",
        Ac_Ala3_NHMe = "md22_Ac-Ala3-NHMe.npz",
        buckeyball_catcher = "md22_buckeyball_catcher.npz",
        double_walled_nanotube = "md22_double_walled_nanotube.npz", 
    )

    existing_datasets = datasets_dict.keys()

    def __init__(
        self,
        dbpath,
        molecule=None,
        subset=None,
        download=True,
        collect_triples=False,
        load_only=None,
        environment_provider=spk.environment.SimpleEnvironmentProvider(),
    ):
        if not os.path.exists(dbpath) and molecule is None:
            raise False
            # raise AtomsDataError("Provide a valid dbpath or select desired molecule!")

        # if molecule is not None and molecule not in MD17.datasets_dict.keys():
        #     raise False
        #     # raise AtomsDataError("Molecule {} is not supported!".format(molecule))

        self.molecule = molecule

        available_properties = [MD17.energy, MD17.forces]

        super(MD22, self).__init__(
            dbpath=dbpath,
            subset=subset,
            load_only=load_only,
            collect_triples=collect_triples,
            download=download,
            available_properties=available_properties,
            environment_provider=environment_provider,
        )

    def _download(self):

        logging.info("Downloading {} data".format(self.molecule))
        tmpdir = tempfile.mkdtemp("MD")
        rawpath = os.path.join(tmpdir, self.datasets_dict[self.molecule])
        url = (
            "http://www.quantum-machine.org/gdml/data/npz/"
            + self.datasets_dict[self.molecule]
        )

        request.urlretrieve(url, rawpath)

        logging.info("Parsing molecule {:s}".format(self.molecule))

        data = np.load(rawpath)

        numbers = data["z"]
        atoms_list = []
        properties_list = []
        for positions, energies, forces in zip(data["R"], data["E"], data["F"]):
            properties_list.append(dict(energy=energies, forces=forces))
            atoms_list.append(Atoms(positions=positions, numbers=numbers))

        self.add_systems(atoms_list, properties_list)
        self.update_metadata(dict(data_source=self.datasets_dict[self.molecule]))

        logging.info("Cleanining up the mess...")
        logging.info("{} molecule done".format(self.molecule))
        shutil.rmtree(tmpdir)


datapath = None
molecules = None
r = None
out_path = None
num_workers = None
all_molecules = None
label_builder = None
def atom_to_xyz(atom_type, atom_position):
    atom_dict = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}
    with open(os.path.join(out_path, f"error.xyz"), 'a') as f:
        f.write(f'>>>>>>>>>>>>>>>>>>>Error number {counter.value:3d}<<<<<<<<<<<<<<<<<<<<<<<<<')
        for i in range(len(atom_type)):
            f.write(f"{atom_dict[int(atom_type[i].numpy())]:<10}{atom_position[i][0]:>10.5f}{atom_position[i][1]:>10.5f}{atom_position[i][2]:>10.5f}")
        f.write('\n')    

def get_all_molecules():
    global all_molecules
    if dataset_name == 'MD17':
        all_molecules = [
                'ethanol',
                'malonaldehyde',
                'naphthalene',
                'salicylic_acid',
                'toluene',
                'uracil',] 
    elif dataset_name == 'MD22':

        all_molecules = [
        'buckyball_catcher',
        'DHA',
        'double_walled_nanotube',
        'AT_AT_CG_CG',
        'AT_AT',
        'Ac_Ala3_NHMe', 
        "stachyose"]             



def parse_args(jupyter = False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str, default = 'MD22', choices= ['MD17', 'MD22'] , help = 'Dataset to process')
    parser.add_argument('--datapath', type=str, default=None)
    parser.add_argument('--molecule', type=str, default='buckyball_catcher', help='which molecule to process, default is all, seperated by ,')    
    parser.add_argument('--broadcast_radius', type=float, default=3.0)
    parser.add_argument('--out_path', type=str, default=None)
    parser.add_argument('--dataset_identifier', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--min_nodes_foreachGroup', type=int, default=4)
    parser.add_argument('--ignore_errors', type=bool, default=True)
    parser.add_argument('--group_builder', type=str, default='kmeans', choices=['rdkit', 'kmeans','spectral','spectral_two'], help='which group builder to use')
    parser.add_argument('--fixed', type=int, default=0, choices=[0,1], help='all the data use the same graph')
    parser.add_argument('--subtract_mean', type=int, default=0, choices=[0,1], help='whether to normalize the dataset')
    if(jupyter):
        args = parser.parse_args(args = [])
    else:
        args = parser.parse_args()
    # args = parser.parse_args()
    global dataset_name
    dataset_name = args.dataset
    get_all_molecules()
    global datapath
    datapath = args.datapath
    global molecules
    molecules = [mol for mol in args.molecule.split(',')] if args.molecule != 'all' else all_molecules
    # if molecules not in all_molecules:
    #     assert(False)
    global subtract_mean
    subtract_mean = args.subtract_mean  
    global r
    r = args.broadcast_radius
    global out_path
    out_path = args.out_path
    global num_workers
    num_workers = args.num_workers
    global ignore_errors
    ignore_errors = args.ignore_errors
    global label_builder
    label_builder = args.group_builder

    config = {}
    for key, value in vars(args).items():
        if key.startswith('not_'):
            config[key[4:]] = not value
        config[key] = value
    return config


def write_images_to_lmdb(mp_arg):
    db_path, samples, idx, pid, mol, energy_mean, energy_std, config = mp_arg
    
    # Recreate dataset within worker to avoid pickling issues
    dataset = MD22(os.path.join(datapath, f"{dataset_name}_{mol}.db"),
            molecule = mol, load_only =["energy","forces"])
    
    if os.path.exists(db_path):
        os.remove(db_path)
    db = lmdb.open(
        db_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    pbar = tqdm(
        total=len(samples),
        position=pid,
        desc="Preprocessing data into LMDBs",
    )
    
    # if config["fixed"]:
    #     item = dataset[0]
    #     data_fix = Data()
    #     data_fix.pos = item['_positions']
    #     data_fix.atomic_numbers = item['_atomic_numbers'].reshape(-1, 1)
    #     data_fix.num_nodes = data_fix.atomic_numbers.shape[0]
    #     if label_builder == 'kmeans':
    #         nodes_len = data_fix.atomic_numbers.shape[0]
    #         build_label(data_fix, num_labels = int(nodes_len/config["min_nodes_foreachGroup"]),method=config["group_builder"])
    #     else:
    #         rdkit_label_builder(data_fix, config["min_nodes_foreachGroup"])
        
        
    for i in samples:
        # print(i,int(i))
        item = dataset[int(i)]
        # xyz = io.read(item)
        data = Data()
        data.atomic_numbers = item['_atomic_numbers'].reshape(-1, 1)
        if subtract_mean:
            data.energy = item['energy'].unsqueeze(1) - energy_mean # normalize energy
        else:
            data.energy = item['energy'].unsqueeze(1)
        data.forces = item['forces']
        data.num_nodes = data.atomic_numbers.shape[0]
        data.pos = item['_positions']
        neighbor_finder = RadiusGraph(r = r)
        data = neighbor_finder(data)
        min_nodes_foreachGroup = config["min_nodes_foreachGroup"] # for ball catcher is set 10.
        try:
            # if config["fixed"]:
            #     data.labels = data_fix.labels
            #     data.num_labels = data_fix.num_labels
            #     build_grouping_graph(data)
            #     build_neighborhood_n_interaction(data)
            # elif 
            if label_builder != 'rdkit':
                nodes_len = data.atomic_numbers.shape[0]
                if label_builder == "spectral_two":
                    build_label_two(data,num_labels = int(nodes_len/min_nodes_foreachGroup))
                else:
                    build_label(data, num_labels = int(nodes_len/min_nodes_foreachGroup),method = config["group_builder"])
                    build_grouping_graph(data)
                    build_neighborhood_n_interaction(data)
            else:
                rdkit_label_builder(data, min_nodes_foreachGroup)
                build_grouping_graph(data)
                build_neighborhood_n_interaction(data)
        except:
            with counter.get_lock():
                counter.value += 1
            print(f"Error in {i}")
            atom_to_xyz(data.atomic_numbers, data.pos)
            if ignore_errors:
                continue
            else:
                raise('Error in fragmentation') 
        
        
        txn = db.begin(write=True)
        txn.put(
            f"{idx}".encode("ascii"),
            pickle.dumps(data, protocol=-1),
        )
        txn.commit()
        idx += 1
        pbar.update(1)
        # if idx>100:break
    # Save count of objects in lmdb.
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn.commit()
    return []




def main(mol,config):      
    global counter
    counter = Value('i', 0)
    print(f"Processing {mol}...")
    dataset = MD22(os.path.join(datapath, f"{dataset_name}_{mol}.db"),
            molecule = mol, load_only =["energy","forces"])
    # normalized_tag = "_normalized" if subtract_mean else ""
    rmMean = '_rmMean_' if subtract_mean else '' 
    out = os.path.join(out_path, f"{dataset_name}/{mol}/{config['dataset_identifier']}")
        
    os.makedirs(out, exist_ok=True)
    # Initialize lmdb paths
    db_paths = [
        
        os.path.join(out, "data.%04d.lmdb" % i)
        for i in range(num_workers)
    ]
    # Chunk the trajectories into args.num_workers splits
    chunked_txt_files = np.array_split(np.arange(len(dataset)), num_workers)

    # Extract features
    idx = [0] * num_workers

    energys = []
    for i in range(len(dataset)):
        # print(i,int(i))
        energys.append(dataset[i]["energy"])
    energy_mean = torch.mean(torch.asarray(energys)).item()
    energy_std = torch.std(torch.asarray(energys)).item()
    print("energy mean is: ",energy_mean, 
          "energy std is: ", energy_std)
    
    
    pool = Pool(num_workers)
    mp_args = [
        (
            db_paths[i],
            chunked_txt_files[i],
            idx[i],
            i,
            mol,
            energy_mean,
            energy_std,
            config
        )
        for i in range(num_workers)
    ]
    _ = zip(*pool.imap(write_images_to_lmdb, mp_args))
    print('Finished')
    print(f'Error count of {mol}: ', counter.value)
    

if __name__ == "__main__":
    config = parse_args()
    for mol in molecules:
        main(mol,config)
    