import bisect
import pickle
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import lmdb
import os
import torch

class LmdbDataset(Dataset):
    r"""Dataset class to load from LMDB files containing relaxation
    trajectories or single point computations.
    Useful for Structure to Energy & Force (S2EF), Initial State to
    Relaxed State (IS2RS), and Initial State to Relaxed Energy (IS2RE) tasks.
    Args:
            @path: the path to store the data
            @task: the task for the lmdb dataset
            @split: splitting of the data
            @transform: some transformation of the data
    """
    energy = 'energy_U0'
    forces = 'forces'
    def __init__(self, path,
                    task = 's2ef', split = '200k', transform=None, proportion = 0.1):
        super(LmdbDataset, self).__init__()

        if task == "s2ef" and split != "test":
            if split in ["200k", "2M", "20M", "all", "rattled", "md"]:
                output_path = os.path.join(path, task, split, "train")
            else:
                output_path = os.path.join(path, task, "all", split)
        else:
            raise NotImplementedError("Task and split combination not implemented yet")
        self.path = Path(output_path)
        db_paths = sorted(self.path.glob("*.lmdb"))
        # if no lmdb file then perform downloading and extracting
        if len(db_paths) == 0:
            self.download_and_prepare(path, task, split)
        

        db_paths = sorted(self.path.glob("*.lmdb"))
        assert len(db_paths) > 0, f"No LMDBs found in '{self.path}'"

        self.metadata_path = self.path / "metadata.npz"

        self._keys, self.envs = [], []
        for db_path in db_paths:
            self.envs.append(self.connect_db(db_path))
            length = pickle.loads(
                self.envs[-1].begin().get("length".encode("ascii"))
            )
            self._keys.append(list(range(length)))

        keylens = [len(k) for k in self._keys]
        self._keylen_cumulative = np.cumsum(keylens).tolist()
        self.num_samples = sum(keylens)
        self.transform = transform
        self.proportion = proportion


    def __len__(self):
        return int(self.num_samples * self.proportion)

    def __getitem__(self, idx):
        if not self.path.is_file():
            # Figure out which db this should be indexed from.
            db_idx = bisect.bisect(self._keylen_cumulative, idx)
            # Extract index of element within that db.
            el_idx = idx
            if db_idx != 0:
                el_idx = idx - self._keylen_cumulative[db_idx - 1]
            assert el_idx >= 0

            # Return features.
            datapoint_pickled = (
                self.envs[db_idx]
                .begin()
                .get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
            )
            data_object = pickle.loads(datapoint_pickled)
            data_object.id = f"{db_idx}_{el_idx}"
        else:
            datapoint_pickled = self.env.begin().get(self._keys[idx])
            data_object = pickle.loads(datapoint_pickled)

        if self.transform is not None:
            data_object = self.transform(data_object)

        data = data_object
        return {'_positions': data.pos, 'energy_U0':torch.Tensor([data.y]), 'forces':data.force,
        '_atomic_numbers': data.atomic_numbers.long(), '_neighbors': torch.LongTensor(data.neighbor),
        '_cell': data.cell.squeeze(), '_cell_offset': torch.Tensor(data.cell_offsets), '_idx': torch.LongTensor([idx])}

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
        return env

    def close_db(self):
        if not self.path.is_file():
            for env in self.envs:
                env.close()
        else:
            self.env.close()

    def download_and_prepare(path, task, split, num_workers = 8):
        r'''
        Download and prepare the dataset.
        @param path: path to store the data
        @task task: The task for downloading, reference @ ocp officials
        @split: train/test
        '''
        script_dir = '/home/hul/v-yunyangli/lightnp/lightnp/data/datasets/download_ocp.py'
        os.system(f"python {script_dir} \
        --task {task} --split {split}  --get-edges --num-workers {num_workers} \
        --ref-energy --data-path {path}")
    
