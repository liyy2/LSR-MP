import bisect
import pickle
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import lmdb


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
    energy = 'energy'
    forces = 'forces'
    def __init__(self, path, transforms = [], name = None):
        super(LmdbDataset, self).__init__()
        self.path = Path(path) if isinstance(path, str) else [Path(p) for p in path]
        self.num_sub_datasets = 1 if isinstance(path, Path) else len(path)
        db_paths = sorted(self.path.glob("*.lmdb")) if isinstance(self.path, Path) else [sorted(p.glob("*.lmdb")) for p in self.path]
        if isinstance(path, list):
            identifier = [len(db_paths[i]) * [i] for i in range(self.num_sub_datasets)]
            db_paths = [item for sublist in db_paths for item in sublist] # flatten list
            identifier = np.array([item for sublist in identifier for item in sublist])
            self.identifier  = identifier
        assert len(db_paths) > 0, f"No LMDBs found in '{self.path}'"
        self._keys, self.envs = [], []
        self.db_paths = db_paths
        self.open_db()
        self.transforms = transforms
        self.name = name
    
    def open_db(self):
        for db_path in self.db_paths:
            self.envs.append(self.connect_db(db_path))
            length = pickle.loads(
                self.envs[-1].begin().get("length".encode("ascii"))
            )
            self._keys.append(list(range(length)))

        keylens = [len(k) for k in self._keys]
        self._keylen_cumulative = np.cumsum(keylens).tolist()
        self.num_samples = sum(keylens)

        if isinstance(self.path, list):
            self.sample_list = []
            for i in range(self.num_sub_datasets):
                self.sample_list.append(np.sum(np.array(keylens)[self.identifier == i]))
            
           
   
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if len(self.envs) == 0:
            self.open_db()   
        if isinstance(self.path, list):
            flag = 1
        elif isinstance(self.path, Path):
            if not self.path.is_file():
                flag = 1
            else: 
                flag = 0
        else:
            raise ValueError("Path should be either a list or a Path object")
        
        if flag == 0:
            datapoint_pickled = self.env.begin().get(self._keys[idx])
            data_object = pickle.loads(datapoint_pickled)
        else:
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
            data_object.id = el_idx #f"{db_idx}_{el_idx}"



        data = data_object
        for transform in self.transforms:
            data = transform(data)
        return data

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
            self.envs = []
        else:
            self.env.close()
            self.env = None
            