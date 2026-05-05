import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class ScanObjectNN(Dataset):
    """ScanObjectNN h5 loader.

    Expected layout::

        <data_dir>/<split>/training_objectdataset<version>.h5
        <data_dir>/<split>/test_objectdataset<version>.h5

    Default ``split='main_split'`` and ``version='_augmentedrot_scale75'`` give
    the PB_T50_RS variant used in the Platonic Transformer paper.
    """

    def __init__(
        self,
        data_dir,
        subset='train',
        split='main_split',
        version='_augmentedrot_scale75',
        transform=None,
        nbr_pts=2048,
    ):
        super().__init__()
        self.root = data_dir
        self.subset = subset
        self.split = split
        self.version = version
        self.transform = transform
        self.nbr_pts = nbr_pts

        if subset == 'train':
            fname = f'training_objectdataset{version}.h5'
        elif subset == 'test':
            fname = f'test_objectdataset{version}.h5'
        else:
            raise ValueError(f"subset must be 'train' or 'test', got '{subset}'")

        path = os.path.join(self.root, self.split, fname)
        with h5py.File(path, 'r') as h5:
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)

        if nbr_pts > self.points.shape[1]:
            raise ValueError(
                f"Requested {nbr_pts} points but the file holds only "
                f"{self.points.shape[1]} per object."
            )

        self.nbr_classes = 15
        if int(self.labels.max()) != self.nbr_classes - 1:
            raise ValueError(
                f"Expected labels in [0,{self.nbr_classes - 1}] but got "
                f"max label {int(self.labels.max())}"
            )

        print(f'Successfully loaded ScanObjectNN [{subset}] shape={self.points.shape}')

    def __getitem__(self, idx):
        pt_idxs = np.random.choice(self.points.shape[1], (self.nbr_pts,), replace=False)
        current_points = self.points[idx, pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        label = torch.tensor([int(self.labels[idx])])

        data = Data(pos=current_points, y=label)

        if self.transform is not None:
            return self.transform(data)
        return data

    def __len__(self):
        return self.points.shape[0]
