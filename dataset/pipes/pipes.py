import os
import pickle
import logging
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from ..data_util import crop_pc, voxelize
from ..build import DATASETS


@DATASETS.register_module()
class PIPES(Dataset):

    """pipes dataset, loading the subsampled entire room as input without block/sphere subsampling.
    number of points per room in average, median, and std: (794855.5, 1005913.0147058824, 939501.4733064277)
    Args:
        data_root (str, optional): Defaults to 'data/pipes/pool'.
        voxel_size (float, optional): the voxel size for donwampling. Defaults to 0.04.
        voxel_max (_type_, optional): subsample the max number of point per point cloud. Set None to use all points.  Defaults to None.
        split (str, optional): Defaults to 'train'.
        transform (_type_, optional): Defaults to None.
        loop (int, optional): split loops for each epoch. Defaults to 1.
        presample (bool, optional): wheter to downsample each point cloud before training. Set to False to downsample on-the-fly. Defaults to False.
        variable (bool, optional): where to use the original number of points. The number of point per point cloud is variable. Defaults to False.
    """
    def __init__(self,
                 voxel_size: float = 0.04,
                 voxel_max=None,
                 split: str = 'train',
                 transform=None,
                 classes=None,
                 num_classes: int = 1,
                 num_per_class=None,
                 cmap=None,
                 gravity_dim: int = 2,
                 data_root: str = '/home',
                 loop: int = 1,
                 presample: bool = False,
                 variable: bool = False,
                 shuffle: bool = True,
                 ):

        super().__init__()
        self.split = split
        self.voxel_size = voxel_size
        self.transform = transform
        self.voxel_max = voxel_max
        self.classes = classes
        self.num_classes = num_classes
        self.cmap = cmap
        self.gravity_dim = gravity_dim
        self.data_root = data_root    
        self.voxel_size = voxel_size 
        self.loop = loop
        self.presample = presample
        self.variable = variable
        self.shuffle = shuffle


        if split == 'train':
            raw_root = os.path.join(data_root, 'train')
            self.raw_root = raw_root
            data_list = sorted(os.listdir(raw_root))
            data_list = [item[:-4] for item in data_list]
            self.data_list = [item for item in data_list]     #  TODO de aqui coge train
        else:
            raw_root = os.path.join(data_root, 'val')
            self.raw_root = raw_root
            data_list = sorted(os.listdir(raw_root))
            data_list = [item[:-4] for item in data_list]
            self.data_list = [item for item in data_list]      #  TODO de aqui coge val
            for idx, name in enumerate(self.data_list):
                print(str(idx+1) + " - " + str(name))    

        np.random.seed(0)
        self.data = []
        for item in tqdm(self.data_list, desc=f'Loading pipes {split} split'):
            data_path = os.path.join(raw_root, item + '.npy')
            cdata = np.load(data_path).astype(np.float32)
            cdata[:, :3] -= np.min(cdata[:, :3], 0)
            if voxel_size:
                coord, feat, label = cdata[:,0:3], cdata[:, 3:6], cdata[:, 6:7]
                uniq_idx = voxelize(coord, voxel_size)
                coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
                cdata = np.hstack((coord, feat, label))
            self.data.append(cdata)
        npoints = np.array([len(data) for data in self.data])
        logging.info('split: %s, median npoints %.1f, avg num points %.1f, std %.1f' % (
            self.split, np.median(npoints), np.average(npoints), np.std(npoints)))

        self.data_idx = np.arange(len(self.data_list))
        assert len(self.data_idx) > 0
        logging.info(f"Totally {len(self.data_idx)} samples in {split} set")

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        if self.presample:
            coord, feat, label = np.split(self.data[data_idx], [3, 6], axis=1)
        else:
            data_path = os.path.join(
                self.raw_root, self.data_list[data_idx] + '.npy')
            cdata = np.load(data_path).astype(np.float32)
            cdata[:, :3] -= np.min(cdata[:, :3], 0)
            coord, feat, label = cdata[:, :3], cdata[:, 3:6], cdata[:, 6:7]
            coord, feat, label = crop_pc(
                coord, feat, label, self.split, self.voxel_size, self.voxel_max,
                downsample=not self.presample, variable=self.variable, shuffle=self.shuffle)
        label = label.squeeze(-1).astype(np.long)
        data = {'pos': coord, 'x': feat, 'y': label}
        # pre-process.
        if self.transform is not None:
            data = self.transform(data)

        if 'heights' not in data.keys():
            data['heights'] =  torch.from_numpy(coord[:, self.gravity_dim:self.gravity_dim+1].astype(np.float32))
        return data

    def __len__(self):
        return len(self.data_idx) * self.loop

