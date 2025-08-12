'''                                        
Copyright 2024 Image Processing Research Group of University Federico
II of Naples ('GRIP-UNINA'). All rights reserved.
                        
Licensed under the Apache License, Version 2.0 (the "License");       
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at                    
                                           
    http://www.apache.org/licenses/LICENSE-2.0
                                                      
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,    
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                         
See the License for the specific language governing permissions and
limitations under the License.
''' 

import os
import random
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from collections import defaultdict
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler
from torchvision.datasets import ImageFolder
from PIL import ImageFile
from .cap_dset import CapDataset, CapDataset_
from .processing import make_processing, add_processing_arguments
ImageFile.LOAD_TRUNCATED_IMAGES = True
#np.random.seed(seed=17)

class ListToDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

class PathNameDataset(ImageFolder):
    def __init__(self, **keys):
        super().__init__(**keys)

    def __getitem__(self, index):
        # Loading sample
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return {"img": sample, "target": target, "path": path}


# def get_dataset(opt, dataroot):
#     dset_lst = []
#     # NOTE: get the classes for the current directory
#     if os.path.isdir(os.path.join(dataroot, "0_real")):
#         classes = ['.',]
#     else:
#         classes = os.listdir(dataroot)

#     transform = make_processing(opt)
#     print('CLASSES:', classes)
#     for cls in classes:
#         root = dataroot + "/" + cls
#         if os.path.isdir(root + "/0_real"):
#             dset = PathNameDataset(root=root, transform=transform)
#             print("#images %6d in %s" % (len(dset), root))
#             dset_lst.append(dset)

#     return torch.utils.data.ConcatDataset(dset_lst)


def get_bal_sampler(dataset):
    targets = [0,1]
    #targets = []
    #for d in dataset.datasets:
    #    targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1.0 / torch.tensor(ratio, dtype=torch.float)
    if torch.all(w==w[0]):
        print(f"RandomSampler: # {ratio}")
        sampler = RandomSampler(dataset, replacement = False)
    else:
        w = w / torch.sum(w)
        print(f"WeightedRandomSampler: # {ratio}, Weightes {w}")
        sample_weights = w[targets]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights)
        )
    return sampler


def add_dataloader_arguments(parser):
    # parser is an argparse.ArgumentParser
    #
    # This adds the arguments necessary for dataloader
    # parser.add_argument(
    #     "--dataroot", type=str, help="Path to the dataset to use during training"
    # )
    # The path containing the train and the validation data to train on
    parser.add_argument('--batched_syncing', action='store_true', help='synchronize the batches')
    parser.add_argument('--adm', action='store_true', help='account for ADM training')
    parser.add_argument("--data_cap", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_threads", default=8, type=int, help='# threads for loading data')
    parser.add_argument("--seed", default=8, type=int, help='# SEED')
    parser.add_argument("--real_dir",  type=str, default=None,
                        help="Folder with genuine images (overrides dataroot)")
    parser.add_argument("--fake_dir",  type=str, default=None,
                        help="Folder with counterfeit images (overrides dataroot)")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction reserved for validation when real_dir/fake_dir are used")
    parser = add_processing_arguments(parser)
    return parser

def create_dataloader(opt, subdir=".", is_train=True):

    np.random.seed(opt.seed)

    if getattr(opt, "real_dir", None) and getattr(opt, "fake_dir", None):

        if not hasattr(opt, "_tfds"):
            tfm = make_processing(opt)

            ds_train = CapDataset_(
                opt.real_dir, opt.fake_dir,
                transform=tfm,
                batched_syncing=opt.batched_syncing,
                data_cap=opt.data_cap,
                use_inversions=opt.use_inversions,
                seed=opt.seed,
            )

            val_cap = None
            if opt.data_cap is not None:
                val_cap = max(1, opt.data_cap // 5)

            ds_valid = CapDataset_(
                opt.real_dir, opt.fake_dir,
                transform=tfm,
                batched_syncing=False,        # flat batches
                data_cap=val_cap,             # smaller set
                use_inversions=opt.use_inversions,
                seed=opt.seed,
            )

            val_len   = int(len(ds_train) * opt.val_split)
            train_len = len(ds_train) - val_len
            gen = torch.Generator().manual_seed(opt.seed)
            idx_train, _ = torch.utils.data.random_split(
                range(len(ds_train)), [train_len, val_len], generator=gen
            )

            opt._tfds = {
                "train": Subset(ds_train, idx_train.indices),
                "valid": ds_valid,            # ← no Subset here
            }

        dataset = opt._tfds["train" if is_train else "valid"]
    else:
        dataroot = os.path.join(opt.dataroot, subdir)
        if (opt.data_cap is not None or opt.batched_syncing) and is_train:
            tfm = make_processing(opt)
            dataset = CapDataset(root_dir=dataroot, data_cap=opt.data_cap, transform=tfm, batched_syncing=opt.batched_syncing, use_inversions=opt.use_inversions, seed=opt.seed)
            dataset = torch.utils.data.ConcatDataset([dataset])
        else:
            dataset = get_dataset(opt, dataroot)
    loader = DataLoader(dataset, batch_size=opt.batch_size, sampler=get_bal_sampler(dataset) if is_train else None, num_workers=int(opt.num_threads))
    return loader

