import os
import pprint
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision.utils import save_image
from torch.utils.data import Dataset
from torchvision import transforms
from .record_transform import *
import time
import glob

def get_fakes(root_dir):
    image_list = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if '1_fake' in dirnames:
            b_dir = os.path.join(dirpath, '1_fake')
            images = glob.glob(os.path.join(b_dir, '*.*'))
            image_list.extend(images)

    return image_list

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage

    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)



class CapDataset(Dataset):
    def __init__(self, root_dir, data_cap,transform=None,batched_syncing=False,use_inversions=False,seed=17):
        self.root_dir = root_dir
        self.transform = transform
        self.batched_syncing = batched_syncing
        paths = os.listdir(os.path.join(self.root_dir, '0_real'))
        if self.batched_syncing:
            self.recorder = TransformRecorder(self.transform)
        self.files = []
        random.seed(seed)
        paths = sorted(paths)
        if data_cap is not None:
            paths = random.sample(paths, data_cap)

        if use_inversions:
            for path in paths:
                rpath = os.path.join(os.path.join(self.root_dir, '0_real'), path)
                fpath = os.path.join(os.path.join(self.root_dir, '1_fake'),path.replace('.jpg','.png'))
                self.files.append((rpath, 0))
                if not self.batched_syncing:
                    self.files.append((fpath,1))
        else:
            fpaths = get_fakes(root_dir=self.root_dir)
            fpaths = random.sample(fpaths, data_cap)
            for idx, path in enumerate(paths):
                rpath = os.path.join(os.path.join(self.root_dir, '0_real'), path)
                fpath = os.path.join(self.root_dir, fpaths[idx])
                self.files.append((rpath, 0))
                self.files.append((fpath,1))


    def filter_dataset(self, keep_count):
        self.files = self.files[keep_count:]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path, target = self.files[index]
        sample = default_loader(path)#Image.open(path)
        if self.batched_syncing:
            SEED = int(time.time())
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)

            sample = self.transform(sample)
            fpath = path.replace('.jpg','.png').replace('0_real', '1_fake')
            if not os.path.exists(fpath):
                fpath = fpath.replace('.png', '.jpg')
            fsample = default_loader(fpath)
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            fsample = self.transform(fsample)
            return {"img": sample, "target": target, "path": path}, {"img": fsample, "target": 1, "path": fpath}
        else:
            if self.transform is not None:
                sample = self.transform(sample)
            return {"img": sample, "target": target, "path": path}
            
class CapDataset_(Dataset):
    def __init__(self,
                 real_dir,
                 fake_dir,
                 data_cap=None,
                 transform=None,
                 batched_syncing=False,
                 use_inversions=False,
                 seed: int = 17,
                 real_exts = ('.jpg', '.jpeg', '.png'),
                 fake_ext = '.png',
                 file_list: Optional[List[str]] = None):

        self.real_dir        = real_dir
        self.fake_dir        = fake_dir
        self.transform       = transform
        self.batched_syncing = batched_syncing
        self.use_inversions  = use_inversions

        if self.batched_syncing:
            self.recorder = TransformRecorder(self.transform)

        # If a file list isn't provided, create one by reading the directory.
        if file_list is None:
            real_files = [f for f in os.listdir(real_dir)
                          if f.lower().endswith(real_exts)]
            real_files.sort()
            random.seed(seed)
            if data_cap is not None:
                real_files = random.sample(real_files, min(data_cap, len(real_files)))
        # If a list is provided, use it directly.
        else:
            real_files = file_list

        fake_lookup: dict[str, str] = {}
        if use_inversions:
            for f in os.listdir(fake_dir):
                if f.lower().endswith(fake_ext):
                    fake_lookup[os.path.splitext(f)[0]] = os.path.join(fake_dir, f)
        else:
            # fallback list for non-inversion mode
            fake_pool = [os.path.join(fake_dir, f)
                         for f in os.listdir(fake_dir)
                         if f.lower().endswith(real_exts)]
            random.shuffle(fake_pool)

        self.files: list[tuple[str, int]] = []
        self._fake_index: dict[str, str] = {}   # only used when syncing

        for idx, rf in enumerate(tqdm(real_files, desc="Indexing real/fake pairs")):
            r_path = os.path.join(real_dir, rf)
            base   = os.path.splitext(rf)[0]

            if use_inversions:
                f_path = fake_lookup.get(base, None)
            else:
                f_path = fake_pool[idx % len(fake_pool)] if fake_pool else None

            # keep only reals with a mate
            if f_path is None or not os.path.isfile(f_path):
                continue

            self.files.append((r_path, 0))

            if self.batched_syncing:
                self._fake_index[base] = f_path   # retrieved on-the-fly later
            else:
                self.files.append((f_path, 1))

    def filter_dataset(self, keep_count):
        self.files = self.files[keep_count:]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path, target = self.files[index]
        sample = default_loader(path)

        if self.batched_syncing:
            SEED = int(time.time())
            random.seed(SEED);  np.random.seed(SEED)
            torch.manual_seed(SEED);  torch.cuda.manual_seed(SEED)

            if self.transform is not None:
                sample = self.transform(sample)

            base  = os.path.splitext(os.path.basename(path))[0]
            fpath = self._fake_index[base]                       # mapped at init
            fsample = default_loader(fpath)

            random.seed(SEED);  np.random.seed(SEED)
            torch.manual_seed(SEED);  torch.cuda.manual_seed(SEED)

            if self.transform is not None:
                fsample = self.transform(fsample)

            return (
                {"img": sample,  "target": target, "path": path},
                {"img": fsample, "target": 1,      "path": fpath},
            )
        else:
            if self.transform is not None:
                sample = self.transform(sample)
            return {"img": sample, "target": target, "path": path}