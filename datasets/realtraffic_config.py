# =========================== A2I Copyright Header ===========================
#
# Copyright (c) 2003-2019 University of Oxford. All rights reserved.
# Authors: Applied AI Lab, Oxford Robotics Institute, University of Oxford
#          https://ori.ox.ac.uk/labs/a2i/
#
# This file is the property of the University of Oxford.
# Redistribution and use in source and binary forms, with or without
# modification, is not permitted without an explicit licensing agreement
# (research or commercial). No warranty, explicit or implicit, provided.
#
# =========================== A2I Copyright Header ===========================

import os
import json
# from shutil import copytree

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
import PIL
from PIL import Image, ImageFile

from forge import flags
from forge.experiment_tools import fprint

from utils.misc import loader_throughput, np_img_centre_crop

# from third_party.shapestacks.shapestacks_provider import _get_filenames_with_labels
# from third_party.shapestacks.segmentation_utils import load_segmap_as_matrix


flags.DEFINE_string('data_folder', 'data/cars_real_final/', 'Path to data folder.')
# flags.DEFINE_string('split_name', 'default', '{default, blocks_all, css_all}')
flags.DEFINE_integer('img_size', 64, 'Dimension of images. Images are square.')
flags.DEFINE_integer('num_obj_min', 3, 'Minimum number of objects in the image.')
flags.DEFINE_integer('num_obj_max', 6, 'Maximum number of objects in the image.')
# flags.DEFINE_boolean('shuffle_test', False, 'Shuffle test set.')

flags.DEFINE_integer('num_workers', 4, 'Number of threads for loading data.')
# flags.DEFINE_boolean('load_instances', False, 'Load instances.')
# flags.DEFINE_boolean('copy_to_tmp', False, 'Copy files to /tmp.')

flags.DEFINE_integer('K_steps', 9, 'Number of recurrent steps.')


# MAX_SHAPES = 6
# CENTRE_CROP = 196


def load(cfg, **unused_kwargs):
  del unused_kwargs
  if not os.path.exists(cfg.data_folder):
    raise Exception("Data folder does not exist.")
  print(f"Using {cfg.num_workers} data workers.")

  # Training
  tng_set = CarsRealTraffic(data_root=cfg.data_folder,
                        train=True,
                        image_size=cfg.img_size)
  tng_loader = DataLoader(tng_set,
                          batch_size=cfg.batch_size,
                          shuffle=True,
                          num_workers=cfg.num_workers)
  # TODO: Validation
  val_set = CarsRealTraffic(data_root=cfg.data_folder,
                        train=False,
                        image_size=cfg.img_size)
  val_loader = DataLoader(val_set,
                          batch_size=cfg.batch_size,
                          shuffle=False,
                          num_workers=cfg.num_workers)
  # Test
  tst_set = CarsRealTraffic(data_root=cfg.data_folder,
                        train=False,
                        image_size=cfg.img_size)
  tst_loader = DataLoader(tst_set,
                          batch_size=1,
                          shuffle=False,
                          num_workers=1)

  # Throughput stats
  loader_throughput(tng_loader)

  return (tng_loader, val_loader, tst_loader)


class CarsRealTraffic(Dataset):
    
  """Data Handler that loads cars data."""

  def __init__(self, data_root, train=True, seq_len=30, image_size=64, **kwargs):
    self.root_dir = data_root 
    self.image_size = image_size

    self.dir = sorted([self.root_dir+f for f in os.listdir(self.root_dir) if f[0]=='f'])
    # split data in training and test set as 80/20
    split_idx = int(np.rint(len(self.dir) * 0.8))
    if train:
      self.dir = self.dir[:split_idx]  # 80% (491) of 614 videos total
    else:
      self.dir = self.dir[split_idx:]  # 20% (123) of 614 videos total

    self.data = []
    for i in range(len(self.dir)):
      dir_name = self.dir[i]
      seq_ims  = sorted([dir_name+'/'+f for f in os.listdir(dir_name) if f[-5:]=='.jpeg'])
      for j in range(len(seq_ims)-3*seq_len):
        ## Look at the h option and load only 2...
        self.data.append(seq_ims[j:j+2*seq_len:2])

    # self.N = int(kwargs['epoch_size'])
    self.N = len(self.data)
    self.seq_len = seq_len
    # Transforms
    # T = [transforms.Resize(image_size)]
    T = []
    T.append(transforms.ToTensor())
    self.transform = transforms.Compose(T)

  def __len__(self):
    return self.N
  
  def __getitem__(self, index):
    images = self.data[index%len(self.data)]
    rnd_idx = np.random.randint(low=0, high=len(images))  # pick a random image
    im = (np.asarray(Image.open(images[rnd_idx]).convert('RGB')\
        .resize((self.image_size,self.image_size),PIL.Image.LANCZOS))\
        .reshape(self.image_size, self.image_size, 3))
    return {'input' : self.transform(im)}
