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


flags.DEFINE_string('data_folder', 'data/CLEVR3', 'Path to data folder.')
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
  tng_set = ClevrOCFDataset(data_root=cfg.data_folder,
                        train=True,
                        image_size=cfg.img_size)
  tng_loader = DataLoader(tng_set,
                          batch_size=cfg.batch_size,
                          shuffle=True,
                          num_workers=cfg.num_workers)
  # TODO: Validation
  val_set = ClevrOCFDataset(data_root=cfg.data_folder,
                        train=False,
                        image_size=cfg.img_size)
  val_loader = DataLoader(val_set,
                          batch_size=cfg.batch_size,
                          shuffle=False,
                          num_workers=cfg.num_workers)
  # Test
  tst_set = ClevrOCFDataset(data_root=cfg.data_folder,
                        train=False,
                        image_size=cfg.img_size)
  tst_loader = DataLoader(tst_set,
                          batch_size=1,
                          shuffle=False,
                          num_workers=1)

  # Throughput stats
  loader_throughput(tng_loader)

  return (tng_loader, val_loader, tst_loader)


class ClevrOCFDataset(Dataset):
    
  """Data Handler that loads 2 balls in a Bowl synthetic data."""

  def __init__(self, data_root, train=True, seq_len=1, image_size=64, epoch_size=300, **kwargs):
    self.root_dir = data_root 
    self.train = train
    if train:
      self.data_dir = '%s/train/' % self.root_dir
      self.ordered = False
    else:
      self.data_dir = '%s/test/' % self.root_dir
      self.ordered = True 
    self.dirs = []
    for d1 in os.listdir(self.data_dir):
      if d1[-4:]=='.png':
        self.dirs.append('%s/%s' % (self.data_dir, d1))

    self.image_size = image_size 
    self.seed_is_set = False # multi threaded loading
    self.d = 0
    self.N = len(self.dirs)
    # Transforms
    # T = [transforms.Resize(image_size)]
    T = []
    T.append(transforms.ToTensor())
    self.transform = transforms.Compose(T)
      
  def set_seed(self, seed):
    if not self.seed_is_set:
      self.seed_is_set = True
      np.random.seed(seed)
        
  def __len__(self):
    return self.N

  # def get_seq(self, index):
  #   d = self.dirs[index%(len(self.dirs))]
  #   image_seq = []
  #   im = (np.asarray(Image.open(d).convert('RGB')\
  #       .resize((self.image_size, self.image_size), PIL.Image.LANCZOS))\
  #       .reshape(1, self.image_size, self.image_size, 3)
  #       # .astype('float32') \
  #       # - 127.5) / 255
  #   image_seq.append(im)
  #   image_seq = np.concatenate(image_seq, axis=0)
  #   image_seq = torch.from_numpy(image_seq[0,:,:,:]).permute(2,0,1)
  #   return image_seq


  def __getitem__(self, index):
    self.set_seed(index)
    d = self.dirs[index%(len(self.dirs))]
    try:
      im = (np.asarray(Image.open(d).convert('RGB')\
          .resize((self.image_size, self.image_size), PIL.Image.LANCZOS))\
          .reshape(self.image_size, self.image_size, 3))
    except (IOError, SyntaxError) as excp:
      print(excp)
      print(f">>> Corrupt image file '{d}'!")
      im = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
    return {'input' : self.transform(im)}
