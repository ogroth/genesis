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


flags.DEFINE_string('data_folder', 'data/CLEVR_v1.0', 'Path to data folder.')
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
  tng_set = ClevrDataset(data_root=cfg.data_folder,
                        mode='train',
                        image_size=cfg.img_size,
                        num_objects_min=cfg.num_obj_min,
                        num_objects_max=cfg.num_obj_max)
  tng_loader = DataLoader(tng_set,
                          batch_size=cfg.batch_size,
                          shuffle=True,
                          num_workers=cfg.num_workers)
  # TODO: Validation
  val_set = ClevrDataset(data_root=cfg.data_folder,
                        mode='val',
                        image_size=cfg.img_size,
                        num_objects_min=cfg.num_obj_min,
                        num_objects_max=cfg.num_obj_max)
  val_loader = DataLoader(val_set,
                          batch_size=cfg.batch_size,
                          shuffle=False,
                          num_workers=cfg.num_workers)
  # Test
  tst_set = ClevrDataset(data_root=cfg.data_folder,
                        mode='test',
                        image_size=cfg.img_size,
                        num_objects_min=cfg.num_obj_min,
                        num_objects_max=cfg.num_obj_max)
  tst_loader = DataLoader(tst_set,
                          batch_size=1,
                          shuffle=False,
                          num_workers=1)

  # Throughput stats
  loader_throughput(tng_loader)

  return (tng_loader, val_loader, tst_loader)


class ClevrDataset(Dataset):

  def __init__(
      self,
      data_root, mode='train', image_size=64,
      num_objects_min=3, num_objects_max=6):
    self.root_dir = data_root
    self.image_files = []
    self.image_annotations = []
    if mode == 'train':  # merge train and val splits
      for split in ['train', 'val']:
        # load file names and annotations
        self.image_files.extend([
            os.path.join(self.root_dir, 'images', split, fn) \
            for fn in sorted(os.listdir(os.path.join(data_root, 'images', split)))
        ])
        with open(os.path.join(data_root, 'scenes', 'CLEVR_%s_scenes.json' % split)) as fp:
          train_scenes = json.load(fp)['scenes']
        self.image_annotations.extend(train_scenes)
      # filter by number of objects
      obj_cnt_list = list(
          zip(
              self.image_files,
              [len(scn['objects']) for scn in self.image_annotations]
          )
      )
      obj_cnt_filter_index = [
          t[1] >= num_objects_min and t[1] <= num_objects_max \
          for t in obj_cnt_list
      ]
      # apply filter
      self.image_files = [self.image_files[idx] for idx, t in enumerate(obj_cnt_filter_index) if t]
      self.image_annotations = [self.image_annotations[idx] for idx, t in enumerate(obj_cnt_filter_index) if t]
      print(">>> Loaded %d images from split 'train+val'." % len(self.image_files))
    elif mode == 'val':
      for split in ['val']:
        # load file names and annotations
        self.image_files.extend([
            os.path.join(self.root_dir, 'images', split, fn) \
            for fn in sorted(os.listdir(os.path.join(data_root, 'images', split)))
        ])
        with open(os.path.join(data_root, 'scenes', 'CLEVR_%s_scenes.json' % split)) as fp:
          train_scenes = json.load(fp)['scenes']
        self.image_annotations.extend(train_scenes)
      # filter by number of objects
      obj_cnt_list = list(
          zip(
              self.image_files,
              [len(scn['objects']) for scn in self.image_annotations]
          )
      )
      obj_cnt_filter_index = [
          t[1] >= num_objects_min and t[1] <= num_objects_max \
          for t in obj_cnt_list
      ]
      # apply filter
      self.image_files = [self.image_files[idx] for idx, t in enumerate(obj_cnt_filter_index) if t]
      self.image_annotations = [self.image_annotations[idx] for idx, t in enumerate(obj_cnt_filter_index) if t]
      print(">>> Loaded %d images from split 'val'." % len(self.image_files))
    else:  # load test data only
      self.image_files.extend([
          os.path.join(self.root_dir, 'images', 'test', fn) \
          for fn in sorted(os.listdir(os.path.join(data_root, 'images', 'test')))
      ])
      self.image_files = self.image_files
      print(">>> Loaded %d images from split 'test'." % len(self.image_files))
    self.image_size = image_size
    self.seed_is_set = False
    self.N = len(self.image_files)
    self.seq_len = 1
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

  def __getitem__(self, index):
    # self.set_seed(index)
    # image_seq = []
    # ImageFile.LOAD_TRUNCATED_IMAGES = True
    image_file = self.image_files[index]
    try:
      im = (np.asarray(Image.open(image_file).convert('RGB')\
        .resize((self.image_size, self.image_size), PIL.Image.LANCZOS))\
        .reshape(self.image_size, self.image_size, 3))
    except (IOError, SyntaxError) as excp:
      print(excp)
      print(f">>> Corrupt image file '{image_file}'!")
      im = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
      # .astype('float32') \
      # - 127.5) / 255
    # image_seq.append(im)
    # image_seq = np.concatenate(image_seq, axis=0)
    # if self.seq_len == 1:
    #   image_seq = torch.from_numpy(image_seq[0,:,:,:]).permute(2,0,1)
    # return {'input' : self.transform(img)}
    return {'input' : self.transform(im)}

    # def __getitem__(self, idx):
    #     # --- Load image ---
    #     # File name example:
    #     # data_dir + /recordings/env_ccs-hard-h=2-vcom=0-vpsf=0-v=60/
    #     # rgb-w=5-f=2-l=1-c=unique-cam_7-mono-0.png
    #     file = self.filenames[idx]
    #     img = Image.open(file)
    #     output = {'input': self.transform(img)}

    #     # --- Load instances ---
    #     if self.load_instances:
    #         file_split = file.split('/')
    #         cam = file_split[4].split('-')[5][4:]
    #         map_path = os.path.join(
    #             self.data_dir, 'iseg', file_split[3],
    #             'iseg-w=0-f=0-l=0-c=original-cam_' + cam + '-mono-0.map')
    #         masks = load_segmap_as_matrix(map_path)
    #         masks = np.expand_dims(masks, 0)
    #         masks = np_img_centre_crop(masks, CENTRE_CROP)
    #         masks = torch.FloatTensor(masks)
    #         if self.img_size != masks.shape[2]:
    #             masks = masks.unsqueeze(0)
    #             masks = F.interpolate(masks, size=self.img_size)
    #             masks = masks.squeeze(0)
    #         output['instances'] = masks.type(torch.LongTensor)

    #     return output
