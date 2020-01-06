import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py

import os
import os.path

import cv2

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))

# load rgb frames from collection of images
def load_rgb_frames(image_dir, vid, start, num):
  frames = []
  vid_name = vid.split('/')[-1]
  for i in range(start, start+num):
    try:
        img = cv2.imread(os.path.join(image_dir, vid, vid_name+'_'+str(i).zfill(6)+'.jpg'))[:, :, [2, 1, 0]]
    except Exception:
        raise Exception('ERROR ' + os.path.join(image_dir, vid, vid_name+'_'+str(i).zfill(6)+'.jpg'))
        
    # img = cv2.imread(os.path.join(image_dir, vid, vid_name+'_'+str(i).zfill(6)+'.jpg'))[:, :, [2, 1, 0]]
    w,h,c = img.shape
    # resize to preserve aspect ratio so that biggest dimension is 226
    if w < 226 or h < 226:
        d = 226.-min(w,h)
        sc = 1+d/min(w,h)
        img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
    img = (img/255.)*2 - 1
    frames.append(img)
  return np.asarray(frames, dtype=np.float32)

# load flow frames from numpy files
def load_flow_frames(image_dir, vid, start, num):
  frames = []
  vid_name = vid.split('/')[-1]
  # TODO - change to grab appropriate flow numpy file
  for i in range(start, start+num):
    flo = np.load(os.path.join(image_dir, vid+'_flow', vid_name+'-'+str(i).zfill(6)+'_flow.npy'))
    
    w,h,c = flo.shape
    assert c == 2 # make sure there is an x and y channel
    if w < 224 or h < 224:
        d = 224.-min(w,h)
        sc = 1+d/min(w,h)
        flo = cv2.resize(flo,dsize=(0,0),fx=sc,fy=sc)
        
    # should already be normalized between -1 and 1
    # imgx = (imgx/255.)*2 - 1
    # imgy = (imgy/255.)*2 - 1
    # img = np.asarray([imgx, imgy]).transpose([1,2,0])
    flo = flo.transpose([1,2,0])
    frames.append(flo)
  return np.asarray(frames, dtype=np.float32)

# initialize Breakfast dataset
def make_dataset(split_file, split, root, mode):
    dataset = []
    # read list of video names in file
    # assumes that split files are different for training / testing
    with open(split_file, 'r') as f:
        video_list = f.read().split('\n')[0:-1]

    print("Making {} dataset with {} videos".format(split, len(video_list)))

    i = 0
    for vid in video_list:
        vid_path = os.path.join(root, vid)
        if not os.path.exists(vid_path):
            print("Cannot find video {}".format(vid_path))
            continue
        num_frames = len(os.listdir(vid_path))
        if mode == 'flow':
            flow_path = vid_path + "_flow"
            if not os.path.exists(flow_path):
                print("Cannot find flow directory {}".format(flow_path))
                continue
            if not num_frames == len(os.listdir(flow_path)):
                print("{} Flow frames ({}) does not match rgb frames ({})".format(vid_path, len(os.listdir(flow_path)), num_frames))
                continue
            
        # TODO - are labels going to be used? If so, need to load / parse them
        # label = np.zeros((num_classes,num_frames), np.float32)
        # # 
        # fps = num_frames/data[vid]['duration']
        # for ann in data[vid]['actions']:
        #     for fr in range(0,num_frames,1):
        #         if fr/fps > ann[1] and fr/fps < ann[2]:
        #             label[ann[0], fr] = 1 # binary classification

        dataset.append((vid, num_frames))
        i += 1
    
    return dataset


class Breakfast(data_utl.Dataset):

    def __init__(self, split_file, split, root, mode, transforms=None, save_dir='', num=0):
        
        self.data = make_dataset(split_file, split, root, mode)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.save_dir = save_dir

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, nf = self.data[index]
        # if os.path.exists(os.path.join(self.save_dir, vid+'.npy')):
        #     return 0, 0, vid

        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.root, vid, 0, nf)
        else:
            imgs = load_flow_frames(self.root, vid, 0, nf)

        imgs = self.transforms(imgs)

        # TODO - return labels when parsing is implemented
        return video_to_tensor(imgs), vid

    def __len__(self):
        return len(self.data)
