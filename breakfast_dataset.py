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

DEF_IMG_SIZE = 226

FLOW_POSTSCRIPT = '_opflow'
VIDEO_FORMAT = '.avi'

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
# TODO - REWRITE to grab frames from cv2 VideoCapture
def load_rgb_frames(parent_path, video_name):
    # compute optical rgb features
    print("Extracting rgb for {}".format(video_name))
    video_path = os.path.join(parent_path, video_name + VIDEO_FORMAT)
    cap = cv2.VideoCapture(video_path)
    ret, frame1 = cap.read()

    if not ret:
        print('Could not read {}'.format(video_path))
        return

    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    print("Video shape {}".format(prvs.shape))
    hsv = np.zeros_like(frame1)

    count = 0
    frame2 = frame1
    flow_frames = []

    while(ret):
        # convert to grayscale
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        # compute flow
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow = np.clip(flow, -20, 20) / 20
        flow = resize(flow)
        flow_frames.append(flow)

        prvs = next
        ret, frame2 = cap.read()
        count += 1

    cap.release()
    flow_frames = np.asarray(flow_frames, dtype=np.float32)
    if save:
        np.save(flow_filename, flow_frames) # save optical flow features
    print("{} flow frames shape: {}".format(video_name, flow_frames.shape))
    return flow_frames

def resize(img, size=DEF_IMG_SIZE):
    w,h,c = img.shape
    if w != size or h != size:
        sc = float(size) / float(max(w, h))
        return cv2.resize(img, dsize=(size, size), fx=sc, fy=sc)
    return img
    

def extract_flow(parent_path, video_name, save=False):
    # check for precomputed features
    flow_filename = os.path.join(parent_path, video_name + FLOW_POSTSCRIPT + '.npy')
    if os.path.exists(flow_filename):
        print("Found flow for {}".format(video_name))
        return np.load(flow_filename)

    # compute optical flow features
    print("Extracting flow for {}".format(video_name))
    video_path = os.path.join(parent_path, video_name + VIDEO_FORMAT)
    cap = cv2.VideoCapture(video_path)
    ret, frame1 = cap.read()

    if not ret:
        print('Could not read {}'.format(video_path))
        return

    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    print("Video shape {}".format(prvs.shape))
    hsv = np.zeros_like(frame1)

    count = 0
    frame2 = frame1
    flow_frames = []

    while(ret):
        # convert to grayscale
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        # compute flow
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow = np.clip(flow, -20, 20) / 20
        flow = resize(flow)
        flow_frames.append(flow)

        prvs = next
        ret, frame2 = cap.read()
        count += 1

    cap.release()
    flow_frames = np.asarray(flow_frames, dtype=np.float32)
    if save:
        np.save(flow_filename, flow_frames) # save optical flow features
    print("{} flow frames shape: {}".format(video_name, flow_frames.shape))
    return flow_frames


def get_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


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
        if not os.path.exists(vid_path + '.avi'):
            print("Cannot find video {}".format(vid_path))
            continue
        
        # num_frames = get_frame_count(vid_path + '.avi')

        # TODO - check if features have already been extracted
        # TODO - are labels going to be used? If so, need to load / parse them

        dataset.append(vid)
        i += 1
    
    return dataset


class Breakfast(data_utl.Dataset):

    def __init__(self, split_file, split, root, mode, transforms=None, save_dir='', num=0, pad=None):
        
        self.data = make_dataset(split_file, split, root, mode)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.save_dir = save_dir
        self.pad = pad

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid = self.data[index]

        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.root, vid)
        else:
            imgs = extract_flow(self.root, vid, save=False)

        n_frames = imgs.shape[0]

        if self.pad:
            if self.mode == 'rgb':
                # pad array with edge values
                imgs = np.concatenate((np.tile(imgs[0], (self.pad,1,1)), imgs, np.tile(imgs[-1], (self.pad,1,1))), axis=0) 
            elif self.mode == 'flow':
                sh = imgs.shape[1:]
                pad = np.zeros((self.pad,) + sh, dtype=np.float32)
                imgs = np.concatenate((pad, imgs, pad), axis=0)

        print("{} data shape: {}".format(vid, imgs.shape))

        # TODO - return labels when parsing is implemented
        return video_to_tensor(imgs), vid, n_frames

    def __len__(self):
        return len(self.data)
