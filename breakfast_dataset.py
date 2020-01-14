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

def resize(img, size=DEF_IMG_SIZE):
    w,h,c = img.shape
    if w != size or h != size:
        sc = float(size) / float(max(w, h))
        return cv2.resize(img, dsize=(size, size), fx=sc, fy=sc)
    return img
    

def extract_flow(parent_path, video_name):
    print("Extracting flow for {}".format(video_name))
    video_path = os.path.join(parent_path, video_name)
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
    flow_frames = np.array(flow_frames)
    print("{} flow frames shape: {}".format(video_name, flow_frames.shape))
    return flow_frames


# load flow frames from numpy files
def load_flow_frames(image_dir, vid, start, num):
  frames = []
  vid_name = vid.split('/')[-1]
  # TODO - change to grab appropriate flow numpy file
  for i in range(start, start+num):
    flo = np.load(os.path.join(image_dir, vid+'_flow', vid_name+'_'+str(i).zfill(6)+'_flow.npy'))
    
    h,w,c = flo.shape
    print("Flow shape {},{},{}".format(w,h,c))
    assert c == 2 # make sure there is an x and y channel
    if w < 224 or h < 224:
        d = 224.-min(w,h)
        sc = 1+d/min(w,h)
        flo = cv2.resize(flo,dsize=(0,0),fx=sc,fy=sc)
        
    flo = flo.transpose([2,1,0])
    frames.append(flo)
  return np.asarray(frames, dtype=np.float32)


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

        # if mode == 'flow':
        #     flow_path = vid_path + "_flow"
        #     if not os.path.exists(flow_path):
        #         print("Cannot find flow directory {}".format(flow_path))
        #         continue
        #     num_frames = len([f for f in os.listdir(flow_path) if not f[0] == '.'])
        
        # else:
        num_frames = get_frame_count(vid_path + '.avi')

            # if not num_frames == len(os.listdir(flow_path)):
            #     print("{} Flow frames ({}) does not match rgb frames ({})".format(vid_path, len(os.listdir(flow_path)), num_frames))
            #     continue
            
        # TODO - are labels going to be used? If so, need to load / parse them

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
            # imgs = load_flow_frames(self.root, vid, 0, nf)
            imgs = extract_flow(self.root, vid + '.avi')

        print("{} data shape: {}".format(vid, imgs.shape))
        imgs = self.transforms(imgs)

        # TODO - return labels when parsing is implemented
        return video_to_tensor(imgs), vid

    def __len__(self):
        return len(self.data)
