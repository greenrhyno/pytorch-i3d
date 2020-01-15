import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import sys
import argparse

ENDPOINT='Mixed_5c'
DEF_MODE = 'flow'
DEF_FRAME_WINDOW = 65 # should be odd
DEF_ROOT = '/mnt/raptor/ryan/breakfast_data'
VIDEO_LIST = '/home/ryan/breakfast_splits/single_video.test'
DEF_SAVE_DIR = '/mnt/raptor/ryan/breakfast_i3d/' + ENDPOINT
DEF_MODEL_SAVE_DIR = '/mnt/raptor/ryan/pytorch-i3d/saved_models'
SAVE_POSTFIX = '_i3d_' 

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow', default=DEF_MODE)
parser.add_argument('-window', type=int, default=DEF_FRAME_WINDOW)
parser.add_argument('-load_model', type=str, default='')
parser.add_argument('-root', type=str, default=DEF_ROOT)
parser.add_argument('-video_list', type=str, default=VIDEO_LIST)
parser.add_argument('-gpu', type=str, default="0")
parser.add_argument('-save_dir', type=str, default=DEF_SAVE_DIR)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms

import numpy as np
from pytorch_i3d import InceptionI3d
from breakfast_dataset import Breakfast as Dataset

def run(max_steps=64e3, window=DEF_FRAME_WINDOW, mode=DEF_MODE, root=DEF_ROOT, video_list=VIDEO_LIST, 
    batch_size=1, load_model='', save_dir=DEF_SAVE_DIR):

    print("\n*******************\nExtracting Breakfast i3d Features\n")
    print("Mode: {}".format(mode))
    print("Root Data Directory: {}".format(root))
    print("Video List: {}".format(video_list))
    print("Save Directory: {}\n".format(save_dir))
    
    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    n_temporal_pad = int((window - 1) / 2) 

    dataset = Dataset(video_list, 'testing', root, mode, test_transforms, num=-1, save_dir=save_dir, pad=n_temporal_pad)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)  
    
    print(dataset.data)

    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2, final_endpoint=ENDPOINT)
    else:
        i3d = InceptionI3d(400, in_channels=3, final_endpoint=ENDPOINT)

    # i3d.load_state_dict(torch.load(load_model))
    i3d.cuda()

    i3d.train(False)  # Set model to evaluate mode

    print(len(dataloader))
                
    # Iterate over data.
    for data in dataloader:
        # get the inputs
        inputs, name, n_frames = data
        name = name[0]
        save_file_name = name + SAVE_POSTFIX + '_' + mode
        # if features exist, skip video
        # if os.path.exists(os.path.join(save_dir, save_file_name+'.npy')): #TODO - add back in after testing
        #     continue

        b,c,t,h,w = inputs.shape
        print("{} shape: {}".format(name, inputs.shape))
        assert(t == n_frames + n_temporal_pad * 2)

        features = []
        for frame_idx in range(n_frames):
            ip = Variable(torch.from_numpy(inputs.numpy()[:,:,frame_idx:frame_idx+window]).cuda())
            features.append(i3d.extract_features(ip).squeeze(0).permute(1,2,3,0).data.cpu().numpy())
        final_features = np.concatenate(features, axis=0)
        np.save(os.path.join(save_dir, save_file_name), final_features)
        print("{} features saved (shape: {})".format(save_file_name, final_features.shape))


if __name__ == '__main__':
    run(mode=args.mode, window=args.window, root=args.root, load_model=args.load_model, 
    video_list=args.video_list, save_dir=args.save_dir)
