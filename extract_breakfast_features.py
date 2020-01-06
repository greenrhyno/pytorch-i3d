import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import sys
import argparse

N_ACTION_CLASSES = 48
DEF_MODE = 'rgb'
DEF_ROOT = '/home/ryangreen/projects/BreakfastII_15fps_qvga_sync'
DEF_TRAIN_SPLIT = '/home/ryangreen/projects/breakfast_splits/split_test.train'
DEF_TEST_SPLIT = '/home/ryangreen/projects/breakfast_splits/split_test.test'
DEF_SAVE_DIR = '/home/ryangreen/Futureman/BreakfastII_15fps_qvga_sync/Breakfast_i3d'
DEF_MODEL_SAVE_DIR = '/home/ryangreen/projects/i3d_models'
SAVE_POSTFIX = '_i3d_400'

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow', default=DEF_MODE)
parser.add_argument('-load_model', type=str, default='')
parser.add_argument('-root', type=str, default=DEF_ROOT)
parser.add_argument('-gpu', type=str, default="0")
parser.add_argument('-save_dir', type=str, default=DEF_MODEL_SAVE_DIR)

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
# from breakfast_consts import BreakfastConsts as Consts


def run(max_steps=64e3, mode=DEF_MODE, root=DEF_ROOT, 
    train_split=DEF_TRAIN_SPLIT, test_split=DEF_TEST_SPLIT, 
    batch_size=1, load_model='', save_dir=DEF_SAVE_DIR):

    print("\n*******************\nExtracting Breakfast i3d Features\n")
    print("Mode: {}".format(mode))
    print("Root Data Directory: {}".format(root))
    print("Train Split: {}".format(train_split))
    print("Test Split: {}".format(test_split))
    print("Save Directory: {}\n".format(save_dir))
    
    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(train_split, 'training', root, mode, test_transforms, num=-1, save_dir=save_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    val_dataset = Dataset(test_split, 'testing', root, mode, test_transforms, num=-1, save_dir=save_dir)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)    

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    # datasets = {'train': dataset, 'val': val_dataset}

    
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)
    # i3d.replace_logits(N_ACTION_CLASSES) 
    # i3d.load_state_dict(torch.load(load_model))
    i3d.cuda()

    for phase in ['train', 'val']:
        i3d.train(False)  # Set model to evaluate mode
                    
        # Iterate over data.
        for data in dataloaders[phase]:
            # get the inputs
            inputs, name = data
            name = name[0]
            print(name, SAVE_POSTFIX)
            save_file_name = name + SAVE_POSTFIX
            if os.path.exists(os.path.join(save_dir, save_file_name+'.npy')):
                continue

            b,c,t,h,w = inputs.shape
            if t > 1600:
                features = []
                for start in range(1, t-56, 1600):
                    end = min(t-1, start+1600+56)
                    start = max(1, start-48)
                    ip = Variable(torch.from_numpy(inputs.numpy()[:,:,start:end]).cuda(), volatile=True)
                    features.append(i3d.extract_features(ip).squeeze(0).permute(1,2,3,0).data.cpu().numpy())
                np.save(os.path.join(save_dir, save_file_name), np.concatenate(features, axis=0))
            else:
                # wrap them in Variable
                inputs = Variable(inputs.cuda(), volatile=True)
                features = i3d.extract_features(inputs)
                np.save(os.path.join(save_dir, save_file_name), features.squeeze(0).permute(1,2,3,0).data.cpu().numpy())


if __name__ == '__main__':
    run(mode=args.mode, root=args.root, load_model=args.load_model, save_dir=args.save_dir)
