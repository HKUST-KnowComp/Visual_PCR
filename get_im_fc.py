import os
import glob
import numpy as np
import random
import time
import json
import os.path as osp
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from tqdm import tqdm
import h5py
from collections import OrderedDict
import argparse

parser = argparse.ArgumentParser(description='extract features of Resnet')

parser.add_argument('--image_dir', type=str, default='data/images',
                    help='image dir')
parser.add_argument('--resnet', type=int, default=152,
                    help='use resnet 101 or 152')
parser.add_argument('--append_hdf5', action='store_true',
                    help='append to existing hdf5, allow feature extraction after abortion')


args = parser.parse_args()

class myResnet(nn.Module):
    def __init__(self, resnet):
        super(myResnet, self).__init__()
        self.resnet = resnet

    def forward(self, img):
        x = img.unsqueeze(0)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)
        
        return x

# prepare resnet
if args.resnet == 101:
    resnet = models.resnet101(pretrained=True)
elif args.resnet == 152:
    resnet = models.resnet152(pretrained=True)
net = myResnet(resnet).cuda().eval()
trans = transforms.Compose([
    transforms.Resize((448,448)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# create hdf5
h5_filename = 'data/resnet{}_feat.hdf5'.format(args.resnet)
if args.append_hdf5:
    f = h5py.File(h5_filename, 'a')
    written_keys = f.keys()
else:
    f = h5py.File(h5_filename, 'w')

for split in ['train', 'val', 'test']:
    # load data
    data = [json.loads(line) for line in open('data/{}.vispro.1.1.jsonlines'.format(split))]

    # for each image
    for dialog in tqdm(data):
        filename = dialog['image_file']

        # skip images already extracted
        if args.append_hdf5 and 'dl:%s:%d' % (split, dialog_id) in written_keys:
            continue

        # extract feature and write to hdf5
        filename = osp.join(args.image_dir, filename)
        img = Image.open(filename)
        if len(np.array(img).shape) < 3:
            img = Image.merge('RGB', (img,) * 3)
        with torch.no_grad():
            feat = net(trans(img).cuda())
        feat = feat.squeeze(0).cpu().data.numpy()
        f.create_dataset(dialog['doc_key'], data=feat)

# save result
f.close()
print('Results saved to ' + h5_filename)
