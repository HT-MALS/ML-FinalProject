from torch.nn import Module
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Module
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import sys
import zipfile
import csv
from collections import OrderedDict
from PIL import Image

extracting = zipfile.ZipFile('test.zip')
extracting.extractall()
test_INFO = pd.read_csv("sampleSubmission1.csv")
test_DATA= 'test'

class FeatureBlock(nn.Sequential):
    def __init__(self, input_channels, output_channels):
        super(FeatureBlock, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        # add_module:在现有model中增添子module
        self.add_module('conv0', nn.Conv3d(input_channels, output_channels, kernel_size=3, stride=1, padding=0, bias=False)),
        self.add_module('norm0', nn.BatchNorm3d(output_channels)),
        self.add_module('relu0', nn.ReLU(inplace=True))

class Bottleneck(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.add_module('conv', nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False))
        self.add_module('norm', nn.BatchNorm3d(num_features=out_channels))
        self.add_module('relu', nn.ReLU(inplace=True))

class DenseLayer(nn.Sequential):
    def __init__(self, input_channels, growth_rate, bn_size, bottleneck_ratio, drop_rate):    # bn_size need to be 4
        super(DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(input_channels)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(input_channels, bn_size*growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),

        if bottleneck_ratio >0:
          self.add_module('bottleneck', Bottleneck(bn_size * growth_rate, bottleneck_ratio * growth_rate))
          in_channels = bottleneck_ratio * growth_rate
        else:
          in_channels = bn_size * growth_rate

        self.add_module('conv2', nn.Conv3d(in_channels, growth_rate,kernel_size=3, stride=1, padding=1, bias=False)),

        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class DenseBlock(nn.Sequential):
    def __init__(self, num_layers, input_channels, bn_size, growth_rate, bottleneck_ratio, drop_rate):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(input_channels + i * growth_rate, growth_rate, bn_size, bottleneck_ratio, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class Transition(nn.Sequential):
    def __init__(self, input_channels, output_channels):
        super(Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(input_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(input_channels, output_channels, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))

class Flatten(Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Classification(nn.Sequential):

    def __init__(self, input_channels, output_classes):
        super(Classification, self).__init__()

        self.input_channels = input_channels
        self.output_classes = output_classes

        self.add_module('norm', nn.BatchNorm3d(num_features=input_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('pool', nn.AvgPool3d(kernel_size=7, stride=1))
        self.add_module('flatten', Flatten())
        self.add_module('linear', nn.Linear(input_channels, output_classes))


class DenseNet(nn.Module):
    def __init__(self, growth_rate=16, block_config=(4, 4, 4),
                 num_init_features=32, bn_size=4, bottleneck_ratio=0, drop_rate=0, num_classes=2):
        super(DenseNet, self).__init__()

        self.features = FeatureBlock(input_channels=1,output_channels=num_init_features)


        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers=num_layers, input_channels=num_features,
                       bn_size=bn_size, growth_rate=growth_rate, bottleneck_ratio=bottleneck_ratio, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = Transition(input_channels=num_features, output_channels=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.classification = Classification(input_channels=num_features,output_classes=num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = self.classification(features)
        out = F.softmax(out, dim=1)
        return out
    
    
    

class testDataset(Dataset):
    def __init__(self,test_INFO,test_DATA,cropsize=32, transform=None):
        index = []
        index += list(test_INFO.index)
        self.index = tuple(sorted(index))
        self.transform = transform
        self.cropsize = cropsize

    def __getitem__(self, item):
        name = test_INFO.loc[self.index[item], 'name']
        data = np.load(os.path.join(test_DATA, '%s.npz' % name))
        voxel_temp = 2*data['voxel']/255-1
        voxel_temp = voxel_temp[50 - self.cropsize // 2:50 + self.cropsize // 2,
                     50 - self.cropsize // 2:50 + self.cropsize // 2,
                     50 - self.cropsize // 2:50 + self.cropsize // 2]
        voxel = np.expand_dims(voxel, axis=0)
        return name, voxel
    
    def __len__(self):
        return len(self.index)

class ToTensor(object):
    def __call__(self, sample):

        return torch.from_numpy(sample)

def main():
    print(os.getcwd()) #获取当前工作目录路径

    test_dataset = testDataset(test_INFO,test_DATA,cropsize=32, transform=ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,drop_last=False)

    File = open("Submission.csv", "w", newline='')
    csv_file = csv.writer(File)
    csv_file.writerow(["Id", "Predicted"])
    model = DenseNet()
    model.load_state_dict(torch.load('Model.pkl'))
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    with torch.no_grad():
        for ID,image in test_loader:
            image = image.type(torch.FloatTensor)
            image = image.to(device)
            outputs = model(image)
            outputs=torch.softmax(outputs,1)
            csv_file.writerow([str(ID[0]), (outputs.data[0][1]).item()])
    File.close()

if __name__ == '__main__':
   main()


