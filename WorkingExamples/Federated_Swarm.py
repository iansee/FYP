import tensorflow as tf
import h5py
import collections
import six

from syft.workers.websocket_server import WebsocketServerWorker
import torch
import sys
import syft
import sys
import argparse
from torchvision import datasets
from torchvision import transforms
import numpy as np

import tensorflow_federated as tff
import os.path
from tensorflow_federated.python.simulation.hdf5_client_data import HDF5ClientData


class TrainDataset:
    def __init__(self,transform=None, number=2, slice_of_data):
        fileprefix = "fed_emnist_digitsonly"
        #dir_path = os.path.dirname("/home/mininet/")
        dir_path = os.getcwd()
        train = HDF5ClientData(os.path.join(dir_path, fileprefix + '_train.h5'))
        trainFile = h5py.File(os.path.join(dir_path, fileprefix + '_train.h5'), "r")
        _EXAMPLES_GROUP = "examples"
        numberofclients = len(train.client_ids)
        data = np.empty((0,28,28), np.float32)
        target = np.empty((0), np.int_)
        offset = int(number) - 1
        for i in range(int(numberofclients/20):
            for j in slice_of_data:
                clientdataset = collections.OrderedDict((name, ds[()]) for name, ds in sorted(
                    six.iteritems(trainFile[HDF5ClientData._EXAMPLES_GROUP][train.client_ids[i*20+j]])))
                data = np.concatenate((data, clientdataset['pixels']))
                target = np.concatenate((target, clientdataset['label']), axis=0)
        self.target = list(target)
        self.data = list(data)
        self.transform = transform
        trainFile.close()
        del train

    def __getitem__(self, index):
        x=self.data[index]
        y=self.target[index]
        if self.transform:
            x = self.transform(x)
        return x,y
    def __len__(self):
        return len(self.target)

def main(number, slice_of_data):
    mnist_dataset = TrainDataset(transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(
        (0.1307,), (0.3081,))
        ]), number=number, slice_of_data=slice_of_data)
    _id = 'h%s'%number
    ip  = '10.0.0.%s'%number

    hook = syft.TorchHook(torch)

    server = WebsocketServerWorker(id = _id,host =ip, port = 8778,hook=hook,verbose=True)
    print ("Worker:{}, Dataset contains {}".format(_id,str(len(mnist_dataset.data))))
    dataset = syft.BaseDataset(
    data=mnist_dataset.data, targets=mnist_dataset.target, transform=mnist_dataset.transform
        )
    key = "targeted"
    server.add_dataset(dataset, key=key)
    server.start()


main (sys.argv[1],sys.argv[2])