
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

fractionToUse = 5

class TrainDataset:
    def __init__(self,transform=None, id="h2"):
        fileprefix = "fed_emnist_digitsonly"
        dir_path = os.path.dirname("/home/mininet/")
        train = HDF5ClientData(os.path.join(dir_path, fileprefix + '_train.h5'))
        trainFile = h5py.File(os.path.join(dir_path, fileprefix + '_train.h5'), "r")
        _EXAMPLES_GROUP = "examples"
        numberofclients = len(train.client_ids)
        data = np.empty((0,28,28), np.float32)
        target = np.empty((0), np.int_)
        if id == "h2":
            offset = 0
        elif id == "h3":
            offset = 1
        elif id == "h4":
            offset = 2
        for i in range(int(numberofclients/(3*fractionToUse))):
            clientdataset = collections.OrderedDict((name, ds[()]) for name, ds in sorted(
                six.iteritems(trainFile[HDF5ClientData._EXAMPLES_GROUP][train.client_ids[i*3*fractionToUse+offset]])))
            data = np.concatenate((data, clientdataset['pixels']))
            target = np.concatenate((target, clientdataset['label']), axis=0)
        self.target = list(target)
        self.data = list(data)
        self.transform = transform

    def __getitem__(self, index):
        x=self.data[index]
        y=self.target[index]
        if self.transform:
            x = self.transform(x)
        return x,y
    def __len__(self):
        return len(self.target)

def main(_id,ip):
    mnist_dataset = TrainDataset(transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(
        (0.1307,), (0.3081,))
        ]), id=_id)

    hook = syft.TorchHook(torch)
    server = WebsocketServerWorker(id = _id,host =ip, port = 8778,hook=hook,verbose=True)
    print ("Worker:{}, Dataset contains {}".format(_id,str(len(mnist_dataset.data))))
    dataset = syft.BaseDataset(
    data=mnist_dataset.data, targets=mnist_dataset.target, transform=mnist_dataset.transform
        )
    key = "targeted"
    server.add_dataset(dataset, key=key)
    server.start()


main (sys.argv[1], sys.argv[2])
