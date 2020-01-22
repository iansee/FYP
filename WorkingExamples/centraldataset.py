import torch
import torch.nn as nn
import torch.nn.functional as F
import asyncio
import syft
from syft.workers.websocket_client import WebsocketClientWorker
from syft.federated.monitor import monitoring
from syft.frameworks.torch.federated import utils
from torchvision import datasets
from torchvision import transforms

import h5py
import collections
import six
import os.path
#from tensorflow_federated.python.simulation.hdf5_client_data import HDF5ClientData
from no_tff import HDF5ClientData
import numpy as np
import tensorflow as tf
import sys

global batchsize
global lr
global no_epoch
global federated_rounds


batchsize=1000
lr = 0.01
no_epoch = 1
federated_rounds = 10
fractionToUse = 5
target_accuracy = 80

class TestDataset:
    def __init__(self,transform=None):
        fileprefix = "fed_emnist_digitsonly"
        #dir_path = os.path.dirname("/home/mininet/")
        dir_path = os.getcwd()
        test = HDF5ClientData(os.path.join(dir_path, fileprefix + '_test.h5'))
        testFile = h5py.File(os.path.join(dir_path, fileprefix + '_test.h5'), "r")
        _EXAMPLES_GROUP = "examples"
        numberofclients = len(test.client_ids)
        data = np.empty((0,28,28), np.float32)
        target = np.empty((0), np.int_)
        for i in range(int(numberofclients/fractionToUse)):
            clientdataset = collections.OrderedDict((name, ds[()]) for name, ds in sorted(
                six.iteritems(testFile[HDF5ClientData._EXAMPLES_GROUP][test.client_ids[i*fractionToUse]])))
            # data = np.concatenate((data, np.reshape(clientdataset['pixels'], (-1, 1, 28, 28))))
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

class TrainDataset:
    def __init__(self,transform=None,dataset=None,target=None):
        self.target = target
        self.data = dataset
        self.transform = transform
        
    def __getitem__(self, index):
        x=self.data[index]
        y=self.target[index]
        if self.transform:
            x = self.transform(x)
        return x,y
    
    def __len__(self):
        return len(self.target)


#Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(lr,epoch,train_loader):
    net = Net()
    optimizer = torch.optim.SGD(net.parameters(),lr=lr)
    for i in range (0,epoch):
        current_epoch = i+1
        for batch_no,(data,target) in enumerate (train_loader):
            optimizer.zero_grad()
            output = net(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            print ("Epoch:" + str(current_epoch))
            print ("Batch" + str(batch_no))
            print ("Loss:" + str(loss.item()))
    return net

def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for (data, target) in test_loader:
          output = model(data)
          test_loss += F.nll_loss(output, target, size_average=False).item()
          pred = output.data.max(1, keepdim=True)[1]
          correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    accuracy = 100.*correct / len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),accuracy))
    return accuracy



async def get_performance(worker):
    network = await worker.perf_ping()
    data = worker.data_setamount()

    print (worker.id+" has network performance of ")
    for x in network:
        print (x + ':' + str(network[x]))
        
    print (worker.id+" has dataset of {}".format(data))

async def get_dataset(worker):
    worker_dataset = await worker.data_setget()
    print ('Worker {} has returned dataset'.format(worker.id))
    return worker_dataset

def connect_to_nodes(nodes):
    hook = syft.TorchHook(torch)
    workers = []
    for i in range(2,int(nodes)+1):
        ip = '10.0.0.{}'.format(str(i))
        socket = {"host": ip, "hook":hook,"verbose":True}

        worker_name = 'h{}'.format(str(i))
        print ('connected to {}'.format(worker_name))

        clientworker = WebsocketClientWorker(id=worker_name,port=8778,**socket)
        workers.append(clientworker)
    return workers


def loaddataset(datasetlist,targetlist):
    train_loader = torch.utils.data.DataLoader(TrainDataset(transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]),
                                                            dataset = datasetlist,target=targetlist),batch_size=batchsize,shuffle=True)
    return train_loader

async def getdata(nodes=10):

    workers = connect_to_nodes(nodes)
    
    print ('Performance measurments')
    performance = await asyncio.gather(
        *[get_performance(worker)
          for worker in workers])
    
    centralizedmonitor = monitoring()
    centralizedmonitor.start()

    
    data = await asyncio.gather(
        *[get_dataset(worker)
          for worker in workers])
    combineddataset = []
    combinedtargetset = []
    
    for dataset in data:
        if dataset is not None:
            current_data_list = dataset[0]
            current_target_list = dataset[1]

            print (len(current_data_list))
            print (len(current_target_list))

            combineddataset += current_data_list
            combinedtargetset += current_target_list

        '''
        for current_data in current_data_list:
            globaldata = np.concatenate((globaldata,current_data))
        for current_target in current_target_list:
            globaltarget = np.concatenate((globaltarget,current_target))
        '''
    print (len(combineddataset))
    print (len(combinedtargetset))

    
    for worker in workers:
        worker.close()

    networkcost=centralizedmonitor.stop()
    for x in networkcost:
        print (x + ':' + str(networkcost[x]))

    totalcost = centralizedmonitor.getnetworkcost()
    totalcost = totalcost / 1000000
    print ("The total network cost for the centralized model is:{} MB".format(totalcost))
    



def main():

    asyncio.get_event_loop().run_until_complete(getdata(10))
    


main()

