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
from no_tff import HDF5ClientData
import numpy as np
import tensorflow as tf
import sys

global batchsize
global lr
global no_epoch
global no_federated_epochs
global federated_rounds


import csv

batchsize=1000
lr = 0.01
#no_epoch = 50
no_federated_epochs = 1
max_federated_rounds = 50
#fractionToUse = 5
target_accuracy = 100
a_nom = 0.5
b_nom = 1


class TestDataset:
    def __init__(self,transform=None):
        fileprefix = "fed_emnist_digitsonly"
        dir_path = os.getcwd()
        test = HDF5ClientData(os.path.join(dir_path, fileprefix + '_test.h5'))
        testFile = h5py.File(os.path.join(dir_path, fileprefix + '_test.h5'), "r")
        _EXAMPLES_GROUP = "examples"
        numberofclients = len(test.client_ids)
        data = np.empty((0,28,28), np.float32)
        target = np.empty((0), np.int_)
        for i in range(int(numberofclients)):
            clientdataset = collections.OrderedDict((name, ds[()]) for name, ds in sorted(
                six.iteritems(testFile[HDF5ClientData._EXAMPLES_GROUP][test.client_ids[i]])))
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
    



#Training dataset and loader
'''
class TrainDataset:
    def __init__(self,transform=None):
        fileprefix = "fed_emnist_digitsonly"
        dir_path = os.getcwd()
        train = HDF5ClientData(os.path.join(dir_path, fileprefix + '_train.h5'))
        trainFile = h5py.File(os.path.join(dir_path, fileprefix + '_train.h5'), "r")
        _EXAMPLES_GROUP = "examples"
        numberofclients = len(train.client_ids)
        data = np.empty((0,28,28), np.float32)
        target = np.empty((0), np.int_)
        for i in range(int(numberofclients/fractionToUse)):
            clientdataset = collections.OrderedDict((name, ds[()]) for name, ds in sorted(
                six.iteritems(trainFile[HDF5ClientData._EXAMPLES_GROUP][train.client_ids[i*fractionToUse]])))
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

        
train_loader = torch.utils.data.DataLoader(TrainDataset(transform=transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(
    (0.1307,), (0.3081,))
])),batch_size=batchsize,shuffle=True)
'''


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
'''
def train(lr,epoch):
    net = Net()
    optimizer = torch.optim.SGD(net.parameters(),lr=lr)
    loss_data = []
    acc_data = []
    for i in range (0,epoch):
        current_epoch = i+1
        print ("Epoch:" + str(current_epoch))
        for batch_no,(data,target) in enumerate (train_loader):
            optimizer.zero_grad()
            output = net(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            # print ("Epoch:" + str(current_epoch))
            # print ("Batch" + str(batch_no))
            # print ("Loss:" + str(loss.item()))
        test_acc, test_loss = test(net)
        loss_data.append(test_loss)
        acc_data.append(test_acc)
    with open('results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(loss_data)
        writer.writerow(acc_data)
    return net
'''

test_loader = torch.utils.data.DataLoader(TestDataset(transform=transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(
    (0.1307,), (0.3081,))
])),batch_size=batchsize,shuffle=True)

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
    return accuracy, test_loss

@torch.jit.script
def loss_fn(pred, target):
    return F.nll_loss(input=pred, target=target)

async def fit_model_on_worker(
    worker: WebsocketClientWorker,
    traced_model: torch.jit.ScriptModule,
    batch_size: int,
    curr_round: int,
    lr: float,
    no_federated_epochs: int
):
    train_config = syft.TrainConfig(
        model=traced_model,
        loss_fn=loss_fn,
        batch_size=batch_size,
        shuffle=True,
        epochs=no_federated_epochs,
        optimizer="SGD",
        optimizer_args={"lr": lr},
    )

    #send the training config
    train_config.send(worker)
    #Call async fit on worker - async fit calls the method calls self fit method
    print("Training round {}, calling fit on worker: {}".format(curr_round, worker.id))
    loss = await worker.async_fit(dataset_key="targeted", return_ids=[0])
    print("Training round: {}, worker: {}, avg_loss: {}".format( curr_round, worker.id, loss.item()))
    #Call back to the model
    model = train_config.get_model().obj

    return worker.id, model, loss

async def get_performance(worker):
    network = await worker.perf_ping()

    data = worker.data_setamount()

    print (worker.id+" has network performance of ")
    total_bytes = 0
    for x in network:
        print (x + ':' + str(network[x]))
        total_bytes = total_bytes + network[x]
    print (worker.id+" has dataset of {}".format(data))
    return (total_bytes/4000000, data)


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


#Changes made to SYFT implementation of websocket_client to always keep fit alive
async def sendmodel(nodes=10):
    totalnetworkcost = 0
    
    workers = connect_to_nodes(nodes)
    (mock_data, target) = test_loader.__iter__().next()
    model = Net()
    global traced_model

    traced_model = torch.jit.trace(model,mock_data)

    print ('Performance measurments')
    performance = await asyncio.gather(
        *[get_performance(worker)
          for worker in workers])
    cost_dict = { "h"+str(i+2) : performance[i][0] for i in range(0, len(performance) ) }
    utility_dict = { "h"+str(i+2) : performance[i][1] for i in range(0, len(performance) ) }
    training_count_dict = { "h"+str(i+2) : 0 for i in range(0, len(performance)) }

    #Write to CSV initially
    to_file = ["Round","Accuracy","Loss","Workers called","Total networkcost"]

    with open('federated_results.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(to_file)
    
    
    for current_round in range(max_federated_rounds):
        
        
        centralizedmonitor = monitoring()
        centralizedmonitor.start()
        
        print ("Starting round" + str(current_round))
        chosen_workers = choose_worker(cost_dict, utility_dict, training_count_dict)
        chosen_worker_string = ""
        for w in chosen_workers:
            training_count_dict[w] += 1
            chosen_worker_string = chosen_worker_string + w + " "
        results = await asyncio.gather(
            *[
                fit_model_on_worker(worker = worker,
                                         traced_model = traced_model,
                                         batch_size = batchsize,
                                         curr_round = current_round,
                                         lr = lr,
                                         no_federated_epochs = no_federated_epochs)
                for worker in filter(lambda w : w.id in chosen_workers, workers)]
            )
        models = {}
        loss_vals = {}
        network_dictonary = {}

        centralizedmonitor.stop()
        costofround = centralizedmonitor.getnetworkcost()
        costofround = costofround / 1000000
        totalnetworkcost += costofround
        
        
        for worker_id, worker_model, worker_loss in results:
            if worker_model is not None:
                models[worker_id] = worker_model

        avg_model = utils.federated_avg(models)
        traced_model = avg_model
        print ("Evaluating averaged model")
        accuracy, loss = test(traced_model)
        
        string_accuracy = ('{:.2f}'.format(accuracy))
        string_loss = str(loss)
        string_round = str(current_round + 1)
        
        to_file = [string_round,string_accuracy,string_loss,chosen_worker_string,totalnetworkcost]

        with open('federated_results.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(to_file)

        if accuracy > target_accuracy:
            print("Target accuracy has been reached. Terminating training.")
            break;


    print ("Finished Federated training - closing connections")

    for worker in workers:
        worker.close()

def choose_worker(cost_dict, utility_dict, training_count_dict):
    efficiency = { host : (utility_dict[host]/(a_nom*training_count_dict[host]+b_nom))/cost_dict[host] for host in utility_dict}
    return [w[0] for w in sorted(efficiency.items(), key=lambda h: h[1], reverse=True)[:3]]

def main():
    #centralized_model = train(lr,no_epoch)
    #accuracy = test(centralized_model)
    asyncio.get_event_loop().run_until_complete(sendmodel(nodes=10))




main()
