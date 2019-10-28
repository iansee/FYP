from syft.workers import WebsocketServerWorker
import torch
import sys
import syft
import sys
import argparse
from torchvision import datasets
from torchvision import transforms
import numpy as np


def main(number):
    mnist_dataset = datasets.MNIST(
        root="./files/",
        train=True,
        download=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    selected_data = mnist_dataset.data
    selected_targets = mnist_dataset.targets
    _id = 'h%s'%number
    ip  = '10.0.0.%s'%number

    hook = syft.TorchHook(torch)
    
    server = WebsocketServerWorker(id = _id,host =ip, port = 8778,hook=hook,verbose=True)
    
    print ("Worker:{}, Dataset contains {}".format(_id,str(len(selected_data))))
    dataset = syft.BaseDataset(
    data=selected_data, targets=selected_targets, transform=mnist_dataset.transform
        )
    key = "targeted"
    server.add_dataset(dataset, key=key)
    server.start()
    

main (sys.argv[1])

    
