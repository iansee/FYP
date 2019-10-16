from syft.workers import WebsocketServerWorker
import torch
import sys
import syft
import sys
import argparse
from torchvision import datasets
from torchvision import transforms
import numpy as np


def main(_id,ip):
    mnist_dataset = datasets.MNIST(
        root="./files/",
        train=True,
        download=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    labels = {
        "h2":[0,1,2,3],
        "h3":[4,5,6],
        "h4":[7,8,9]
        }
    hook = syft.TorchHook(torch)
    server = WebsocketServerWorker(id = _id,host =ip, port = 8778,hook=hook,verbose=True)
    keep_labels=labels[_id]
    indices = np.isin(mnist_dataset.targets, keep_labels).astype("uint8")
    selected_data = (
            torch.native_masked_select(mnist_dataset.data.transpose(0, 2), torch.tensor(indices))
            .view(28, 28, -1)
            .transpose(2, 0)
        )
    print ("Worker:{} with labels: {}, Dataset contains {}".format(_id,str(keep_labels),str(len(selected_data))))
    selected_targets = torch.native_masked_select(mnist_dataset.targets, torch.tensor(indices))
    dataset = syft.BaseDataset(
    data=selected_data, targets=selected_targets, transform=mnist_dataset.transform
        )
    key = "targeted"
    server.add_dataset(dataset, key=key)
    server.start()
    

main (sys.argv[1], sys.argv[2])

    
