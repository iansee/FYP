import tensorflow as tf
import torch
import syft 
from syft.workers import WebsocketServerWorker
from syft.generic.metrics import NetworkMonitor


def main():
    hook = syft.TorchHook(torch)
    data = torch.tensor([[1.0],[2.0],[3.0],[4.0]], requires_grad = True)
    target = torch.tensor([[1.0], [2.0], [3.0], [4.0]], requires_grad = False)
    dataset = syft.BaseDataset(data,target)
    
    h1 = WebsocketServerWorker(id="h1",host="10.0.0.1",port="8778",hook=hook)
    h1.add_dataset(dataset, key ="train")
    h1.start()
    return h1

if __name__ == "__main__":
    main()

