import torch
import torch.nn as nn
import torch.nn.functional as F
import asyncio
import syft
from syft.workers import WebsocketClientWorker
from syft.frameworks.torch.federated import utils
from torchvision import datasets
from torchvision import transforms

global batchsize
global lr
global no_epoch
global federated_rounds

batchsize=1000
lr = 0.01
no_epoch = 1
federated_rounds = 10

train_loader = torch.utils.data.DataLoader(
  datasets.MNIST('./files/', train=True, download=False,
                             transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batchsize, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  datasets.MNIST('./files/', train=False, download=False,
                             transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batchsize, shuffle=True)


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

def train(lr,epoch):
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
    #send monitoring command
    message = worker.create_message_execute_command(command_name="start_monitoring",command_owner="self")
    serialized_message = syft.serde.serialize(message)
    worker._recv_msg(serialized_message)
    #send the training config
    train_config.send(worker)
    #Call async fit on worker - async fit calls the method calls self fit method
    print("Training round {}, calling fit on worker: {}".format(curr_round, worker.id))
    loss = await worker.async_fit(dataset_key="targeted", return_ids=[0])
    print("Training round: {}, worker: {}, avg_loss: {}".format( curr_round, worker.id, loss.item()))
    #Call back to the model
    model = train_config.model_ptr.get().obj
    #Stop monitoring command
    message = worker.create_message_execute_command(command_name="stop_monitoring",command_owner="self")
    serialized_message = syft.serde.serialize(message)
    network_info = worker._recv_msg(serialized_message)
    #Deserialize the response recieved
    network_info = syft.serde.deserialize(network_info)
    return worker.id, model, loss, network_info

#Changes made to SYFT implementation of websocket_client to always keep fit alive
async def sendmodel():
    hook = syft.TorchHook(torch)
    h2_websocket = {"host": "10.0.0.2", "hook": hook, "verbose": True}
    h3_websocket = {"host": "10.0.0.3", "hook": hook, "verbose": True}
    h4_websocket = {"host": "10.0.0.4", "hook": hook, "verbose": True}

    h2 = WebsocketClientWorker(id = "h2",port = 8778,**h2_websocket)
    h3 = WebsocketClientWorker(id = "h3",port = 8778,**h3_websocket)
    h4 = WebsocketClientWorker(id = "h4",port = 8778,**h4_websocket)

    workers = [h2,h3,h4]
    (mock_data, target) = test_loader.__iter__().next()
    model = Net()
    global traced_model

    traced_model = torch.jit.trace(model,mock_data)
    
    for current_round in range(federated_rounds):
        print ("Starting round" + str(current_round))
        results = await asyncio.gather(
            *[
                fit_model_on_worker(worker = worker,
                                         traced_model = traced_model,
                                         batch_size = batchsize,
                                         curr_round = current_round,
                                         lr = lr,
                                         no_federated_epochs = no_epoch)
                for worker in workers]
            )
        models = {}
        loss_vals = {}
        network_dictonary = {}
        for worker_id, worker_model, worker_loss, network in results:
            if worker_model is not None:
                models[worker_id] = worker_model
                print ("Evaluating WORKER {}".format(worker_id))
                test(worker_model)
                for x in network:
                    print (x + ':' + str(network[x]))

                
        avg_model = utils.federated_avg(models)
        traced_model = avg_model
        print ("Evaluating averaged model")
        test(traced_model)
        
    print ("Finished Federated training - closing connections")
    for worker in workers:
        worker.close()


def main():
    #centralized_model = train(lr,no_epoch)
    #accuracy = test(centralized_model)
    asyncio.get_event_loop().run_until_complete(sendmodel())
    



main()

