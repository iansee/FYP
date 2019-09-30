import tensorflow as tf
import torch
from torch import nn
from torch import optim
import syft 
from syft.workers import WebsocketClientWorker
import time

@torch.jit.script
def loss_fn(target, pred):
    return ((target.view(pred.shape).float() - pred.float()) ** 2).mean()

hook = syft.TorchHook(torch)

model = nn.Linear(1,1)
mock_data = torch.zeros(1)
traced_model  = torch.jit.trace(model,mock_data)


optimizer = "SGD"

batch_size = 4
optimizer_args = {"lr" : 0.1, "weight_decay" : 0.01}
max_nr_batches = -1  # not used in this example
shuffle = True


train_config = syft.TrainConfig(model=traced_model,
                              loss_fn=loss_fn,
                              optimizer=optimizer,
                              batch_size=batch_size,
                              optimizer_args=optimizer_args,
                              epochs=5,
                              shuffle=shuffle)

arw = {"host":"10.0.0.1","hook":hook}
h1 = WebsocketClientWorker(id="h1",port=8778,**arw)
train_config.send(h1)


message = h1.create_message_execute_command(command_name="start_monitoring",command_owner="self")
serialized_message = syft.serde.serialize(message)
h1._recv_msg(serialized_message)


time.sleep(3)
for epoch in range(10):
    time.sleep(3)
    loss = h1.fit(dataset_key="train")  # ask alice to train using "xor" dataset
    print("-" * 50)
    print("Iteration %s: h1's loss: %s" % (epoch, loss))

message = h1.create_message_execute_command(command_name="stop_monitoring",command_owner="self")
serialized_message = syft.serde.serialize(message)
h1._recv_msg(serialized_message)

new_model = train_config.model_ptr.get()
h1.close()

