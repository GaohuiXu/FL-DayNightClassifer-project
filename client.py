import flwr as fl
import torch
from utils import *
from dataset import load_data

# ==========================================
# 1. Define the flower client 
# ==========================================
class DayNightClient(fl.client.NumPyClient):
    def __init__(self, net, train_loader, val_loader, device):
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
    
    def fit(self, parameters, config):
        # 1. receive the model parameters from server
        set_parameters(self.net, parameters)

        # 2. train on the local training dataset
        train(self.net, self.train_loader, epochs=1, device=self.device)

        # 3. extract the updated parameters and upload to server, together with num of local training samples
        return get_parameters(self.net), len(self.train_loader.dataset), {}
    
    def evaluate(self, parameters, config):
        # 1. receive the model parameters from server
        set_parameters(self.net, parameters)

        # 2. evaluate on the local validation dataset
        loss, accuracy = eval(self.net, self.val_loader, device=self.device)

        # 3. return the evaluation results
        return float(loss), len(self.val_loader.dataset), {"accuracy": float(accuracy)}
    
# ==========================================
# 2. Configure ClientApp 
# ==========================================
def client_fn(context: fl.common.Context):
    device = torch.device("cude: 0" if torch.cuda.is_available() else "cpu")

    # 1. push the instance to device
    net = SimpleCNN().to(device)

    # 2. load the custom dataset for current client
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    train_loader, val_loader = load_data(partition_id=partition_id, num_partitions=num_partitions)

    return DayNightClient(net, train_loader, val_loader, device).to_client()

app = fl.client.ClientApp(client_fn=client_fn)