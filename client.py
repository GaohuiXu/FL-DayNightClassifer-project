import flwr as fl
import torch
from utils import *
from dataset import load_data
import os

def l2_update_norm(old_params, new_params):
    import numpy as np
    sq = 0.0
    for a, b in zip(old_params, new_params):
        d = (b - a).astype("float64")
        sq += np.sum(d * d)
    return float(np.sqrt(sq))

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
    # Receive global parameters
        set_parameters(self.net, parameters)

        # Read config sent by server
        local_epochs = int(config.get("local_epochs", 1))
        lr = float(config.get("lr", 1e-3))
        batch_size = int(config.get("batch_size", 32))
        server_round = int(config.get("server-round", -1))

        # OPTIONAL: print once per actor for debugging
        # print(f"[fit] round={server_round} epochs={local_epochs} lr={lr} bs={batch_size}")

        # If you want batch_size to be dynamic, you need loaders rebuilt or DataLoader adjusted.
        # Minimal approach: ignore batch_size for now, or rebuild loaders when it changes.
        # We'll do the clean version next (change load_data signature).
        
        old_params = get_parameters(self.net)

        # Train locally (make sure your train() can accept lr; otherwise set optimizer inside train)
        train_loss = train(self.net, self.train_loader, epochs=local_epochs, device=self.device, lr=lr)

        new_params = get_parameters(self.net)
        update_norm = l2_update_norm(old_params, new_params)

        metrics = {}
        if train_loss is not None:
            metrics["train_loss"] = float(train_loss)
        metrics["update_norm"] = float(update_norm)

        return new_params, len(self.train_loader.dataset), metrics
    
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1. push the instance to device
    net = SimpleCNN().to(device)

    print("node_config keys:", list(context.node_config.keys()))
    print("node_config:", context.node_config)

    # 2. load the custom dataset for current client
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = int(context.run_config.get("batch_size", 32))
    train_loader, val_loader = load_data(partition_id, num_partitions, batch_size=batch_size)

    return DayNightClient(net, train_loader, val_loader, device).to_client()

app = fl.client.ClientApp(client_fn=client_fn)