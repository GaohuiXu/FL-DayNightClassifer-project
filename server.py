import flwr as fl
from utils import *

# ==========================================
# 1. overall accuracy
# ==========================================
def weighted_average(metrics):
    """
    return overall accuracy of all participated clients
    """
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    return {"accuracy": sum(accuracies) / sum(examples)}

# ==========================================
# 2.  Configure ServerApp 
# ==========================================
def server_fn(context: fl.common.Context):
    # read from pyproject.toml
    num_rounds = context.run_config.get("num-server-rounds", 3)

    net = SimpleCNN() 
    ndarrays = get_parameters(net) 
    initial_parameters = fl.common.ndarrays_to_parameters(ndarrays) 
    
    # Define the FL strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,            # In each round, 100% clients participate the training
        fraction_evaluate=1.0,       # In each round, 100% clients participate the evaluation
        min_fit_clients=2,           # at least 2 clients are required for training
        min_evaluate_clients=2,      # at least 2 clients are required for evaluation
        min_available_clients=2,     # the system needs at least 2 existing clients
        evaluate_metrics_aggregation_fn=weighted_average, # overall accuracy used as aggregation metric
        initial_parameters=initial_parameters,
    )
    
    config = fl.server.ServerConfig(num_rounds=num_rounds)
    return fl.server.ServerAppComponents(strategy=strategy, config=config)

app = fl.server.ServerApp(server_fn=server_fn)