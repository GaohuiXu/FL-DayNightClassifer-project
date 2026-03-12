import flwr as fl
from utils import *

# ==========================================
# 1. Metrics aggregation
# ==========================================
def weighted_average_eval(metrics):
    """Aggregate evaluate metrics (e.g., accuracy) weighted by num_examples."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def weighted_average_fit(metrics):
    """Aggregate fit metrics (e.g., train_loss, update_norm) weighted by num_examples."""
    examples = [num_examples for num_examples, _ in metrics]
    total = sum(examples) if examples else 1

    # Some clients might not return all keys; be defensive
    def wavg(key, default=0.0):
        vals = []
        for num_examples, m in metrics:
            if key in m:
                vals.append(num_examples * float(m[key]))
        return sum(vals) / total if vals else default

    return {
        "train_loss": wavg("train_loss"),
        "update_norm": wavg("update_norm"),
    }

# ==========================================
# 2. Per-round config sent to clients
# ==========================================
def fit_config(server_round: int):
    # Example schedule; tune later
    if server_round < 3:
        lr = 1e-3
    else:
        lr = 5e-4

    return {
        "server-round": server_round,  # optional: Flower often injects this anyway in newer APIs
        "local_epochs": 1,
        "batch_size": 32,
        "lr": lr,
    }

# ==========================================
# 3. Configure ServerApp
# ==========================================
def server_fn(context: fl.common.Context):
    num_rounds = int(context.run_config.get("num-server-rounds", 3))

    net = SimpleCNN()
    ndarrays = get_parameters(net)
    initial_parameters = fl.common.ndarrays_to_parameters(ndarrays)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        initial_parameters=initial_parameters,

        # NEW: send config to clients each round
        on_fit_config_fn=fit_config,

        # NEW: aggregate fit metrics to avoid warning
        fit_metrics_aggregation_fn=weighted_average_fit,

        # Existing: aggregate eval metrics (accuracy)
        evaluate_metrics_aggregation_fn=weighted_average_eval,
    )

    config = fl.server.ServerConfig(num_rounds=num_rounds)
    return fl.server.ServerAppComponents(strategy=strategy, config=config)

app = fl.server.ServerApp(server_fn=server_fn)