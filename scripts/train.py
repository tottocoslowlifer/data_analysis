import pandas as pd
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from experiment_tools.set_up import start_experiment
from model.mymodel import NeuralNetwork
from trainer.loop import model_train
from utils.cfg_diff import get_config, get_diff
from utils.preprocessing import prepare_nn
from utils.result import plot_data, torch_compare_preds, result_print


default_filename = "../config/default/model.json"
exp_filename = "../config/experiment/model.json"

_, cfg, default_str, exp_str = get_config(default_filename, exp_filename)

logger = start_experiment(cfg)
logger = get_diff(default_str, exp_str, logger)

device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"device: {device}")

df = pd.read_csv(cfg["data_path"])
out = prepare_nn(
    df, cfg["params"]["test_size_1"], cfg["params"]["test_size_2"]
)

logger.info(f"train data records: {len(out['train_data'])}")
logger.info(f"valid data records: {len(out['valid_data'])}")
logger.info(f"eval data records: {len(out['eval_data'])}")

train_dataloader = DataLoader(
    dataset=out["train_dataset"],
    batch_size=cfg["dataloader_params"]["batch_size"],
    shuffle=cfg["dataloader_params"]["shuffle"],
    sampler=cfg["dataloader_params"]["sampler"],
    batch_sampler=cfg["dataloader_params"]["batch_sampler"],
    num_workers=cfg["dataloader_params"]["num_workers"],
    collate_fn=cfg["dataloader_params"]["collate_fn"],
    pin_memory=cfg["dataloader_params"]["pin_memory"],
    drop_last=cfg["dataloader_params"]["drop_last"],
    timeout=cfg["dataloader_params"]["timeout"],
    worker_init_fn=cfg["dataloader_params"]["worker_init_fn"],
    prefetch_factor=cfg["dataloader_params"]["prefetch_factor"],
    persistent_workers=cfg["dataloader_params"]["persistent_workers"],
    pin_memory_device=cfg["dataloader_params"]["pin_memory_device"]
)

valid_dataloader = DataLoader(
    dataset=out["valid_dataset"],
    batch_size=cfg["dataloader_params"]["batch_size"],
    shuffle=cfg["dataloader_params"]["shuffle"],
    sampler=cfg["dataloader_params"]["sampler"],
    batch_sampler=cfg["dataloader_params"]["batch_sampler"],
    num_workers=cfg["dataloader_params"]["num_workers"],
    collate_fn=cfg["dataloader_params"]["collate_fn"],
    pin_memory=cfg["dataloader_params"]["pin_memory"],
    drop_last=cfg["dataloader_params"]["drop_last"],
    timeout=cfg["dataloader_params"]["timeout"],
    worker_init_fn=cfg["dataloader_params"]["worker_init_fn"],
    prefetch_factor=cfg["dataloader_params"]["prefetch_factor"],
    persistent_workers=cfg["dataloader_params"]["persistent_workers"],
    pin_memory_device=cfg["dataloader_params"]["pin_memory_device"]
)

model = NeuralNetwork(
    input_dim=cfg["params"]["input_dim"],
    output_dim=cfg["params"]["output_dim"],
    first_dim=cfg["params"]["first_dim"],
    second_dim=cfg["params"]["second_dim"],
    slope=cfg["params"]["slope"]
).to(device=device)

logger.info(f"model architecture:\n\n{model}\n")

optimizer = SGD(model.parameters(), lr=cfg["params"]["lr"])
criterion = nn.MSELoss()

model, training_data = model_train(
    epochs=cfg["params"]["epochs"],
    train_dataloader=train_dataloader,
    valid_dataloader=valid_dataloader,
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    batch_size=cfg["dataloader_params"]["batch_size"],
    device=device
)

out_dir = cfg["log"]["log_file"].replace("NeuralNetwork.log", "")

plot_data(training_data, out_dir)
torch.save(model.state_dict(), f"{out_dir}model_weight.pth")

y_pred = model(out["X_eval"]).detach().numpy().copy()
y_pred = out["scaler"].inverse_transform(pd.DataFrame(y_pred))

y_eval = out["scaler"].inverse_transform(
    out["eval_data"]["observed"].to_frame()
)
torch_compare_preds(y_eval, y_pred, out_dir)

logger = result_print(
    model, out["scaler"], out["train_data"], out["X_train"], logger, "train"
)
logger = result_print(
    model, out["scaler"], out["valid_data"], out["X_valid"], logger, "valid"
)
logger = result_print(
    model, out["scaler"], out["eval_data"], out["X_eval"], logger, "eval"
)
