
import warnings
from typing import Any, Dict, Tuple

import torch
from torchmeta.toy import Sinusoid
from torchmeta.utils.data import BatchMetaDataLoader

warnings.filterwarnings(action="ignore")
def get_dataloader(config: Dict[str, Any]) -> Tuple[BatchMetaDataLoader, BatchMetaDataLoader, BatchMetaDataLoader]:
    train_dataset = Sinusoid(
        num_samples_per_task=config["num_shots"] * 2,
        num_tasks=config["num_batches_train"] * config["batch_size"],
        noise_std=None,
    )
    train_dataloader = BatchMetaDataLoader(
        train_dataset, batch_size=config["batch_size"]
    )

    val_dataset = Sinusoid(
        num_samples_per_task=config["num_shots"] * 2,
        num_tasks=config["num_batches_val"] * config["batch_size"],
        noise_std=None,
    )
    val_dataloader = BatchMetaDataLoader(val_dataset, batch_size=config["batch_size"])

    test_dataset = Sinusoid(
        num_samples_per_task=config["num_shots"] * 2,
        num_tasks=config["num_batches_test"] * config["batch_size"],
        noise_std=None,
    )
    test_dataloader = BatchMetaDataLoader(test_dataset, batch_size=config["batch_size"])

    return train_dataloader, val_dataloader, test_dataloader
config = {
    "num_shots": 5,
    "batch_size": 16,
    "num_batches_train": 6000,
    "num_batches_test": 2000,
    "num_batches_val": 100,
    "device": "cpu",  # "cuda" or "cpu"
}

train_dataloader, val_dataloader, test_dataloader = get_dataloader(config)
for batch_idx, batch in enumerate(val_dataloader):
    xs, ys = batch
    support_xs = (
        xs[:, : config["num_shots"], :].to(device=config["device"]).type(torch.float)
    )
    query_xs = (
        xs[:, config["num_shots"] :, :].to(device=config["device"]).type(torch.float)
    )
    support_ys = (
        ys[:, : config["num_shots"], :].to(device=config["device"]).type(torch.float)
    )
    query_ys = (
        ys[:, config["num_shots"] :, :].to(device=config["device"]).type(torch.float)
    )

    print(
        f"support_x shape : {support_xs.shape}\n",
        f"support_y shape : {support_ys.shape}\n",
        f"query_x shape   : {query_xs.shape}\n",
        f"query_y shape   : {query_ys.shape}",
    )

    break

# support_x shape : torch.Size([16, 5, 1]) # batch_size=16, num_shot=5
# support_y shape : torch.Size([16, 5, 1])
# query_x shape   : torch.Size([16, 5, 1])
# query_y shape   : torch.Size([16, 5, 1])