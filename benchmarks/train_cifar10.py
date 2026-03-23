#!/usr/bin/env python3
"""
CIFAR-10 training benchmark comparing FusedMuon, vanilla Muon, and AdamW.

Model : torchvision ResNet-18 (adapted for 32x32 CIFAR images)
Epochs: 20
Batch : 128

Outputs per-optimizer JSON logs to benchmarks/logs/{name}_cifar10.json and
prints a summary table at the end.
"""

import json
import os
import sys
import time

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------------
# Optional imports (handle missing gracefully)
# ---------------------------------------------------------------------------
try:
    import torchvision
    import torchvision.transforms as transforms
except ImportError:
    print("ERROR: torchvision is required. Install it with: pip install torchvision")
    sys.exit(1)

from benchmarks.reference_muon import Muon as VanillaMuon

try:
    from muon_fused import FusedMuon
except ImportError:
    FusedMuon = None
    print("[WARN] muon_fused.FusedMuon could not be imported; FusedMuon benchmark will be skipped.")

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
NUM_EPOCHS = 20
BATCH_SIZE = 128
LR_MUON = 0.02
LR_ADAMW = 1e-3
WEIGHT_DECAY = 0.1
NS_STEPS = 5
NUM_WORKERS = 4

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_dataloaders():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    data_root = os.path.join(SCRIPT_DIR, "data")
    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True,
                                            transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True,
                                           transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=NUM_WORKERS,
                                              pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=NUM_WORKERS,
                                             pin_memory=True)
    return trainloader, testloader


# ---------------------------------------------------------------------------
# Model (ResNet-18 adapted for CIFAR-10)
# ---------------------------------------------------------------------------

def make_model(device):
    model = torchvision.models.resnet18(num_classes=10)
    # Replace the first conv layer: CIFAR-10 images are 32x32, not 224x224
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model = model.to(device)
    return model


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def _can_use_muon(p):
    return p.ndim >= 2


def build_optimizer(name, model):
    """Return (optimizer, display_name)."""
    if name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=LR_ADAMW,
                                weight_decay=WEIGHT_DECAY)
        return opt, "AdamW"

    elif name == "vanilla_muon":
        muon_params = [p for p in model.parameters() if _can_use_muon(p)]
        adamw_params = [p for p in model.parameters() if not _can_use_muon(p)]
        groups = []
        if muon_params:
            groups.append({"params": muon_params, "use_muon": True})
        if adamw_params:
            groups.append({"params": adamw_params, "use_muon": False})
        opt = VanillaMuon(groups, lr=LR_MUON, wd=WEIGHT_DECAY, ns_steps=NS_STEPS)
        return opt, "VanillaMuon"

    elif name == "fused_muon":
        if FusedMuon is None:
            return None, "FusedMuon"
        muon_params = [p for p in model.parameters() if _can_use_muon(p)]
        adamw_params = [p for p in model.parameters() if not _can_use_muon(p)]
        groups = []
        if muon_params:
            groups.append({"params": muon_params, "use_muon": True})
        if adamw_params:
            groups.append({"params": adamw_params, "use_muon": False})
        opt = FusedMuon(groups, lr=LR_MUON, wd=WEIGHT_DECAY, ns_steps=NS_STEPS)
        return opt, "FusedMuon"

    else:
        raise ValueError(f"Unknown optimizer: {name}")


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    opt_time_ms = 0.0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        opt_time_ms += (t1 - t0) * 1000.0
        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    return total_loss / total_samples, opt_time_ms


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
    return correct / total


# ---------------------------------------------------------------------------
# Run one full experiment
# ---------------------------------------------------------------------------

def run_experiment(opt_name, device):
    trainloader, testloader = get_dataloaders()
    model = make_model(device)
    criterion = nn.CrossEntropyLoss()

    optimizer, display_name = build_optimizer(opt_name, model)
    if optimizer is None:
        print(f"  [SKIP] {display_name} is not available.")
        return None, display_name

    log = {"optimizer": display_name, "epochs": []}
    wall_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        train_loss, opt_time = train_one_epoch(model, trainloader, optimizer, criterion, device)
        test_acc = evaluate(model, testloader, device)
        epoch_time = time.time() - epoch_start

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "test_accuracy": round(test_acc, 5),
            "epoch_time_s": round(epoch_time, 3),
            "optimizer_step_time_ms": round(opt_time, 3),
        }
        log["epochs"].append(record)
        print(f"  [{display_name}] Epoch {epoch:2d}/{NUM_EPOCHS}  "
              f"loss={train_loss:.4f}  acc={test_acc:.4f}  "
              f"epoch={epoch_time:.1f}s  opt={opt_time:.1f}ms")

    log["total_wall_time_s"] = round(time.time() - wall_start, 2)
    return log, display_name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires a GPU.")
        sys.exit(1)

    device = torch.device("cuda")
    log_dir = os.path.join(SCRIPT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)

    configs = ["fused_muon", "vanilla_muon", "adamw"]
    all_logs = {}

    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"  Training with: {cfg}")
        print(f"{'='*60}")
        log, display_name = run_experiment(cfg, device)
        if log is not None:
            log_path = os.path.join(log_dir, f"{cfg}_cifar10.json")
            with open(log_path, "w") as f:
                json.dump(log, f, indent=2)
            print(f"  Log saved to {log_path}")
            all_logs[display_name] = log

    # --- Summary table ---
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    header = f"{'Optimizer':>14s}  {'Final Loss':>11s}  {'Final Acc':>10s}  {'Total Time':>11s}  {'Avg Opt/Epoch':>14s}"
    print(header)
    print("-" * len(header))

    for name, log in all_logs.items():
        epochs = log["epochs"]
        final = epochs[-1]
        avg_opt = sum(e["optimizer_step_time_ms"] for e in epochs) / len(epochs)
        print(f"{name:>14s}  {final['train_loss']:11.5f}  {final['test_accuracy']:10.4f}  "
              f"{log['total_wall_time_s']:9.1f}s  {avg_opt:12.1f}ms")


if __name__ == "__main__":
    main()
