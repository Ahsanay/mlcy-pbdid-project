"""
Step 2: reproduce PB-DID (Zeeshan et al., 2022) at the scale of Bot-IoT 5%.

What is kept from the paper
  - Normal records from UNSW-NB15, DoS and DDoS records from Bot-IoT
  - the 26 flow + TCP features of Table 4
  - balanced chunks: same number of records for each class in a chunk,
    DoS/DDoS records unique per chunk, the Normal part repeated
  - 128 records of one class form one training example with one label
  - embedding (16) -> LSTM(20) -> LSTM(20) -> dense softmax
    dropout 0.1, tanh / sigmoid activations, 6 epochs
  - three binary tasks and the three-class task, accuracy averaged over chunks

What is different (also listed in the README)
  - 4 chunks of 409,600 records per class instead of 14 chunks of 2.218M,
    because we use the 5% Bot-IoT subset
  - the embedding is a learned linear map of the 26 feature values; the
    paper's table gives 53,248 embedding weights but does not say how real
    valued features are turned into indices
  - features are log scaled and standardised with training statistics
  - recurrent dropout is not available in the PyTorch LSTM, so only the
    ordinary dropout of 0.1 is used

Run:  python src/reproduce_pbdid.py                 (all 26 features)
      python src/reproduce_pbdid.py --variant noid  (without time and IP features)
      python src/reproduce_pbdid.py --chunks 1      (quick run, one chunk, nothing saved)
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

from prepare_data import FEATURES, OUT as PROCESSED, ROOT

SEED = 42
GROUP = 128            # records per training example (paper: batch of 128)
GROUPS_PER_CLASS = 3200  # per chunk -> 409,600 records per class
N_CHUNKS = 4
EPOCHS = 6
CLASSES = ["Normal", "DDoS", "DoS"]   # label order used in the paper's figures
ID_FEATURES = ["stime", "ltime", "saddr", "daddr"]


class PBDID(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.embed = nn.Linear(n_features, 16)
        self.lstm1 = nn.LSTM(16, 20, batch_first=True)
        self.drop = nn.Dropout(0.1)
        self.lstm2 = nn.LSTM(20, 20, batch_first=True)
        self.out = nn.Linear(20, n_classes)

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.lstm1(x)
        x = self.drop(x)
        x, _ = self.lstm2(x)
        x = self.drop(x[:, -1, :])
        return self.out(x)


def make_groups(df, features):
    """Cut a class into consecutive groups of 128 records (time order)."""
    n = (len(df) // GROUP) * GROUP
    x = df[features].to_numpy(dtype=np.float32)[:n]
    return x.reshape(-1, GROUP, len(features))


def scale(train, test):
    """log scale, then zero mean / unit variance using the training part."""
    f = lambda a: np.sign(a) * np.log1p(np.abs(a))
    train, test = f(train), f(test)
    mu = train.mean(axis=(0, 1), keepdims=True)
    sd = train.std(axis=(0, 1), keepdims=True) + 1e-6
    return (train - mu) / sd, (test - mu) / sd


def train_and_test(xtr, ytr, xte, n_classes, seed):
    torch.manual_seed(seed)
    model = PBDID(xtr.shape[2], n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    xtr_t, ytr_t = torch.from_numpy(xtr), torch.from_numpy(ytr)
    for _ in range(EPOCHS):
        model.train()
        order = torch.randperm(len(xtr_t))
        for i in range(0, len(order), 64):
            idx = order[i:i + 64]
            opt.zero_grad()
            loss = loss_fn(model(xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(torch.from_numpy(xte[i:i + 512])).argmax(1)
                          for i in range(0, len(xte), 512)])
    return pred.numpy(), sum(p.numel() for p in model.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["full", "noid"], default="full")
    ap.add_argument("--chunks", type=int, default=N_CHUNKS,
                    help="number of chunks to run (default: all 4)")
    args = ap.parse_args()

    features = list(FEATURES)
    if args.variant == "noid":
        features = [f for f in features if f not in ID_FEATURES]
    out_dir = os.path.join(ROOT, "results", "reproduction_" + args.variant)
    os.makedirs(out_dir, exist_ok=True)

    unsw = pd.read_parquet(os.path.join(PROCESSED, "unsw_26.parquet"))
    bot = pd.read_parquet(os.path.join(PROCESSED, "botiot_26.parquet"))
    groups = {
        "Normal": make_groups(unsw[unsw.category == "Normal"], features),
        "DDoS": make_groups(bot[bot.category == "DDoS"], features),
        "DoS": make_groups(bot[bot.category == "DoS"], features),
    }
    del unsw, bot
    for k, v in groups.items():
        print(f"{k}: {len(v)} groups of {GROUP}")

    rng = np.random.default_rng(SEED)
    # Normal: one fixed sample, repeated in every chunk (as in the paper)
    normal_idx = rng.choice(len(groups["Normal"]), GROUPS_PER_CLASS, replace=False)
    # DoS / DDoS: disjoint groups for every chunk
    attack_idx = {c: rng.permutation(len(groups[c]))[:N_CHUNKS * GROUPS_PER_CLASS]
                  .reshape(N_CHUNKS, GROUPS_PER_CLASS) for c in ["DDoS", "DoS"]}

    tasks = {
        "Normal_vs_DDoS": ["Normal", "DDoS"],
        "Normal_vs_DoS": ["Normal", "DoS"],
        "DDoS_vs_DoS": ["DDoS", "DoS"],
        "Three_class": CLASSES,
    }
    results = {t: [] for t in tasks}
    n_params = {}

    n_run = max(1, min(args.chunks, N_CHUNKS))
    for chunk in range(n_run):
        part = {"Normal": groups["Normal"][normal_idx],
                "DDoS": groups["DDoS"][attack_idx["DDoS"][chunk]],
                "DoS": groups["DoS"][attack_idx["DoS"][chunk]]}
        # 80 / 20 split of the groups, same split positions for every class
        split = np.random.default_rng(SEED + chunk).permutation(GROUPS_PER_CLASS)
        cut = int(0.8 * GROUPS_PER_CLASS)
        tr, te = split[:cut], split[cut:]

        for task, names in tasks.items():
            t0 = time.time()
            xtr = np.concatenate([part[c][tr] for c in names])
            xte = np.concatenate([part[c][te] for c in names])
            ytr = np.repeat(np.arange(len(names)), len(tr)).astype(np.int64)
            yte = np.repeat(np.arange(len(names)), len(te)).astype(np.int64)
            xtr, xte = scale(xtr, xte)
            pred, n_par = train_and_test(xtr, ytr, xte, len(names), SEED + chunk)
            acc = accuracy_score(yte, pred)
            p, r, f1, _ = precision_recall_fscore_support(
                yte, pred, labels=range(len(names)), zero_division=0)
            cm = confusion_matrix(yte, pred, labels=range(len(names)))
            results[task].append({
                "chunk": chunk + 1, "accuracy": float(acc),
                "precision": p.tolist(), "recall": r.tolist(), "f1": f1.tolist(),
                "confusion_matrix": cm.tolist()})
            n_params[task] = n_par
            print(f"chunk {chunk + 1}  {task:15s} acc {acc:.4f}  "
                  f"({time.time() - t0:.0f}s)")

    summary = {"variant": args.variant, "features": features,
               "records_per_class_per_chunk": GROUPS_PER_CLASS * GROUP,
               "chunks": n_run, "epochs": EPOCHS, "tasks": {}}
    print("\naverage over chunks")
    for task, names in tasks.items():
        acc = np.mean([r["accuracy"] for r in results[task]])
        cm = np.sum([r["confusion_matrix"] for r in results[task]], axis=0)
        summary["tasks"][task] = {
            "classes": names, "mean_accuracy": float(acc),
            "std_accuracy": float(np.std([r["accuracy"] for r in results[task]])),
            "mean_precision": np.mean([r["precision"] for r in results[task]], axis=0).tolist(),
            "mean_recall": np.mean([r["recall"] for r in results[task]], axis=0).tolist(),
            "mean_f1": np.mean([r["f1"] for r in results[task]], axis=0).tolist(),
            "confusion_matrix_sum": cm.tolist(),
            "parameters": n_params[task], "per_chunk": results[task]}
        print(f"  {task:15s} {acc * 100:.2f}%")

    if n_run < N_CHUNKS:
        print(f"quick run with {n_run} of {N_CHUNKS} chunks, saved results are not changed")
        return
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("saved", os.path.join(out_dir, "results.json"))


if __name__ == "__main__":
    main()
