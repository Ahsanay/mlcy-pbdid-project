"""
Step 3: extension experiments.

A. Imbalance-aware five-class detection on Bot-IoT
   (Normal, DoS, DDoS, Reconnaissance, Theft), one record = one example.
   Five ways of handling the imbalance are compared on the same splits:
     plain         cross-entropy, no handling
     undersample   equal count per class (the strategy of the original paper)
     weighted      class-weighted cross-entropy
     smote         SMOTE on the training split only
     focal         focal loss (gamma = 2) with class weights

B. Cross-dataset test on the classes both datasets contain
   (Normal, DoS, Reconnaissance): train on one dataset, test on the other,
   and compare with the in-dataset hold-out score.

C. Cost: number of parameters and inference time per record for the
   reproduced LSTM, the record-level network and a smaller network.

The time stamps and IP addresses are not used here (22 features), because
they identify the capture, not the behaviour.

Run:  python src/extension.py
      python src/extension.py --seeds 1   (quick run, one seed, nothing saved)
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split

from prepare_data import FEATURES, OUT as PROCESSED, ROOT
from reproduce_pbdid import ID_FEATURES, PBDID

FEATS = [f for f in FEATURES if f not in ID_FEATURES]
SEEDS = [1, 2, 3, 4, 5]
EPOCHS = 15
MIN_STEPS = 3000
OUT_DIR = os.path.join(ROOT, "results", "extension")


def log_scale(a):
    return np.sign(a) * np.log1p(np.abs(a))


def standardise(train, *others):
    train = log_scale(train)
    mu, sd = train.mean(0), train.std(0) + 1e-6
    return [(train - mu) / sd] + [(log_scale(o) - mu) / sd for o in others]


def make_net(n_in, n_out, hidden=(64, 32)):
    layers, last = [], n_in
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(0.1)]
        last = h
    layers.append(nn.Linear(last, n_out))
    return nn.Sequential(*layers)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight, self.gamma = weight, gamma

    def forward(self, logits, target):
        logp = torch.log_softmax(logits, 1)
        pt = logp.gather(1, target[:, None]).squeeze(1).exp()
        ce = nn.functional.nll_loss(logp, target, weight=self.weight,
                                    reduction="none")
        return ((1 - pt) ** self.gamma * ce).mean()


def fit(x, y, n_classes, loss_fn, seed, hidden=(64, 32)):
    torch.manual_seed(seed)
    net = make_net(x.shape[1], n_classes, hidden)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    xt, yt = torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y)
    # small training sets (undersampling) get a smaller batch and more
    # epochs, so every method sees at least MIN_STEPS weight updates
    batch = 512 if len(xt) >= 5120 else 32
    steps_per_epoch = int(np.ceil(len(xt) / batch))
    epochs = max(EPOCHS, int(np.ceil(MIN_STEPS / steps_per_epoch)))
    for _ in range(epochs):
        net.train()
        order = torch.randperm(len(xt))
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            opt.zero_grad()
            loss_fn(net(xt[idx]), yt[idx]).backward()
            opt.step()
    return net.eval()


def predict(net, x):
    with torch.no_grad():
        return torch.softmax(net(torch.from_numpy(x.astype(np.float32))), 1).numpy()


def class_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes)
    w = len(y) / (n_classes * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


def score(y, prob, names):
    pred = prob.argmax(1)
    p, r, f1, _ = precision_recall_fscore_support(
        y, pred, labels=range(len(names)), zero_division=0)
    onehot = np.eye(len(names))[y]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "macro_pr_auc": float(average_precision_score(onehot, prob, average="macro")),
        "precision": dict(zip(names, p.tolist())),
        "recall": dict(zip(names, r.tolist())),
        "f1": dict(zip(names, f1.tolist())),
        "confusion_matrix": confusion_matrix(y, pred, labels=range(len(names))).tolist(),
    }


def average(runs, names):
    out = {}
    for k in ["accuracy", "macro_f1", "weighted_f1", "macro_pr_auc"]:
        v = [r[k] for r in runs]
        out[k] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
    for k in ["precision", "recall", "f1"]:
        out[k] = {n: {"mean": float(np.mean([r[k][n] for r in runs])),
                      "std": float(np.std([r[k][n] for r in runs]))} for n in names}
    out["confusion_matrix_sum"] = np.sum([r["confusion_matrix"] for r in runs], 0).tolist()
    return out


# ---------------------------------------------------------------- part A
def part_a(bot):
    names = ["Normal", "DoS", "DDoS", "Reconnaissance", "Theft"]
    rng = np.random.default_rng(0)
    parts = []
    for c in names:
        d = bot[bot.category == c]
        if len(d) > 100_000:   # DoS and DDoS: 100,000 records each are enough
            d = d.iloc[rng.choice(len(d), 100_000, replace=False)]
        parts.append(d)
    data = pd.concat(parts)
    x = data[FEATS].to_numpy(np.float64)
    y = data["category"].map({n: i for i, n in enumerate(names)}).to_numpy()
    counts = dict(zip(names, np.bincount(y).tolist()))
    print("part A class counts:", counts)

    methods = ["plain", "undersample", "weighted", "smote", "focal"]
    runs = {m: [] for m in methods}
    for seed in SEEDS:
        xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.3,
                                              stratify=y, random_state=seed)
        xtr, xte = standardise(xtr, xte)
        w = class_weights(ytr, len(names))
        for m in methods:
            xa, ya, loss = xtr, ytr, nn.CrossEntropyLoss()
            if m == "undersample":
                n_min = np.bincount(ytr).min()
                r = np.random.default_rng(seed)
                keep = np.concatenate([r.choice(np.where(ytr == c)[0], n_min, replace=False)
                                       for c in range(len(names))])
                xa, ya = xtr[keep], ytr[keep]
            elif m == "weighted":
                loss = nn.CrossEntropyLoss(weight=w)
            elif m == "smote":
                # raise the two small classes to 5,000 records, training split only
                target = {c: max(5000, n) for c, n in enumerate(np.bincount(ytr))}
                xa, ya = SMOTE(sampling_strategy=target, k_neighbors=5,
                               random_state=seed).fit_resample(xtr, ytr)
            elif m == "focal":
                loss = FocalLoss(weight=w)
            net = fit(xa, ya, len(names), loss, seed)
            s = score(yte, predict(net, xte), names)
            runs[m].append(s)
            print(f"  seed {seed} {m:12s} acc {s['accuracy']:.4f} "
                  f"macroF1 {s['macro_f1']:.4f} "
                  f"recall Normal {s['recall']['Normal']:.3f} Theft {s['recall']['Theft']:.3f}")
    test_counts = dict(zip(names, np.bincount(yte).tolist()))
    return {"classes": names, "class_counts": counts, "test_counts_per_seed": test_counts,
            "seeds": SEEDS, "methods": {m: average(runs[m], names) for m in methods}}


# ---------------------------------------------------------------- part B
def part_b(bot, unsw):
    names = ["Normal", "DoS", "Reconnaissance"]
    lab = {n: i for i, n in enumerate(names)}

    def take(df, seed):
        rng = np.random.default_rng(seed)
        parts = []
        for c in names:
            d = df[df.category == c]
            if len(d) > 100_000:
                d = d.iloc[rng.choice(len(d), 100_000, replace=False)]
            parts.append(d)
        d = pd.concat(parts)
        return d[FEATS].to_numpy(np.float64), d["category"].map(lab).to_numpy()

    sets = {"Bot-IoT": take(bot, 0), "UNSW-NB15": take(unsw, 0)}
    counts = {k: dict(zip(names, np.bincount(v[1]).tolist())) for k, v in sets.items()}
    print("part B class counts:", counts)

    out = {"classes": names, "class_counts": counts, "seeds": SEEDS, "runs": {}}
    for src in sets:
        other = [k for k in sets if k != src][0]
        inside, cross = [], []
        for seed in SEEDS:
            x, y = sets[src]
            xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.3,
                                                  stratify=y, random_state=seed)
            xo, yo = sets[other]
            xtr, xte, xo_s = standardise(xtr, xte, xo)
            net = fit(xtr, ytr, len(names),
                      nn.CrossEntropyLoss(weight=class_weights(ytr, len(names))), seed)
            inside.append(score(yte, predict(net, xte), names))
            cross.append(score(yo, predict(net, xo_s), names))
        out["runs"][f"train {src}, test {src}"] = average(inside, names)
        out["runs"][f"train {src}, test {other}"] = average(cross, names)
        print(f"  train {src}: in-dataset macroF1 "
              f"{np.mean([r['macro_f1'] for r in inside]):.4f}, on {other} "
              f"{np.mean([r['macro_f1'] for r in cross]):.4f}")
    return out


# ---------------------------------------------------------------- part C
def part_c():
    def n_params(m):
        return sum(p.numel() for p in m.parameters())

    def per_record_ms(model, x, records):
        model.eval()
        with torch.no_grad():
            model(x)
            t0 = time.perf_counter()
            for _ in range(20):
                model(x)
        return (time.perf_counter() - t0) / 20 / records * 1000

    torch.set_num_threads(1)
    n = 128 * 64
    models = {
        "PB-DID LSTM (26 features, groups of 128)":
            (PBDID(26, 3), torch.randn(64, 128, 26)),
        "Record-level network 64-32 (22 features)":
            (make_net(22, 5, (64, 32)), torch.randn(n, 22)),
        "Light record-level network 16 (22 features)":
            (make_net(22, 5, (16,)), torch.randn(n, 22)),
    }
    out = {}
    for name, (m, x) in models.items():
        out[name] = {"parameters": n_params(m),
                     "ms_per_record_cpu_1_thread": per_record_ms(m, x, n)}
        print(f"  {name}: {out[name]['parameters']} parameters, "
              f"{out[name]['ms_per_record_cpu_1_thread'] * 1000:.2f} microseconds per record")
    return out


def light_model_quality(bot):
    """Part A 'weighted' setting again, with the 16-unit network."""
    names = ["Normal", "DoS", "DDoS", "Reconnaissance", "Theft"]
    rng = np.random.default_rng(0)
    parts = []
    for c in names:
        d = bot[bot.category == c]
        if len(d) > 100_000:
            d = d.iloc[rng.choice(len(d), 100_000, replace=False)]
        parts.append(d)
    data = pd.concat(parts)
    x = data[FEATS].to_numpy(np.float64)
    y = data["category"].map({n: i for i, n in enumerate(names)}).to_numpy()
    runs = []
    for seed in SEEDS:
        xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.3,
                                              stratify=y, random_state=seed)
        xtr, xte = standardise(xtr, xte)
        net = fit(xtr, ytr, len(names),
                  nn.CrossEntropyLoss(weight=class_weights(ytr, len(names))),
                  seed, hidden=(16,))
        runs.append(score(yte, predict(net, xte), names))
    return average(runs, names)


def main():
    global SEEDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=len(SEEDS),
                    help="number of seeds to run (default: all 5)")
    args = ap.parse_args()
    all_seeds = len(SEEDS)
    SEEDS = SEEDS[:max(1, min(args.seeds, all_seeds))]
    os.makedirs(OUT_DIR, exist_ok=True)
    bot = pd.read_parquet(os.path.join(PROCESSED, "botiot_26.parquet"))
    unsw = pd.read_parquet(os.path.join(PROCESSED, "unsw_26.parquet"))

    print("A. five-class Bot-IoT, imbalance handling")
    a = part_a(bot)
    print("B. cross-dataset")
    b = part_b(bot, unsw)
    print("C. cost")
    c = part_c()
    light = light_model_quality(bot)
    print(f"  light network, weighted loss: macroF1 {light['macro_f1']['mean']:.4f}")

    if len(SEEDS) < all_seeds:
        print(f"quick run with {len(SEEDS)} of {all_seeds} seeds, saved results are not changed")
        return
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump({"features": FEATS, "epochs": EPOCHS,
                   "A_imbalance": a, "B_cross_dataset": b,
                   "C_cost": c, "C_light_model_quality": light}, f, indent=2)
    print("saved", os.path.join(OUT_DIR, "results.json"))


if __name__ == "__main__":
    main()
