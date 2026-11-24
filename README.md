# Reproducing and extending PB-DID on Bot-IoT and UNSW-NB15

Course project, Machine Learning for Cybersecurity (CY-5002), Fall 2026
Ahsan Ahmed (26K-7808)

Paper reproduced: M. Zeeshan et al., "Protocol-Based Deep Intrusion Detection for DoS and DDoS Attacks Using UNSW-NB15 and Bot-IoT Data-Sets," IEEE Access, vol. 10, pp. 2269-2283, 2022. DOI 10.1109/ACCESS.2021.3137201

## What is in this repository

```
src/prepare_data.py      builds the 26 common features for both datasets
src/reproduce_pbdid.py   reproduction of the paper's LSTM experiments
src/extension.py         extension experiments (imbalance, cross-dataset, cost)
src/make_figures.py      figures and result tables from the saved results
notebooks/               data_walkthrough.ipynb, the raw data and the cleaning steps
results/                 JSON results, figures, summary_tables.md
requirements.txt         Python packages
```

The datasets are not in the repository because of their size. See the next section.

## Data

Both datasets come from the UNSW pages listed in the course dataset sheet.

| Folder | Files | Source |
|---|---|---|
| `data/bot_iot/` | `UNSW_2018_IoT_Botnet_Full5pc_1.csv` to `_4.csv` | Bot-IoT, folder `Dataset/5%/All features` |
| `data/unsw_nb15_raw/` | `UNSW-NB15_1.csv` to `_4.csv`, `NUSW-NB15_features.csv` | UNSW-NB15, folder `CSV Files` |

Bot-IoT 5% has 3,668,522 records. Raw UNSW-NB15 has 2,540,047 records. The raw UNSW-NB15 files are needed, not the training and testing partitions, because the paper's features use the IP addresses, ports and time stamps, and the partitions do not contain them.

## How to run

Python 3.12 or newer.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/prepare_data.py                  # about 20 seconds
python src/reproduce_pbdid.py               # about 2.5 minutes on a laptop CPU
python src/reproduce_pbdid.py --variant noid
python src/extension.py                     # about 1.5 minutes
python src/make_figures.py
```

All random seeds are fixed in the scripts.

For a quick run of the same code, limit the number of chunks or seeds. A quick run prints its results and does not overwrite the saved ones.

```
python src/reproduce_pbdid.py --chunks 1    # about 40 seconds
python src/extension.py --seeds 1           # about 20 seconds
```

`notebooks/data_walkthrough.ipynb` shows the raw files, the class counts and each cleaning step, with the outputs saved. It imports the functions from `src/prepare_data.py` and checks that its result is identical to the saved feature files. To run it again, open it with `jupyter lab` from the `notebooks/` folder. It needs the data and takes about 30 seconds.

## Reproduction

Setup taken from the paper:

- Normal records from UNSW-NB15, DoS and DDoS records from Bot-IoT
- the 26 flow and TCP features of the paper's Table 4. Bot-IoT has all of them. For UNSW-NB15, 14 are read directly and 12 statistics over windows of 100 connections are computed in `prepare_data.py`
- IP addresses replaced by random unique integers, ARP ports set to -1
- balanced chunks: the same number of records for each class, DoS and DDoS records unique to a chunk, the Normal part repeated in every chunk
- 128 records of one class form one training example
- embedding (16), two LSTM layers (20 units each), softmax output, dropout 0.1, 6 epochs
- three binary tasks and one three-class task, accuracy averaged over the chunks

Differences from the paper:

1. Scale. The paper uses the full Bot-IoT (73 million records) and 14 chunks of 2.218 million records per class. We use the 5% subset, so we have 4 chunks of 409,600 records per class. This covers 99% of the DoS records and 85% of the DDoS records in the subset.
2. Embedding. The paper's model summary gives 53,248 embedding weights but does not explain how real-valued features become embedding indices. We use a learned linear map from the 26 features to 16 dimensions. Our model has 6,895 weights, the paper's has 59,551.
3. Scaling. Features are log scaled and standardised with statistics of the training part.
4. Recurrent dropout (0.3 in the paper) is not available in the PyTorch LSTM and is left out.
5. The paper does not give the train and validation split. We use 80% and 20% of the groups.

Result (accuracy in %, mean and standard deviation over 4 chunks):

| Task | Paper | Ours, 26 features | Ours, without time and IP features |
|---|---|---|---|
| Normal vs DDoS | above 99 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| Normal vs DoS | above 99 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| DDoS vs DoS | above 97 | 99.02 ± 0.82 | 97.99 ± 0.81 |
| Three class | 96.3 | 98.46 ± 1.44 | 85.81 ± 8.97 |

The reported result is reproduced, with the same error pattern as in the paper: Normal is never confused, and the errors are DoS groups labelled as DDoS.

The last column is a check we added. Four of the 26 features are the start time, end time and the two IP addresses. Normal traffic was captured in 2015 on one network and the attacks in 2018 on another, so these four features identify the capture. Without them the three-class accuracy falls to 85.81% and varies strongly between chunks.

## Extension

The extension uses single records, not groups of 128, and leaves out the four time and IP features (22 features).

### A. Imbalance-aware five-class detection on Bot-IoT

Classes and records used: Normal 477, DoS 100,000 (sampled), DDoS 100,000 (sampled), Reconnaissance 91,082, Theft 79. Split 70% / 30%, stratified, 5 seeds. Each test split has 143 Normal and 24 Theft records, so the numbers for these two classes move in steps of about 0.7 and 4 percentage points.

| Method | Accuracy | Macro F1 | Macro PR-AUC | Normal recall | Normal precision | Theft recall | Theft precision |
|---|---|---|---|---|---|---|---|
| No handling | 0.9958 | 0.966 ± 0.023 | 0.991 | 0.943 | 0.972 | 0.825 | 0.968 |
| Equal count per class (paper's strategy) | 0.9432 | 0.757 ± 0.042 | 0.920 | 0.979 | 0.508 | 0.967 | 0.177 |
| Class weights | 0.9912 | 0.919 ± 0.014 | 0.988 | 0.989 | 0.876 | 0.958 | 0.545 |
| SMOTE (training split only) | 0.9960 | 0.974 ± 0.006 | 0.996 | 0.990 | 0.941 | 0.958 | 0.878 |
| Focal loss | 0.9874 | 0.908 ± 0.022 | 0.987 | 0.989 | 0.846 | 0.967 | 0.519 |

Accuracy hardly separates the methods, macro F1 and the per-class numbers do. The paper's equal-count strategy works when every class is large. With five classes it has to cut every class down to the 55 Theft records in the training split, and precision on the small classes collapses. SMOTE gives the best balance.

### B. Cross-dataset test

Classes present in both datasets: Normal, DoS and Reconnaissance. UNSW-NB15 has no DDoS class and no Theft class.

| Setting | Accuracy | Macro F1 | Normal recall | DoS recall | Reconnaissance recall |
|---|---|---|---|---|---|
| train Bot-IoT, test Bot-IoT | 0.999 | 0.967 | 0.997 | 1.000 | 0.999 |
| train Bot-IoT, test UNSW-NB15 | 0.644 | 0.323 | 0.801 | 0.001 | 0.275 |
| train UNSW-NB15, test UNSW-NB15 | 0.969 | 0.927 | 0.986 | 0.975 | 0.843 |
| train UNSW-NB15, test Bot-IoT | 0.030 | 0.030 | 0.236 | 0.030 | 0.029 |

A model trained on one dataset does not work on the other. This also puts the perfect Normal score of the reproduction in context: there Normal comes from one dataset and the attacks from the other.

### C. Cost

| Model | Parameters | Microseconds per record (CPU, one thread) |
|---|---|---|
| PB-DID LSTM, groups of 128 | 6,895 | 0.51 |
| Record-level network 64-32 | 3,717 | 0.03 |
| Light record-level network 16 | 453 | 0.01 |

The light network reaches accuracy 0.962 and macro F1 0.838 on the five-class task with class weights.

## Results files

- `results/reproduction_full/results.json`, `results/reproduction_noid/results.json`
- `results/extension/results.json`
- `results/summary_tables.md`
- `results/figures/fig1_reproduction_confusion.png` to `fig4_cross_dataset.png`
