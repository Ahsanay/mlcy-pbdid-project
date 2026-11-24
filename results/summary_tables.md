# Result tables

## Reproduction: accuracy (%), mean and standard deviation over 4 chunks

| Task | Paper | Ours, 26 features | Ours, 22 features (no time, no IP) |
|---|---|---|---|
| Normal vs DDoS | above 99 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| Normal vs DoS | above 99 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| DDoS vs DoS | above 97 | 99.02 ± 0.82 | 97.99 ± 0.81 |
| Three class | 96.3 | 98.46 ± 1.44 | 85.81 ± 8.97 |

Three-class, per-class precision / recall / F1 (26 features):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Normal | 1.000 | 1.000 | 1.000 |
| DDoS | 0.964 | 0.992 | 0.978 |
| DoS | 0.992 | 0.962 | 0.976 |

## Extension A: five-class Bot-IoT, mean over 5 seeds

Class counts used: Normal 477, DoS 100,000, DDoS 100,000, Reconnaissance 91,082, Theft 79

| Method | Accuracy | Macro F1 | Macro PR-AUC | Normal recall | Normal precision | Theft recall | Theft precision |
|---|---|---|---|---|---|---|---|
| No handling | 0.9958 | 0.966 ± 0.023 | 0.991 | 0.943 | 0.972 | 0.825 | 0.968 |
| Equal count (paper) | 0.9432 | 0.757 ± 0.042 | 0.920 | 0.979 | 0.508 | 0.967 | 0.177 |
| Class weights | 0.9912 | 0.919 ± 0.014 | 0.988 | 0.989 | 0.876 | 0.958 | 0.545 |
| SMOTE | 0.9960 | 0.974 ± 0.006 | 0.996 | 0.990 | 0.941 | 0.958 | 0.878 |
| Focal loss | 0.9874 | 0.908 ± 0.022 | 0.987 | 0.989 | 0.846 | 0.967 | 0.519 |

## Extension B: cross-dataset (Normal, DoS, Reconnaissance), mean over 5 seeds

| Setting | Accuracy | Macro F1 | Normal recall | DoS recall | Reconnaissance recall |
|---|---|---|---|---|---|
| train Bot-IoT, test Bot-IoT | 0.999 | 0.967 | 0.997 | 1.000 | 0.999 |
| train Bot-IoT, test UNSW-NB15 | 0.644 | 0.323 | 0.801 | 0.001 | 0.275 |
| train UNSW-NB15, test UNSW-NB15 | 0.969 | 0.927 | 0.986 | 0.975 | 0.843 |
| train UNSW-NB15, test Bot-IoT | 0.030 | 0.030 | 0.236 | 0.030 | 0.029 |

## Extension C: model cost (CPU, one thread)

| Model | Parameters | Microseconds per record |
|---|---|---|
| PB-DID LSTM (26 features, groups of 128) | 6,895 | 0.51 |
| Record-level network 64-32 (22 features) | 3,717 | 0.03 |
| Light record-level network 16 (22 features) | 453 | 0.01 |

Light network with class weights, five-class Bot-IoT: accuracy 0.9619, macro F1 0.838.
