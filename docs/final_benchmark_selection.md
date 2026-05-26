# Final Benchmark Selection

## Selection Summary

- selection metric = validation Macro F1
- test Macro F1 is confirmation only
- benchmark rank 1 model by validation Macro F1: `ResNet18`
- benchmark top3 by validation Macro F1: `ResNet18`, `LeNet`, `ResNet101`
- benchmark top5 by validation Macro F1: `ResNet18`, `LeNet`, `ResNet101`, `ResNet50`, `ViT`
- default low-data model set suggestion: `ResNet18`, `LeNet`, `ResNet101`

## Selection Rule

- primary sort: `val_macro_f1` descending
- tie-break 1: `test_macro_f1` descending
- tie-break 2: `val_accuracy` descending
- F2 preprocessing comparison should use the benchmark rank 1 model.
- F4/F5 should use benchmark top3 by default and benchmark top5 only as an optional extension.

## Full Ranking Table

| rank | model | val_macro_f1 | test_macro_f1 | val_accuracy | test_accuracy | preprocessing | training_mode | original_epoch_policy |
|---:|---|---:|---:|---:|---:|---|---|---:|
| 1 | ResNet18 | 0.9906 | 0.9807 | 0.9940 | 0.9880 | none | original_epoch | 200 |
| 2 | LeNet | 0.9887 | 0.9672 | 0.9940 | 0.9780 | none | original_epoch | 200 |
| 3 | ResNet101 | 0.9776 | 0.9230 | 0.9819 | 0.9500 | none | original_epoch | 200 |
| 4 | ResNet50 | 0.9773 | 0.9640 | 0.9798 | 0.9760 | none | original_epoch | 200 |
| 5 | ViT | 0.9707 | 0.9175 | 0.9738 | 0.9400 | none | original_epoch | 200 |
| 6 | MLP | 0.9471 | 0.8925 | 0.9496 | 0.9120 | none | original_epoch | 200 |
| 7 | GRU | 0.9196 | 0.8739 | 0.9294 | 0.9060 | none | original_epoch | 200 |
| 8 | LSTM | 0.9164 | 0.8748 | 0.9274 | 0.9120 | none | original_epoch | 200 |
| 9 | BiLSTM | 0.9063 | 0.8716 | 0.9274 | 0.9040 | none | original_epoch | 200 |
| 10 | RNN | 0.7322 | 0.6715 | 0.7440 | 0.7020 | none | original_epoch | 3000 |
| 11 | CNN+GRU | 0.1461 | 0.1611 | 0.3750 | 0.4100 | none | original_epoch | 200 |

## Unsupported Models

- none

