# Wi-Fi CSI-based Human Activity Recognition using UT-HAR

## 프로젝트 개요

이 프로젝트의 목표는 `UT-HAR` 데이터셋을 사용하여 Wi-Fi CSI 기반 `Human Activity Recognition` 성능을 체계적으로 분석하는 것이다. 단순히 여러 모델의 점수를 비교하는 것이 아니라, `real training data ratio`가 줄어드는 상황에서 성능이 얼마나 저하되는지, 그리고 preprocessing, `train-only augmentation`, 대표적인 sequence model 선택이 그 저하를 얼마나 줄일 수 있는지를 재현 가능하게 검증하는 실험 프레임워크를 만드는 것이 핵심이다.

## 프로젝트 핵심 문제의식

Wi-Fi CSI 기반 HAR가 실사용으로 널리 확산되지 못한 가장 큰 이유 중 하나는 `generalization failure`다. CSI 신호는 `environment`, `subject`, `device position`, `room layout`, `noise` 변화에 매우 민감하기 때문에, 한 환경에서 학습한 모델이 다른 환경이나 새로운 조건에서 쉽게 성능 저하를 보일 수 있다. 따라서 이 프로젝트의 핵심 문제는 "높은 점수" 자체보다 "제한된 real labeled data에서 성능 저하를 얼마나 줄일 수 있는가"다.

## Dataset Summary

- 데이터셋: `UT-HAR`
- 입력 shape:
  - `X_train`: `(3977, 250, 90)`
  - `X_val`: `(496, 250, 90)`
  - `X_test`: `(500, 250, 90)`
- 라벨 shape:
  - `y_train`: `(3977,)`
  - `y_val`: `(496,)`
  - `y_test`: `(500,)`
- 문제 유형: `7-class classification`
- 라벨 범위: `0` through `6`
- 파일은 `.csv` 확장자를 가지지만 실제 내용은 NumPy binary이므로 `np.load`로 읽는다.

## 왜 4개 모델만 사용하는가

이 프로젝트는 leaderboard-style benchmark가 아니다. 이미 실험 축이 `model`, `real_ratio`, `augmentation`, `Accuracy`, `Macro F1`, `Weighted F1`, `Confusion matrix`로 충분히 많기 때문에, 모델 수를 과도하게 늘리면 학습 비용은 커지고 결과 해석은 약해진다. 따라서 본 프로젝트는 설명 가능성과 보고서 명확성을 위해 `CNN`, `GRU`, `LSTM`, `CNN+GRU` 네 모델만 사용한다.

## 성능 저하 최소화를 위한 설계 원칙

### A. Data preprocessing

- preprocessing은 low-data degradation minimization 전략의 일부다.
- normalization statistics는 반드시 train split에서만 계산한다.
- `validation/test` leakage를 허용하지 않는다.
- preprocessing choice는 임의로 정하지 않고 controlled ablation으로 정당화한다.
- 선택된 preprocessing policy는 main experiments 전체에 일관되게 적용한다.

### B. Low-data sampling

- `stratified sampling`을 사용한다.
- class distribution을 가능한 한 유지한다.
- `validation/test`는 절대 줄이지 않는다.
- seed를 고정하여 비교 가능성을 확보한다.

### C. Augmentation

- augmentation은 training data에만 적용한다.
- `validation/test`에는 적용하지 않는다.
- CSI-plausible augmentation만 사용한다.

### D. Model selection

- representative하고 explainable한 모델을 사용한다.
- `CNN-only`, `RNN-only`, `hybrid` 구조를 비교한다.
- best checkpoint는 validation `Macro F1` 기준으로 선택한다.

### E. Evaluation

- `Accuracy`만으로는 부족하다.
- `Macro F1`으로 class-balanced performance를 확인한다.
- `Weighted F1`으로 distribution-aware summary를 제공한다.
- `Confusion matrix`로 class-wise failure를 해석한다.
- 모든 metrics는 CSV로 저장하고 주요 결과는 figure로 저장한다.

### F. Training control

- `early stopping`을 사용한다.
- best validation checkpoint를 저장한다.
- `200 epochs` 장기 실험은 피한다.
- 짧은 dry-run 이후 통제된 `max_epochs=50` 범위로 확장한다.

## Preprocessing Ablation 계획

preprocessing은 low-data setting에서 robustness에 영향을 줄 수 있으므로 임의로 정할 수 없다. train-based normalization은 `validation/test leakage`를 막는 데 중요하지만, 어떤 정책이 가장 적절한지는 데이터와 모델에 따라 달라질 수 있다. 따라서 `none`, `train_global_zscore`, `per_sample_zscore`를 `GRU`, `real_ratio=0.25`, `augmentation=false`, `seed=42`, `validation Macro F1` 기준으로 비교하고, main experiments에서 사용할 preprocessing을 결정한다.

## 현재까지 완료된 실험 결과

### A. Stage 0 결과 요약

- `UT-HAR` data check가 성공적으로 완료되었다.
- 확인된 shape:
  - `X_train=(3977, 250, 90)`
  - `X_val=(496, 250, 90)`
  - `X_test=(500, 250, 90)`
  - labels `0..6`
- 생성된 figure:
  - `results/figures/class_distribution.png`
  - `results/figures/sample_csi_heatmap.png`

### B. M2 dry-run 결과 요약

- 실행 명령:
  - `python experiments/02_run_baseline.py --model GRU --epochs 3 --real-ratio 1.0 --augmentation false --preprocessing train_global_zscore --seed 42 --dry-run`
- 이 dry-run은 다음을 검증했다:
  - data loading
  - preprocessing
  - train/val/test loop
  - metrics saving
  - checkpoint saving
  - CPU/GPU fallback logic
- 핵심 metric:
  - `val_macro_f1=0.7764`
  - `test_macro_f1=0.7456`
- 이 결과는 최종 성능 비교 결과가 아니라, main experiment 전에 pipeline이 정상 동작함을 확인한 validation result다.

### C. M2.5 preprocessing ablation 결과 요약

- 실행 명령:
  - `python experiments/06_run_preprocessing_ablation.py --epochs 30 --real-ratio 0.25 --seed 42 --batch-size 64`
- 결과 표:

| preprocessing | val_macro_f1 | test_macro_f1 |
|---|---:|---:|
| `per_sample_zscore` | 0.8851 | 0.8351 |
| `train_global_zscore` | 0.8569 | 0.7983 |
| `none` | 0.6918 | 0.6305 |

- 이 결과는 preprocessing choice가 low-data setting 성능에 실제로 영향을 준다는 점을 보여준다.
- `per_sample_zscore`는 validation `Macro F1`가 가장 높았기 때문에 main experiments용 preprocessing으로 선택한다.
- 주의점:
  - `per_sample_zscore`는 absolute amplitude information 일부를 제거할 수 있다.
  - 그러나 현재 controlled setting에서는 가장 좋은 validation `Macro F1`를 보였다.
- 따라서 main experiments는 `preprocessing=per_sample_zscore`를 기본 정책으로 사용하며, 이후 더 강한 controlled evidence가 나오지 않는 한 이 선택을 유지한다.

## 평가항목 기반 상세 체크리스트

### A. Problem definition clarity

- [ ] 보고서에서 Wi-Fi CSI HAR의 `generalization failure` 문제를 명확히 설명한다.
- [ ] 제한된 real labeled data에서 성능 저하 최소화가 중심 문제임을 분명히 한다.
- [ ] `real_ratio` 감소 실험이 데이터 부족 상황 재현임을 설명한다.

### B. Dataset appropriateness and originality

- [ ] `UT-HAR`의 shape, split, label range를 구체적으로 기술한다.
- [ ] `.csv` 확장자이지만 `np.load`로 읽는 NumPy binary 파일이라는 점을 명확히 적는다.
- [ ] Stage 0 figure를 dataset understanding 근거로 사용한다.

### C. Data preprocessing faithfulness

- [ ] train-based normalization만 사용하고 `validation/test` leakage를 막는다.
- [ ] `stratified sampling`이 low-data split 구성에 사용됨을 설명한다.
- [ ] preprocessing ablation으로 전처리 선택이 임의적이지 않음을 보인다.
- [ ] `none`, `train_global_zscore`, `per_sample_zscore`를 controlled setting에서 비교한다.
- [ ] validation `Macro F1` 기준으로 main experiment preprocessing을 선택한다.
- [ ] `results/metrics/preprocessing_ablation_results.csv`와 `results/figures/preprocessing_ablation_macro_f1.png`를 보고서 근거로 사용한다.

### D. Model selection validity

- [ ] 4개 모델만 사용하는 이유를 해석 가능성과 실험 통제로 설명한다.
- [ ] `CNN`, `GRU`, `LSTM`, `CNN+GRU`의 역할을 구분해서 설명한다.

### E. Performance evaluation appropriateness

- [ ] `Accuracy`, `Macro F1`, `Weighted F1`를 함께 사용한다.
- [ ] best checkpoint는 validation `Macro F1` 기준으로 선택한다.
- [ ] test set은 최종 평가에만 사용한다.

### F. Report completeness

- [ ] Stage 0, M2, M2.5 결과를 method와 result 연결 근거로 사용한다.
- [ ] preprocessing ablation 결과가 main experiment 설계 결정으로 이어짐을 설명한다.
- [ ] figure와 CSV를 보고서/발표용 artifact로 재사용한다.

### G. Final presentation

- [ ] 발표 스토리를 `data scarcity -> preprocessing choice -> model robustness -> augmentation recovery -> best model -> class confusion` 순서로 유지한다.

## 구현 마일스톤

### M0. Environment and GPU setup

- [x] Python 환경 준비
- [x] CUDA availability 확인
- [x] GPU fallback/CPU fallback 로직 확인

### M1. Data check and EDA

- [x] `python experiments/01_check_data.py` 실행
- [x] shape, dtype, label range 검증
- [x] figure 생성 확인

### M2. GRU 3-epoch dry-run

- [x] GPU availability check 출력
- [x] `cuda if available else cpu` device 선택
- [x] `GRU` only
- [x] `real_ratio=1.0`
- [x] `augmentation=false`
- [x] `epochs=3`
- [x] `Accuracy`, `Macro F1`, `Weighted F1` 저장
- [x] `results/metrics/dry_run_results.csv` 생성
- [x] train/val/test loop 동작 확인
- [x] long training 미실행

### M2.5. Preprocessing ablation

- [x] `none`, `train_global_zscore`, `per_sample_zscore` 구현
- [x] `GRU`, `real_ratio=0.25`, `augmentation=false`, `seed=42` 비교
- [x] normalization statistics는 train split에서만 계산
- [x] `validation/test` leakage 방지 확인
- [x] validation `Macro F1` 기준 preprocessing 선택
- [x] `results/metrics/preprocessing_ablation_results.csv` 생성
- [x] `results/figures/preprocessing_ablation_macro_f1.png` 생성
- [x] 선택된 preprocessing을 이후 main experiments에 고정

### M3. Full-data baseline

- [ ] `CNN`, `GRU`, `LSTM`, `CNN+GRU`가 동일 조건에서 실행된다.
- [ ] `real_ratio=1.0`
- [ ] `augmentation=false`
- [ ] `max_epochs=50`
- [ ] `early stopping` enabled
- [ ] best validation checkpoint 저장
- [ ] `results/metrics/baseline_results.csv` 생성
- [ ] baseline figure 생성

### M4. Low-data robustness

- [ ] `real_ratio=1.0, 0.5, 0.25, 0.1`
- [ ] `stratified sampling` 사용
- [ ] `validation/test` unchanged
- [ ] `results/metrics/low_data_results.csv` 생성

### M5. Augmentation recovery

- [ ] augmentation은 training data에만 적용
- [ ] `results/metrics/augmentation_results.csv` 생성

### M6. Confusion matrix and interpretation

- [ ] best model을 validation `Macro F1` 기준으로 선택
- [ ] `results/figures/confusion_matrix_best.png` 생성

### M7. Report and presentation

- [ ] figure와 table 기반으로 보고서 작성
- [ ] limitations와 future work 정리

메모:
- 다음 단계는 `M3 full-data baseline`이다.

## Short Execution Plan

1. Stage 0 complete
2. M2 complete
3. M2.5 complete
4. next: M3 full-data baseline
5. then M4 low-data robustness
6. then M5 augmentation recovery
