# Wi-Fi CSI-based Human Activity Recognition using UT-HAR

## 프로젝트 개요

이 프로젝트의 목표는 `UT-HAR` 데이터셋을 사용하여 Wi-Fi CSI 기반 `Human Activity Recognition` 성능을 체계적으로 분석하는 것이다. 단순히 여러 모델의 점수를 비교하는 것이 아니라, `real training data ratio`가 줄어드는 상황에서 성능이 얼마나 저하되는지, 그리고 preprocessing, `train-only augmentation`, 대표적인 sequence model 선택이 그 저하를 얼마나 줄일 수 있는지를 재현 가능하게 검증하는 실험 프레임워크를 만드는 것이 핵심이다.

## 프로젝트 핵심 문제의식

Wi-Fi CSI 기반 HAR는 `generalization failure` 때문에 실사용 확장이 어렵다. CSI는 `environment`, `subject`, `device position`, `room layout`, `noise` 변화에 민감하며, 새로운 환경마다 많은 `labeled data`를 다시 수집하는 것은 비용이 크다. 따라서 이 프로젝트는 "높은 정확도" 자체보다 "제한된 real labeled data에서 성능 저하를 얼마나 줄일 수 있는가"를 중심으로 다룬다.

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
- 라벨 범위: `0..6`
- 파일은 `.csv` 확장자를 가지지만 실제 내용은 NumPy binary이므로 `np.load`를 사용한다.

## 왜 4개 모델만 사용하는가

이 프로젝트는 leaderboard-style benchmark가 아니다. 실험 축이 이미 `model`, `real_ratio`, `augmentation`, `Accuracy`, `Macro F1`, `Weighted F1`, `Confusion matrix`로 충분히 많기 때문에, 모델 수를 과도하게 늘리면 학습 비용은 커지고 결과 해석은 약해진다. 따라서 본 프로젝트는 `CNN`, `GRU`, `LSTM`, `CNN+GRU` 네 모델만 사용한다.

## 성능 저하 최소화를 위한 설계 원칙

### A. Data preprocessing

- preprocessing은 low-data degradation minimization 전략의 일부다.
- normalization statistics는 train split에서만 계산한다.
- `validation/test` leakage를 허용하지 않는다.
- preprocessing choice는 controlled ablation으로 정당화한다.
- 선택된 preprocessing policy는 main experiments 전체에 일관되게 적용한다.

### B. Low-data sampling

- `stratified sampling`을 사용한다.
- class distribution을 가능한 한 유지한다.
- `validation/test`는 절대 줄이지 않는다.
- seed를 고정해 비교 가능성을 확보한다.

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
- `Macro F1`으로 class-balanced performance를 본다.
- `Weighted F1`으로 distribution-aware summary를 제공한다.
- `Confusion matrix`로 class-wise failure를 해석한다.

### F. Training control

- `early stopping`은 controlled experiment 모드에서만 사용한다.
- original-style 비교에서는 fixed epoch policy를 사용한다.
- 두 모드 모두 best validation checkpoint는 저장한다.

## Preprocessing Ablation 계획

preprocessing은 low-data setting에서 robustness에 영향을 줄 수 있으므로 임의로 정할 수 없다. 따라서 `none`, `train_global_zscore`, `per_sample_zscore`를 `GRU`, `real_ratio=0.25`, `augmentation=false`, `seed=42`, `validation Macro F1` 기준으로 비교하고, main experiments에서 사용할 preprocessing을 결정한다.

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
- 이 dry-run은 data loading, preprocessing, train/val/test loop, metrics saving, checkpoint saving, CPU/GPU fallback logic을 검증했다.
- 핵심 metric:
  - `val_macro_f1=0.7764`
  - `test_macro_f1=0.7456`
- 이 결과는 최종 성능 비교 결과가 아니라 pipeline validation result다.

### C. M2.5 preprocessing ablation 결과 요약

| preprocessing | val_macro_f1 | test_macro_f1 |
|---|---:|---:|
| `per_sample_zscore` | 0.8851 | 0.8351 |
| `train_global_zscore` | 0.8569 | 0.7983 |
| `none` | 0.6918 | 0.6305 |

- 이 결과는 preprocessing choice가 low-data setting 성능에 실제로 영향을 준다는 점을 보여준다.
- validation `Macro F1` 기준으로 `per_sample_zscore`를 main experiments용 preprocessing으로 선택한다.
- 다만 `per_sample_zscore`는 absolute amplitude information 일부를 제거할 수 있으므로, 이후 더 강한 controlled evidence가 나오지 않는 한 현재 선택을 유지한다.

## M3 Full-data Baseline 정책

M3 full-data baseline은 이제 두 가지 실행 정책을 가진다.

### 1. `original_epoch`

- original benchmark-style comparison을 위한 정책
- `training_mode = original_epoch`
- `early stopping` disabled
- model-specific epochs는 `original_baseline`의 UT-HAR policy를 그대로 따른다
- best checkpoint는 여전히 validation `Macro F1` 기준으로 선택한다

현재 매핑:

- `CNN` -> original `LeNet`: `200 epochs`
- `GRU` -> original `GRU`: `200 epochs`
- `LSTM` -> original `LSTM`: `200 epochs`
- `CNN_GRU` -> original `CNN+GRU`: `200 epochs`

### 2. `early_stopping`

- faster, controlled experiment용 정책
- `training_mode = early_stopping`
- `max_epochs`와 `patience`를 사용한다
- best checkpoint는 validation `Macro F1` 기준으로 선택한다

메모:
- earlier early-stopping baseline run은 exploratory result로 유지한다.
- original benchmark-style comparison과 혼합해서 해석하면 안 되며, 결과 파일도 분리해서 저장한다.

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
- [ ] `per_sample_zscore` 선택 근거를 validation `Macro F1`로 설명한다.

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
- [ ] original_epoch와 early_stopping 결과를 혼합하지 않고 분리해서 제시한다.

### G. Final presentation

- [ ] 발표 스토리를 `data scarcity -> preprocessing choice -> model robustness -> augmentation recovery -> best model -> class confusion` 순서로 유지한다.

## 구현 마일스톤

### M0. Environment and GPU setup

- [x] Python 환경 준비
- [x] CUDA availability 확인
- [x] GPU/CPU fallback 로직 확인

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

- 최신 상태:
- [x] `CNN`, `GRU`, `LSTM`, `CNN+GRU`를 동일한 full-data 조건에서 실행했다.
- [x] `real_ratio=1.0`
- [x] `augmentation=false`
- [x] main original-style comparison은 `training_mode=original_epoch`로 수행했다.
- [x] `original_epoch`에서는 `early stopping`을 사용하지 않았다.
- [x] model-specific epochs는 `original_baseline` 기준으로 매핑했다.
- [x] best validation checkpoint를 `Macro F1` 기준으로 저장했다.
- [x] `results/metrics/baseline_results_original_epoch.csv`를 생성했다.
- [x] zoomed baseline figure를 생성했다.

- 아래 미체크 항목은 초기 초안 메모이며 현재 상태를 나타내지 않는다.

- [ ] `CNN`, `GRU`, `LSTM`, `CNN+GRU`가 동일 조건에서 실행된다.
- [ ] `real_ratio=1.0`
- [ ] `augmentation=false`
- [ ] main original-style comparison은 `training_mode=original_epoch`
- [ ] `early stopping` disabled in `original_epoch`
- [ ] model-specific epochs copied from `original_baseline`
- [ ] best validation checkpoint 저장
- [ ] `results/metrics/baseline_results_original_epoch.csv` 생성
- [ ] `results/figures/baseline_original_epoch_macro_f1.png` 생성
- [ ] `results/figures/baseline_original_epoch_accuracy.png` 생성

메모:
- earlier `early_stopping` baseline run은 exploratory result로 분리 보관한다.
- 다음 본실험 비교는 `original_epoch` 기준으로 수행한다.

### M4. Low-data robustness

- 최신 상태:
- [x] `real_ratio=1.0, 0.5, 0.25, 0.1`를 모두 실행했다.
- [x] `real_ratio`는 train split에만 적용했다.
- [x] `validation/test`는 unchanged로 유지했다.
- [x] `stratified sampling`을 사용했다.
- [x] `controlled_generalization`을 사용했다.
- [x] `results/metrics/low_data_results.csv`를 생성했다.
- [x] low-data figure를 생성했다.
- [x] degradation metrics를 계산했다.
- [x] `real_ratio=0.1` 기준 핵심 해석을 report builder에 반영했다.

- 아래 미체크 항목은 초기 초안 메모이며 현재 상태를 나타내지 않는다.

- [ ] `real_ratio=1.0, 0.5, 0.25, 0.1`
- [ ] `stratified sampling` 사용
- [ ] `validation/test` unchanged
- [ ] default training mode는 `controlled_generalization`
- [ ] `original_epoch` 결과와 separate CSV/figure로 관리

### M5. Augmentation recovery

- 최신 상태:
- [ ] train-only augmentation implemented for M5
- [ ] `augmentation=true` runs completed
- [ ] `augmentation_results.csv` generated
- [ ] augmentation improvement figures generated
- [ ] augmentation recovery interpretation completed

- 아래 미체크 항목은 초기 초안 메모이며, M5는 아직 시작 전 상태다.

- [ ] augmentation은 training data에만 적용
- [ ] default training mode는 `controlled_generalization`
- [ ] `original_epoch` 결과와 separate CSV/figure로 관리

### M6. Confusion matrix and interpretation

- [ ] best model을 validation `Macro F1` 기준으로 선택

### M7. Report and presentation

- [ ] figure와 table 기반으로 보고서 작성
- [ ] limitations와 future work 정리

## Short Execution Plan

1. Stage 0 complete
2. M2 complete
3. M2.5 complete
4. M3 complete
5. M4 complete
6. next: M5 augmentation recovery with `training_mode=controlled_generalization`
7. then M6 confusion matrix and final analysis

## 시각화 주의사항

- baseline 성능이 `0.98~0.99`처럼 매우 높으면 `0~1` 전체 y-axis에서는 모델 간 차이가 거의 보이지 않는다.
- main report figure는 zoomed y-axis와 value label을 사용해 작은 차이도 읽을 수 있게 만든다.
- supplementary figure로 `1 - Macro F1`, `1 - Accuracy` 같은 gap plot을 추가해 near-1.0 구간 차이를 더 명확히 보여줄 수 있다.
- `Accuracy`와 `F1` 자체는 `0~1` bounded metric이므로 raw metric에 log scale을 직접 쓰지 않는 것이 기본 원칙이다.
- log scale이 필요하면 `1 - score` 같은 gap metric에만 제한적으로 적용하고, 반드시 `lower is better`를 명시한다.

## 시각화 및 보고서 작성 메모

- `UT-HAR`의 각 sample은 `time step x CSI feature` 형태의 2D matrix이므로 `sample_csi_heatmap.png`는 입력 구조를 직관적으로 보여주는 기본 시각화로 사용한다.
- 동시에 CSI는 time-series 성격도 가지므로 `sample_csi_lineplot.png`를 추가해 선택한 feature index의 시간축 변화를 waveform처럼 설명한다.
- full-data baseline 점수는 매우 높기 때문에 main comparison figure는 zoomed y-axis와 value label을 기본으로 사용한다.
- 차이가 너무 작아 보이면 supplementary figure로 `1 - score` gap plot을 사용한다. 이때 `lower is better`를 명확히 적는다.
- `Macro F1`는 class-wise degradation을 보기 위한 핵심 metric이므로 본문 figure와 해석에서 우선순위를 높게 둔다.
- figure regeneration은 `experiments/08_regenerate_figures.py`로 분리해 재현 가능하게 관리하고, full baseline training이 성공적으로 끝난 뒤 자동 호출할 수 있게 유지한다.

## Controlled Generalization Training Policy

- `original_epoch=200` 정책은 copied benchmark와의 original benchmark-style comparison을 위한 별도 모드로만 유지한다.
- 이후 `M4 low-data robustness`와 `M5 augmentation recovery`는 `200 epochs`를 기계적으로 반복하지 않고 `controlled_generalization` policy를 기본으로 사용한다.
- 목적은 validation score가 한두 epoch 흔들린다고 바로 멈추는 것이 아니라, 회복 가능성은 허용하면서도 과적합을 줄이는 것이다.
- main selection metric은 validation `Macro F1`로 고정한다.
- best checkpoint는 validation `Macro F1` 기준으로 저장한다.
- 최종 test evaluation은 마지막 epoch가 아니라 best validation checkpoint로 수행한다.
- early stopping은 `warmup_epochs`, `patience`, `min_delta`를 함께 사용한다.
- `warmup_epochs` 이전에는 early stopping을 적용하지 않는다.
- `patience`는 best score 이후 몇 epoch까지 회복을 기다릴지 정한다.
- `min_delta`는 매우 작은 fluctuation을 improvement로 오판하지 않게 해 준다.
- `max_epochs`는 상한선이지 반드시 끝까지 도는 고정 길이가 아니다.

권장 초기값:

- `max_epochs = 100`
- `warmup_epochs = 20`
- `patience = 15`
- `min_delta = 0.001`
- `batch_size = 64`, CUDA OOM이면 `32`
- `optimizer = Adam` 또는 `AdamW`
- `weight_decay = 1e-4`
- `gradient_clip_norm = 1.0`
- optional `scheduler = ReduceLROnPlateau` on validation `Macro F1`

low-data setting에서 이 정책이 더 적합한 이유:

- low-data validation score는 sample 수가 적어 fluctuation이 커질 수 있다.
- naive early stopping은 일시적 하락 구간에서 너무 빨리 멈출 수 있다.
- fixed `200 epochs`는 training set memorization과 overfitting 위험을 키운다.
- best-checkpoint selection은 모든 모델을 같은 기준으로 비교하게 해 준다.

## M4 Low-data Robustness 실행 원칙

- `M4`의 핵심 질문은 "real labeled training data가 줄어들 때 어떤 model이 가장 덜 무너지는가?"이다.
- `M4`는 `controlled_generalization`을 기본 training mode로 사용한다.
- `original_epoch=200`은 benchmark-style comparison용으로만 남겨 두고, low-data setting에서는 기본값으로 사용하지 않는다.
- 이유는 fixed `200 epochs`가 특히 low-data setting에서 overfitting을 키울 수 있기 때문이다.
- `real_ratio`는 train split에만 적용한다.
- `validation/test`는 full size를 그대로 유지한다.
- reduced train split은 `stratified sampling`으로 구성한다.
- 주요 저장 파일은 `results/metrics/low_data_results.csv`이다.
- 주요 시각화는 `results/figures/low_data_macro_f1.png`, `results/figures/low_data_accuracy.png`, `results/figures/low_data_degradation_macro_f1.png`, `results/figures/low_data_degradation_accuracy.png`이다.

degradation metric 정의:

- `macro_f1_drop = base_test_macro_f1 - test_macro_f1`
- `macro_f1_retention = test_macro_f1 / base_test_macro_f1`
- `accuracy_drop = base_test_accuracy - test_accuracy`
- `accuracy_retention = test_accuracy / base_test_accuracy`

M4 checklist:

- [ ] `real_ratio`는 train data에만 적용한다.
- [ ] `validation/test`는 unchanged 상태로 유지한다.
- [ ] `stratified sampling`을 사용한다.
- [ ] `controlled_generalization`을 사용한다.
- [ ] `results/metrics/low_data_results.csv`를 생성한다.
- [ ] low-data figure를 생성한다.
- [ ] smallest `Macro F1 drop` model을 식별한다.
- [ ] `real_ratio=0.1`에서 best `Macro F1` model을 식별한다.

## 현재 진행 상태

| Stage | Status | Main artifacts |
|---|---|---|
| Stage 0 | complete | `results/figures/class_distribution.png`, `results/figures/sample_csi_heatmap.png`, `results/figures/sample_csi_lineplot.png` |
| M2 | complete | `results/metrics/dry_run_results.csv` |
| M2.5 | complete | `results/metrics/preprocessing_ablation_results.csv` |
| M3 | complete | `results/metrics/baseline_results_original_epoch.csv` |
| M4 | complete | `results/metrics/low_data_results.csv`, `results/figures/low_data_macro_f1.png`, `results/figures/low_data_accuracy.png`, `results/figures/low_data_degradation_macro_f1.png`, `results/figures/low_data_degradation_accuracy.png` |
| M5 | planned, not run yet | augmentation recovery artifacts do not exist yet |
| M6 | planned, not run yet | confusion matrix / final analysis artifacts do not exist yet |

## 입력 window / CSI frame 해석

- one sample은 `250 timesteps x 90 CSI features` 구조를 가진다.
- 본 processed benchmark에서는 timestep을 `CSI frame index`로 해석한다.
- `90 CSI features`는 `Intel 5300 NIC` style CSI layout 관점에서 `30 subcarriers x 3 antenna pairs = 90 features`로 설명할 수 있다.
- 따라서 one sample에는 `250 CSI frames`가 포함된다.
- `real_ratio`는 selected training sample 수를 줄이는 동시에 total training `CSI frame` budget도 줄인다고 해석할 수 있다.
- 본 프로젝트에서는 `CSI frame count`를 엄밀한 양으로 사용한다.
- 시간 길이는 `sampling_rate`가 있어야 계산할 수 있으므로 가정 기반 estimate로만 사용한다.

공식:

- `sample_duration = 250 / fs` seconds
- `total_training_duration = selected_train_size x 250 / fs` seconds

가정 기반 예시:

- `fs=100Hz`라고 가정하면 one sample은 약 `2.5 seconds`
- full train: `3977 samples = 994,250 frames ≈ 9,942.5 seconds ≈ 165.7 minutes`
- `10% train`: `395 samples = 98,750 frames ≈ 987.5 seconds ≈ 16.5 minutes`

주의:

- 위 시간 환산은 `sampling_rate=100Hz`라는 예시적 가정일 뿐이며, 본 processed benchmark의 실제 `sampling_rate`를 확정하는 의미는 아니다.
- 따라서 `M4`의 `real_ratio`는 selected sample ratio이면서 동시에 selected `CSI frame budget` ratio로도 해석할 수 있지만, 보고서의 주 분석은 frame count와 performance degradation 기준으로 수행한다.
