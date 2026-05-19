# Wi-Fi CSI-based Human Activity Recognition using UT-HAR

## 프로젝트 개요

이 프로젝트의 목표는 `UT-HAR` 데이터셋을 사용하여 Wi-Fi CSI 기반 `Human Activity Recognition` 성능을 체계적으로 분석하는 것이다. 단순히 여러 모델의 점수를 비교하는 것이 아니라, `real training data ratio`가 줄어드는 상황에서 성능이 얼마나 저하되는지, 그리고 `train-only augmentation`, preprocessing, 대표적인 sequence model 선택이 그 저하를 얼마나 줄일 수 있는지를 재현 가능하게 검증하는 실험 프레임워크를 만드는 것이 핵심이다.

## 프로젝트 핵심 문제의식

Wi-Fi CSI 기반 HAR가 실사용으로 널리 확산되지 못한 가장 큰 이유 중 하나는 `generalization failure`다. CSI 신호는 `environment`, `subject`, `device position`, `room layout`, `noise` 변화에 매우 민감하기 때문에, 한 환경에서 학습한 모델이 다른 환경이나 새로운 조건에서 쉽게 성능 저하를 보일 수 있다.

이 문제는 데이터 수집 비용과 직접 연결된다. 실용적인 시스템을 만들려면 새로운 공간이나 배치마다 많은 `labeled data`를 다시 수집해야 하는데, 이는 시간과 인력 비용이 크다. 따라서 실제 응용에서는 "충분한 real labeled data를 항상 확보할 수 있다"는 가정이 약하다. 더 중요한 질문은, `real labeled data`가 적은 상황에서도 성능 저하를 얼마나 작게 유지할 수 있는가이다.

본 프로젝트는 이 문제를 `UT-HAR`에서 작고 통제된 실험으로 재현한다. `train` 데이터를 `100%`, `50%`, `25%`, `10%`로 의도적으로 줄여 `low-data setting`을 만들고, 이때 성능이 얼마나 감소하는지 관찰한다. 이후 `augmentation`과 `model selection`뿐 아니라 preprocessing choice가 그 성능 저하를 얼마나 완화하는지 분석한다.

## Problem Definition

본 과제는 7개 활동 클래스를 구분하는 `7-class classification` 문제다. 여기서 중요한 것은 full-data 성능의 절대값만이 아니다. 더 중요한 것은 `real_ratio` 감소에 따라 성능이 얼마나 무너지는지, 그리고 preprocessing, augmentation, model choice, evaluation design을 통해 그 성능 저하를 얼마나 줄일 수 있는지다. 따라서 최종 보고서는 "최고 점수"보다 "저하 양상, 회복 정도, 클래스별 실패 패턴"을 중심으로 서술해야 한다.

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
- 파일 형식 주의:
  - 파일 확장자는 `.csv`지만 실제 내용은 NumPy binary 형식이므로 `np.load`로 읽어야 한다.

## Research Questions

1. `real training data ratio`가 줄어들수록 모델 성능은 얼마나 감소하는가?
2. `train-only augmentation`이 low-data setting에서 성능을 얼마나 회복시키는가?
3. augmentation 이후 어떤 모델이 가장 좋은 성능을 보이는가?
4. 어떤 모델이 low-data setting에서 가장 작은 성능 저하를 보이는가?
5. 어떤 모델이 augmentation으로 가장 큰 성능 향상을 얻는가?
6. 최고 성능 모델의 `confusion matrix`에서 어떤 활동 클래스들이 주로 혼동되는가?

## 왜 4개 모델만 사용하는가

이 프로젝트는 leaderboard-style benchmark가 아니다. 목표는 가능한 모든 모델을 돌려서 최고 점수를 찾는 것이 아니라, `low-data`와 `augmentation` 조건에서 해석 가능한 비교를 수행하는 것이다. 이미 실험 축이 `model`, `real_ratio`, `augmentation on/off`, `Accuracy`, `Macro F1`, `Weighted F1`, `Confusion matrix`로 충분히 많기 때문에, 모델 수를 과도하게 늘리면 학습 비용이 커지고 결과 해석은 오히려 약해진다.

본 프로젝트는 설명 가능성과 보고서 명확성을 위해 4개 모델만 사용한다.

- `CNN` / `LeNet` 스타일 `CNN`
  - local time-subcarrier pattern baseline
- `GRU`
  - lightweight sequence baseline
- `LSTM`
  - representative recurrent sequence baseline
- `CNN+GRU`
  - local feature extraction + temporal modeling hybrid

이 구성이 의미 있는 이유는 `CNN-only`, `RNN-only`, `hybrid` 구조를 모두 포함하면서도 각 모델의 역할이 분명하기 때문이다. 이는 평가 rubric의 `model selection validity`에도 잘 맞는다. `Transformer`나 더 큰 모델은 향후 확장 가능성으로 언급할 수 있지만, 본 프로젝트의 main scope에는 넣지 않는다.

## 성능 저하 최소화를 위한 설계 원칙

### A. Data preprocessing

- preprocessing은 단순한 기술적 전처리가 아니라 low-data degradation minimization 전략의 일부다.
- normalization은 반드시 train split 기준으로만 계산한다.
- 모든 normalization statistics는 `validation/test` leakage 없이 계산해야 한다.
- 입력 shape handling은 모델별로 일관되게 유지한다.
- 필요하면 normalization parameter를 저장하여 재현성을 확보한다.
- preprocessing choice는 가정으로 정하지 않고 `Preprocessing Ablation`으로 정당화한다.
- 최종 선택된 preprocessing policy는 이후 main experiments 전체에 일관되게 적용한다.
- faithful preprocessing 자체가 rubric의 중요한 평가 항목이므로, 코드와 보고서에서 구체적으로 설명 가능해야 한다.

### B. Low-data sampling

- `stratified sampling`을 사용한다.
- class distribution을 가능한 한 유지한다.
- `random seed`를 고정한다.
- `validation/test`는 절대 줄이지 않는다.
- 모든 모델은 동일한 sampled split에서 비교하여 공정성을 확보한다.

### C. Augmentation

- augmentation은 training data에만 적용한다.
- `validation/test`에는 augmentation을 절대 적용하지 않는다.
- CSI-plausible augmentation만 사용한다:
  - `Gaussian noise`
  - `time shift` / `jitter`
  - `scaling`
  - `time masking`
  - `subcarrier masking`
- augmentation은 "fake test improvement"를 만들기 위한 것이 아니라, real data가 부족할 때 robustness를 높이기 위한 설계다.

### D. Model selection

- representative하고 explainable한 모델을 우선한다.
- `CNN-only`, `RNN-only`, `hybrid` 구조를 비교한다.
- 모델 목록을 과도하게 늘리지 않는다.
- best checkpoint selection은 validation `Macro F1` 기준으로 수행한다.

### E. Evaluation

- `Accuracy`만으로는 부족하다.
- `Macro F1`으로 class-balanced performance를 확인한다.
- `Weighted F1`으로 distribution-aware summary를 제공한다.
- `Confusion matrix`로 class-wise failure를 해석한다.
- 모든 metrics는 CSV로 저장한다.
- 모든 핵심 결과는 report-ready figure로 생성한다.

### F. Training control

- `early stopping`을 사용한다.
- best validation checkpoint를 저장한다.
- 무조건 `200 epochs`를 돌리지 않는다.
- 시작은 `GRU 3-epoch dry-run`으로 한다.
- 이후에도 통제된 `max_epochs=50` 실험으로 확장한다.

## Preprocessing Ablation 계획

preprocessing은 이 프로젝트에서 단순한 보조 단계가 아니다. real labeled data가 적을수록 입력 scale, sample-level amplitude variation, 환경 변화에 의한 분포 흔들림이 모델 성능에 더 크게 반영될 수 있기 때문에, preprocessing choice 자체가 low-data generalization에 영향을 줄 수 있다. 따라서 preprocessing은 "한 번 정해진 기술적 선택"이 아니라, 성능 저하 최소화 전략의 일부로 다뤄야 한다.

특히 train-based normalization은 `validation/test leakage`를 막는 데 중요하다. `X_train`에서 계산한 통계만 `train/val/test`에 공통 적용해야, 모델이 평가 split의 분포 정보를 미리 보는 문제를 피할 수 있다. 하지만 어떤 normalization policy가 low-data setting에서 가장 적절한지는 데이터와 모델에 따라 달라질 수 있으므로, 처음부터 하나를 정답처럼 가정하지 않고 작은 controlled experiment로 비교한다.

비교할 preprocessing 후보는 다음 3개다.

1. `none` / `raw`
   - 입력값을 normalization 없이 그대로 사용한다.
   - preprocessing이 왜 필요한지 보여주는 baseline 역할을 한다.
2. `train_global_zscore`
   - `X_train`에서만 mean/std를 계산한다.
   - train-derived mean/std를 `train/val/test` 전체에 공통 적용한다.
   - leakage를 피하면서 안정적인 scaling을 제공하는 기본 후보다.
3. `per_sample_zscore`
   - 각 sample을 자체 time-subcarrier 값 기준으로 독립 정규화한다.
   - `environment`나 `device` 차이에서 오는 sample-level amplitude variation을 줄일 가능성이 있다.
   - 반대로 절대 amplitude 정보까지 지울 수 있으므로, 무조건 좋다고 가정하지 않고 평가가 필요하다.

controlled setting은 다음과 같이 고정한다.

- `model = GRU`
- `real_ratio = 0.25`
- `augmentation = false`
- `seed = 42`
- `max_epochs = 30 or 50`
- `selection metric = validation Macro F1`

예상 산출물은 다음과 같다.

- `results/metrics/preprocessing_ablation_results.csv`
- `results/figures/preprocessing_ablation_macro_f1.png`

이 결과는 이후 main experiments 전체에서 사용할 preprocessing policy를 하나 선택하는 근거로 사용한다. 선택된 preprocessing은 full-data baseline, low-data robustness, augmentation recovery 실험에 일관되게 고정하며, 그 선택 이유를 보고서에 명시한다. 이를 통해 rubric의 `데이터 전처리의 충실성` 항목을 더 강하게 뒷받침한다.

## Experiment Stages

### Stage 0: data check and EDA

- 데이터 파일 존재 여부, shape, dtype, min/max, label range를 검증한다.
- 클래스 분포와 예시 CSI heatmap을 시각화한다.
- 이 단계는 dataset analysis와 preprocessing reliability를 확보하기 위한 출발점이다.

### Stage 1: GRU dry-run

- `GRU` 1개 모델만 사용해 train/val/test loop, device 선택, metrics 저장 경로를 짧게 검증한다.

### Stage 1.5: preprocessing ablation

- `GRU`, `real_ratio=0.25`, `augmentation=false` 조건에서 preprocessing 후보를 비교한다.
- 이후 main experiment 전체에 적용할 preprocessing policy를 선택한다.

### Stage 2: full-data baseline

- 전체 `train` split을 사용하여 `CNN`, `GRU`, `LSTM`, `CNN+GRU`를 동일한 조건으로 비교한다.

### Stage 3: low-data robustness

- `train` split에서 `stratified sampling`으로 `100%`, `50%`, `25%`, `10%` 비율만 사용한다.
- 실험 목적은 실제 데이터 부족 상황을 재현하고, 성능 저하 폭을 정량화하는 것이다.

### Stage 4: augmentation recovery

- `real_ratio=0.5`, `0.25`, `0.1`에서 `train-only augmentation`을 적용한다.
- augmentation 전후 성능 차이로 recovery 효과를 측정한다.

### Stage 5: best model and confusion matrix

- validation `Macro F1` 기준으로 최고 모델을 선정한다.
- test set에서 최종 평가를 1회 수행한다.
- `confusion matrix`와 `classification report`를 저장하고 클래스별 혼동을 해석한다.

### Stage 6 optional: augmentation ratio or window-size experiment

- core experiment가 모두 안정화된 경우에만 augmentation 강도나 `window-size` 관련 추가 분석을 수행한다.

## Required Metrics

- `Accuracy`
- `Macro F1`
- `Weighted F1`
- `Confusion matrix`
- `Classification report`

## Required Result Files

- `results/metrics/dry_run_results.csv`
- `results/metrics/preprocessing_ablation_results.csv`
- `results/metrics/baseline_results.csv`
- `results/metrics/low_data_results.csv`
- `results/metrics/augmentation_results.csv`
- `results/metrics/classification_report_best.txt`
- `results/figures/class_distribution.png`
- `results/figures/sample_csi_heatmap.png`
- `results/figures/preprocessing_ablation_macro_f1.png`
- `results/figures/low_data_macro_f1.png`
- `results/figures/augmentation_improvement.png`
- `results/figures/confusion_matrix_best.png`

## Code Design Principles

- `original_baseline`은 비교용으로만 유지하고 새 실험 코드와 분리한다.
- 새 실험 코드는 clean, reproducible, report-friendly 구조를 따른다.
- 핵심 스토리는 low-data generalization과 performance degradation minimization이다.
- preprocessing choice도 main experiment 설계의 일부로 다룬다.
- augmentation은 `train-only augmentation` 원칙을 지킨다.
- normalization parameter는 반드시 train split에서만 계산한다.
- training ratio를 줄일 때는 `stratified sampling`을 사용한다.
- 모든 실험은 `random seed`를 고정한다.
- 모든 metric은 CSV로 저장한다.
- 보고서와 발표용 figure를 파일로 저장한다.
- `early stopping`과 best validation checkpoint 저장을 기본 정책으로 둔다.

## 평가항목 기반 상세 체크리스트

### A. Problem definition clarity

- [ ] 보고서에서 Wi-Fi CSI HAR의 `generalization failure` 문제를 명확히 설명한다.
- [ ] `environment`, `subject`, `device position`, `room layout`, `noise` 변화가 CSI generalization을 어렵게 만든다는 점을 설명한다.
- [ ] 환경마다 많은 `labeled data`를 다시 수집하기 어렵다는 실용적 제약을 적는다.
- [ ] 본 프로젝트의 중심 스토리가 "제한된 real data에서 성능 저하 최소화"임을 분명히 한다.
- [ ] `real_ratio` 감소 실험이 "데이터 부족 상황 재현"임을 설명한다.
- [ ] 연구 질문이 `performance degradation`, `augmentation recovery`, `best model`, `smallest degradation model`, `largest improvement model`과 직접 연결되도록 쓴다.
- [ ] `docs/project_plan.md`, `README.md`, `results/metrics/*.csv`, `results/figures/*.png`가 같은 서사를 공유하는지 확인한다.

### B. Dataset appropriateness and originality

- [ ] `UT-HAR`의 shape, split, label range를 구체적으로 기술한다.
- [ ] `.csv` 확장자이지만 `np.load`로 읽는 NumPy binary 파일이라는 점을 명확히 적는다.
- [ ] `UT-HAR`가 CSI time-series data이며 7-class HAR 문제에 적절하다는 점을 설명한다.
- [ ] `MNIST`/`CIFAR` 같은 일반 이미지 데이터가 아니라 domain-specific CSI 데이터라는 점을 강조한다.
- [ ] `experiments/01_check_data.py`가 dataset verification artifact로 사용된다는 점을 연결한다.
- [ ] `results/figures/class_distribution.png`와 `results/figures/sample_csi_heatmap.png`를 데이터 이해를 위한 핵심 시각화로 사용한다.

### C. Data preprocessing faithfulness

- [ ] 6개 원본 파일을 `np.load`로 로드한다고 명시한다.
- [ ] `X`를 `(N, 250, 90)`에서 model-compatible tensor로 바꾸는 규칙을 일관되게 유지한다.
- [ ] train-based normalization만 사용하고 `validation/test` leakage를 막는다.
- [ ] 필요 시 preprocessing parameter 저장 정책을 정의한다.
- [ ] `stratified sampling`으로 low-data split을 구성한다.
- [ ] 모든 모델이 동일한 sampled split을 사용하도록 설계한다.
- [ ] `validation/test`는 절대 축소하거나 재샘플링하지 않는다.
- [ ] faithful preprocessing이 rubric의 핵심 항목임을 보고서에서 분명히 한다.
- [ ] preprocessing ablation을 통해 전처리 선택이 임의적이지 않음을 보인다.
- [ ] `none`, `train_global_zscore`, `per_sample_zscore`를 controlled setting에서 비교한다.
- [ ] validation `Macro F1` 기준으로 main experiment에서 사용할 preprocessing을 선택한다.
- [ ] 선택된 preprocessing이 `low-data generalization`과 어떤 관련이 있는지 보고서에서 설명한다.
- [ ] `results/metrics/preprocessing_ablation_results.csv`와 `results/figures/preprocessing_ablation_macro_f1.png`를 보고서 근거로 사용한다.

### D. Model selection validity

- [ ] 이 프로젝트가 "많은 모델을 돌리는 benchmark"가 아님을 분명히 한다.
- [ ] 4개 모델만 쓰는 이유가 해석 가능성과 실험 통제에 있음을 설명한다.
- [ ] `CNN` / `LeNet`을 local time-subcarrier pattern baseline으로 설명한다.
- [ ] `GRU`를 lightweight sequence baseline으로 설명한다.
- [ ] `LSTM`을 representative recurrent baseline으로 설명한다.
- [ ] `CNN+GRU`를 local feature extraction + temporal modeling hybrid로 설명한다.
- [ ] 모델 수를 과도하게 늘리면 training cost가 커지고 결과 해석이 약해진다는 점을 적는다.
- [ ] `Transformer`와 더 큰 모델은 future work로만 언급한다.

### E. Performance evaluation appropriateness

- [ ] `Accuracy` alone is insufficient라는 점을 보고서에서 설명한다.
- [ ] `Macro F1`을 사용하여 class-wise degradation을 확인한다.
- [ ] `Weighted F1`을 사용하여 distribution-aware summary를 제공한다.
- [ ] `Confusion matrix`를 사용하여 class-wise failure를 해석한다.
- [ ] `classification report`를 저장하고 보고서 해석에 연결한다.
- [ ] 모든 metrics를 CSV로 저장한다.
- [ ] validation `Macro F1` 기준으로 best checkpoint를 선택한다.
- [ ] test set은 최종 평가에만 사용한다.
- [ ] `results/figures/confusion_matrix_best.png`를 발표용 핵심 그림으로 준비한다.

### F. Report completeness

- [ ] low-data generalization이 보고서의 중심 서사인지 확인한다.
- [ ] 성능 저하 최소화라는 문제의식이 introduction, method, results에 일관되게 반영되는지 확인한다.
- [ ] preprocessing ablation 결과가 main experiment 설계 근거로 연결되는지 확인한다.
- [ ] `preprocessing_ablation_macro_f1.png`를 사용해 preprocessing choice를 정당화한다.
- [ ] `low_data_macro_f1.png`를 사용해 `real_ratio` 감소에 따른 성능 저하를 시각화한다.
- [ ] `augmentation_improvement.png`를 사용해 augmentation recovery를 시각화한다.
- [ ] `confusion_matrix_best.png`를 사용해 클래스별 실패를 해석한다.
- [ ] 보고서에 limitations를 포함한다:
  - single dataset
  - environment dependency
  - simple augmentations
  - limited model search
- [ ] 보고서에 future work를 포함한다:
  - multi-dataset validation
  - cross-domain evaluation
  - more advanced augmentation
  - larger models such as `Transformer`

### G. Final presentation

- [ ] 발표 스토리를 `data scarcity -> preprocessing choice -> model robustness -> augmentation recovery -> best model -> class confusion` 순서로 유지한다.
- [ ] figure 중심 발표를 구성하고 표는 보조 자료로 사용한다.
- [ ] `Macro F1`와 `Confusion matrix`가 발표에서도 핵심 지표임을 유지한다.
- [ ] optional experiment가 core message를 방해하지 않도록 제한한다.
- [ ] 발표용 시각화가 보고서 figure와 같은 메시지를 전달하는지 확인한다.

## 구현 마일스톤

### M0. Environment and GPU setup

- [ ] Python `3.10.11` `venv` 생성
- [ ] CUDA-enabled PyTorch 설치
- [ ] `torch.cuda.is_available()` 확인
- [ ] `torch` version과 CUDA availability 기록
- [ ] GPU가 확인되기 전까지는 CUDA 사용을 가정하지 않기

### M1. Data check and EDA

- [x] `python experiments/01_check_data.py` 실행
- [x] shape, dtype, min/max, label range 검증
- [x] `class_distribution.png` 생성 확인
- [x] `sample_csi_heatmap.png` 생성 확인

### M2. GRU 3-epoch dry-run

- [ ] GPU availability check가 출력된다.
- [ ] device가 `cuda if available else cpu`로 선택된다.
- [ ] `GRU`만 사용한다.
- [ ] `real_ratio=1.0`
- [ ] `augmentation=false`
- [ ] `epochs=3`
- [ ] `Accuracy`, `Macro F1`, `Weighted F1`가 저장된다.
- [ ] `results/metrics/dry_run_results.csv`가 생성된다.
- [ ] train/val/test loop가 정상 동작한다.
- [ ] long training은 실행하지 않는다.

### M2.5. Preprocessing ablation

- [ ] `none`, `train_global_zscore`, `per_sample_zscore` 옵션을 구현한다.
- [ ] `GRU`, `real_ratio=0.25`, `augmentation=false`, `seed=42` 조건으로 비교한다.
- [ ] normalization statistics는 `train` split에서만 계산한다.
- [ ] `validation/test` leakage가 없도록 검증한다.
- [ ] validation `Macro F1` 기준으로 preprocessing policy를 선택한다.
- [ ] `results/metrics/preprocessing_ablation_results.csv`를 생성한다.
- [ ] `results/figures/preprocessing_ablation_macro_f1.png`를 생성한다.
- [ ] 선택된 preprocessing을 이후 main experiments에 고정한다.

### M3. Full-data baseline

- [ ] `CNN`, `GRU`, `LSTM`, `CNN+GRU`가 동일 조건에서 실행된다.
- [ ] `real_ratio=1.0`
- [ ] `augmentation=false`
- [ ] `max_epochs=50`
- [ ] `early stopping` enabled
- [ ] best validation checkpoint 저장
- [ ] `results/metrics/baseline_results.csv` 생성

### M4. Low-data robustness

- [ ] `real_ratio=1.0, 0.5, 0.25, 0.1`
- [ ] `stratified sampling` 사용
- [ ] 같은 split seed를 모델 간 공통으로 사용
- [ ] `validation/test` unchanged
- [ ] `100%` 대비 performance drop 계산
- [ ] `results/metrics/low_data_results.csv` 생성
- [ ] `results/figures/low_data_macro_f1.png` 생성

### M5. Augmentation recovery

- [ ] augmentation은 training data에만 적용
- [ ] `real_ratio=0.5, 0.25, 0.1`
- [ ] no-aug vs aug 비교 저장
- [ ] `improvement = Aug Macro F1 - No Aug Macro F1` 계산
- [ ] augmentation 이후 최고 모델 식별
- [ ] augmentation improvement가 가장 큰 모델 식별
- [ ] `results/metrics/augmentation_results.csv` 생성
- [ ] `results/figures/augmentation_improvement.png` 생성

### M6. Confusion matrix and interpretation

- [ ] best model을 validation `Macro F1` 기준으로 선택
- [ ] final test evaluation은 1회만 수행
- [ ] `results/figures/confusion_matrix_best.png` 생성
- [ ] `results/metrics/classification_report_best.txt` 생성
- [ ] confused classes를 보고서에서 해석

### M7. Report and presentation

- [ ] 저장된 figure와 table 기반으로 보고서 작성
- [ ] limitations와 future work 정리
- [ ] 평가 rubric 순서에 맞춰 발표 자료 구성

## 다음 코드 작업 전 검토 기준

- [ ] `docs/project_plan.md`가 프로젝트 motivation을 명확히 설명한다.
- [ ] 4-model scope가 충분히 정당화되어 있다.
- [ ] low-data generalization이 프로젝트의 central story다.
- [ ] preprocessing ablation이 main experiment 설계에 반영되어 있다.
- [ ] 모든 핵심 실험이 CSV와 figure를 생성하도록 설계되어 있다.
- [ ] core experiment가 안정화되기 전 optional experiment를 구현하지 않는다.
- [ ] 다음 구현 목표가 `GRU 3-epoch dry-run`으로 명확하다.

## Short Execution Plan

1. Stage 0 data check를 유지한다.
2. M2 `GRU 3-epoch dry-run`을 구현한다.
3. M2.5 preprocessing ablation을 수행한다.
4. M3 full-data baseline을 실행한다.
5. M4 low-data robustness를 분석한다.
6. M5 augmentation recovery를 분석한다.
7. M6 best model / `confusion matrix`를 정리한다.
8. M7 report / presentation을 마무리한다.
