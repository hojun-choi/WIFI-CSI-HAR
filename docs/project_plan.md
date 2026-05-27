# Wi-Fi CSI HAR Final Experiment Plan

이전 개발 과정에서 생성된 prototype artifacts는 정리되었고, 최종 실험은 아래 official F1-F5.1 workflow 기준으로 관리한다.

## 1. Problem Definition

Wi-Fi CSI 기반 HAR는 행동 인식에 활용될 수 있지만, labeled data가 부족하거나 환경이 바뀌면 일반화가 어려워질 수 있다. 본 프로젝트는 `UT-HAR`를 사용하여 다음 순서로 문제를 검증한다.

- benchmark model comparison
- preprocessing comparison and final selection
- low-data robustness
- augmentation recovery
- augmentation add ratio ablation

핵심 질문은 다음과 같다.

- 어떤 모델과 preprocessing 조합이 UT-HAR에서 가장 안정적인가
- train data가 줄어들어도 성능을 어느 정도 유지할 수 있는가
- synthetic augmentation이 low-data 상황을 실제로 보완하는가
- synthetic sample 비율은 얼마나 사용하는 것이 적절한가

## 2. Dataset and Input Window

- Dataset: `UT-HAR`
- Input array shape in this repo: `(N, 250, 90)`
- One sample shape: `(250, 90)`
- `250 timesteps`는 초 단위 시간이 아니라 `CSI frame index`로 해석한다.
- `90 CSI features`는 benchmark 문맥상 `30 subcarriers x 3 antenna pairs`로 볼 수 있다.
- 시간 환산에는 `sampling_rate`가 필요하다.
- `sampling_rate = fs` Hz라면 sample duration은 `250 / fs` seconds이다.
- `100Hz` 환산은 illustrative assumption일 뿐이며 confirmed ground truth로 사용하지 않는다.

UT-HAR class names:

- `lie down`
- `fall`
- `walk`
- `pickup`
- `run`
- `sit down`
- `stand up`

Report-facing visualization은 numeric label 대신 activity name을 사용한다.

## 3. Official Final Workflow

### F1. Original Benchmark Full Run

목적:

- `UT_HAR_data`에 대해 original benchmark model set을 최대한 그대로 실행한다.
- official final workflow의 시작점으로 사용한다.
- benchmark rank 1, top3, top5를 validation `Macro F1` 기준으로 선정한다.

규칙:

- `preprocessing = none/raw`
- `training_mode = original_epoch`
- no early stopping
- `real_ratio = 1.0`
- `augmentation = false`
- selection metric = validation `Macro F1`
- test `Macro F1` is confirmation only

output:

- `results/metrics/final_benchmark_results.csv`
- `docs/final_benchmark_selection.md`

### F2. Single Preprocessing Comparison

목적:

- F1 benchmark rank 1 model 하나를 사용해 preprocessing 후보를 one-by-one 비교한다.

규칙:

- F2부터는 `controlled_generalization`을 사용한다.
- selection metric = validation `Macro F1`
- test `Macro F1` is confirmation only
- single-method comparison을 combination comparison보다 먼저 수행한다.

output:

- `results/metrics/final_preprocessing_results.csv`

### F3. Preprocessing Combination and Final Selection

목적:

- F2에서 유망한 combination만 추가 비교한다.
- close candidate가 있을 경우 multi-seed stability check로 final preprocessing을 고정한다.

현재 확정 결과:

- final preprocessing: `moving_average_smoothing+minmax_scaling`
- selected by mean validation `Macro F1` across seeds
- test `Macro F1` is confirmation only

output:

- `results/metrics/final_preprocessing_combination_results.csv`
- `results/metrics/final_preprocessing_stability_summary.csv`
- `docs/final_preprocessing_decision.md`

### F4. Low-data Robustness

목적:

- final preprocessing을 고정한 상태에서 labeled train data 감소에 따른 성능 저하를 측정한다.

규칙:

- preprocessing = `moving_average_smoothing+minmax_scaling`
- benchmark top3 models 기본 사용: `ResNet18`, `LeNet`, `ResNet101`
- `real_ratio = 1.0, 0.5, 0.25, 0.1`
- `augmentation = false`
- `training_mode = controlled_generalization`
- validation/test unchanged
- train split만 deterministic stratified sampling으로 축소

output:

- `results/metrics/final_low_data_results.csv`
- `results/figures/final_low_data_macro_f1_by_ratio.png`
- `results/figures/final_low_data_accuracy_by_ratio.png`
- `results/figures/final_low_data_macro_f1_retention_by_ratio.png`
- `results/figures/final_low_data_macro_f1_drop_by_ratio.png`

figure semantics:

- `final_low_data_macro_f1_by_ratio.png` shows raw test `Macro F1`
- `final_low_data_macro_f1_retention_by_ratio.png` shows normalized retention relative to each model's own `real_ratio=1.0` baseline
- in the retention plot, `100% = 1.0` by definition

핵심 결과:

- `ResNet18` is the strongest practical low-data model.
- `real_ratio=0.25`에서 `ResNet18`은 test `Macro F1 = 0.9008`을 기록했다.

### F5. Augmentation Recovery

목적:

- low-data 조건에서 synthetic augmentation이 F4 no-augmentation baseline 대비 성능을 회복하는지 평가한다.

규칙:

- `augmentation_mode = offline_append`
- preprocessing = `moving_average_smoothing+minmax_scaling`
- benchmark top3 model set 사용
- `real_ratio = 0.5, 0.25, 0.1`
- validation/test는 never augmented
- same-`real_ratio` rule:
  augmentation 결과는 반드시 같은 model, 같은 `real_ratio`, 같은 preprocessing의 F4 baseline과만 비교한다.

output:

- `results/metrics/final_augmentation_results.csv`
- `results/figures/final_augmentation_gain_macro_f1_by_ratio.png`
- `results/figures/final_augmentation_gain_accuracy_by_ratio.png`
- `results/figures/final_augmentation_macro_f1_aug_vs_no_aug.png`
- `results/figures/final_augmentation_25_10_summary.png`
- `results/figures/final_augmentation_gain_heatmap.png`

요약:

- `augmentation_add_ratio=1.0` 기준 official F5는 일부 조건에서 회복 효과를 보였지만, 모든 모델/ratio에서 일관되게 유리하지는 않았다.

### F5.1. Augmentation Add Ratio Ablation

목적:

- `augmentation_add_ratio = 0.5, 1.0, 2.0`를 비교해 synthetic data 양이 augmentation 효과를 어떻게 바꾸는지 분석한다.

규칙:

- official F5와 동일한 `offline_append` design 사용
- preprocessing = `moving_average_smoothing+minmax_scaling`
- benchmark top3 models 사용
- `real_ratio = 0.5, 0.25, 0.1`
- same-`real_ratio` baseline comparison 유지
- `augmentation_add_ratio=1.0` rows는 official F5 결과를 `source=reused_final_f5`로 재사용
- `0.5`, `2.0` rows는 `source=trained_ablation`으로 별도 실행

output:

- `results/metrics/final_augmentation_ratio_ablation_results.csv`
- `results/metrics/final_augmentation_ratio_ablation_summary_by_add_ratio.csv`
- `results/figures/final_augmentation_ablation_macro_f1_by_add_ratio.png`
- `results/figures/final_augmentation_ablation_gain_by_add_ratio.png`
- `results/figures/final_augmentation_ablation_best_add_ratio_by_condition.png`
- `results/figures/final_augmentation_ablation_heatmap.png`
- `results/figures/final_augmentation_ablation_add_ratio_summary_macro_f1.png`
- `results/figures/final_augmentation_ablation_add_ratio_summary_gain.png`
- `results/figures/final_augmentation_ablation_add_ratio_positive_rate.png`
- `results/figures/final_augmentation_ablation_add_ratio_accuracy_summary.png`

완료 상태:

- F5.1 augmentation_add_ratio ablation completed

결과 요약:

- `0.5` appears most stable on average
- `1.0` is partially useful but not uniformly optimal
- `2.0` can be too aggressive
- synthetic data amount should be tuned rather than maximized
- aggregate-by-`augmentation_add_ratio` summary analysis across the full 3x3 condition grid confirms that moderate synthetic ratio is more stable than aggressive synthetic ratio
- augmentation showed limited and condition-dependent benefit, not a robust universal solution
- F4 shows that low-data reduction is feasible, especially with `ResNet18`
- F5/F5.1 show that simple augmentation is only partially helpful and does not solve low-data robustness consistently
- future work should focus on physics-aware, class-conditional, and model-specific augmentation
- official reporting should now use F1-F5.1 terminology consistently

### F6. Final Report

규칙:

- official final workflow outputs만 final evidence로 사용한다.
- prototype 결과는 final conclusion에 섞지 않는다.

최종 보고서 포함 항목:

- problem definition
- dataset and input explanation
- benchmark model comparison
- preprocessing comparison
- final preprocessing decision
- low-data robustness
- augmentation recovery
- augmentation add ratio ablation
- limitations
- future work

## 4. Preprocessing Candidate Pool

- `none / raw`: baseline without normalization or smoothing
- `train_global_zscore`: train split에서 fit한 mean/std를 `train/val/test`에 동일 적용
- `per_sample_zscore`: sample별 독립 정규화
- `minmax_scaling`: fixed range rescaling
- `robust_scaling`: median/IQR 기반 scaling
- `savgol_smoothing`: local temporal structure 보존을 기대하는 smoothing candidate
- `moving_average_smoothing`: simple smoothing baseline
- `train_featurewise_zscore`: train-only feature-wise z-score
- `per_sample_featurewise_zscore`: sample별 feature-wise normalization

구현 규칙:

- train-stat transform은 selected train split에만 fit한다.
- fitted statistics는 `train/val/test`에 적용한다.
- deterministic smoothing은 `train/val/test`에 일관되게 적용한다.
- augmentation은 preprocessing과 분리하여 train-only로 사용한다.

## 5. Evaluation Metrics

- `Accuracy`
- `Macro F1`
- `Weighted F1`
- `Confusion Matrix` where useful
- low-data degradation metrics:
  - `macro_f1_drop`
  - `macro_f1_retention`
  - `accuracy_drop`
  - `accuracy_retention`
- augmentation gain metrics:
  - `augmentation_gain_macro_f1`
  - `augmentation_gain_accuracy`

## 6. Execution Checklist

- [x] Clean old generated artifacts
- [x] F1 original benchmark full run completed
- [x] Benchmark rank 1 selected
- [x] Benchmark top3 selected
- [x] F2 single preprocessing comparison completed
- [x] F3 preprocessing combination comparison completed
- [x] Final preprocessing selected
- [x] F4 low-data robustness completed
- [x] F5 augmentation recovery completed
- [x] F5.1 augmentation_add_ratio ablation completed
- [ ] Multi-seed F5.1 validation completed
- [ ] Final report packaging completed

## 7. Next Recommended Work

- multi-seed F5.1
- finer augmentation ratio search
- class/model-specific augmentation
- physics-aware CSI augmentation
