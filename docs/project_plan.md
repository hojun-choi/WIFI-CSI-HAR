# 프로젝트 계획서

## 프로젝트 개요

이 프로젝트는 `UT-HAR`를 사용해 Wi-Fi CSI 기반 `Human Activity Recognition`의 일반화 성능을 분석한다. 핵심 질문은 `real_ratio`가 줄어들 때 성능이 어떻게 저하되는지, 그리고 preprocessing, model 구조, `train-only augmentation`이 그 저하를 얼마나 완화하는지 재현 가능하게 검증하는 것이다.

현재 저장소에는 data checking / EDA, benchmark-style baseline runner, `controlled_generalization` training policy, preprocessing ablation pipeline, low-data robustness pipeline, augmentation runner/smoke test, figure regeneration, report builder가 이미 존재한다. revised final project는 이 codebase를 그대로 재사용하되, 기존 산출물은 final evidence가 아니라 prototype/development output으로 취급한다.

## Final Revised Experiment Roadmap

revised final project는 기존 codebase 위에서 official final workflow를 새로 실행해 결과를 누적하는 방식으로 진행한다. 기존 `results/`와 `reports/` 산출물은 debugging, smoke validation, 설계 점검에는 유용하지만 final report의 근거로 직접 사용하지 않는다.

### F1. Benchmark full run

Purpose:

- full benchmark-style model comparison을 수행한다.
- benchmark rank 1 model과 benchmark top models를 식별한다.

Training:

- `training_mode=original_epoch`
- `original_baseline`의 original benchmark epoch policy를 그대로 사용
- no early stopping
- 지원되는 경우 validation `Macro F1` 기준 best checkpoint 저장

Expected artifacts:

- `results/metrics/final_benchmark_results.csv` 또는 mode-specific equivalent
- `results/figures/final_benchmark_*.png`

### F2. Single preprocessing comparison

Purpose:

- F1 benchmark rank 1 model을 사용해 single preprocessing candidate를 one-by-one 비교한다.
- 기존 `preprocessing_ablation_results.csv`는 prototype reference로만 취급한다.

Training:

- `training_mode=controlled_generalization`
- `warmup_epochs`, `patience`, `min_delta`, `weight_decay`, `gradient_clip_norm`, `scheduler_type`, best validation checkpoint logic 사용
- `augmentation=false`
- `real_ratio=0.25` 기본
- `seed=42`
- selection metric=`validation Macro F1`

Expected artifacts:

- `results/metrics/final_preprocessing_results.csv`
- `results/figures/final_preprocessing_*.png`

### F2-combination. Combination preprocessing comparison

Purpose:

- single preprocessing 결과를 검토한 뒤, promising하거나 논리적으로 상보적인 소수의 combination만 비교한다.
- 모든 조합을 blind search처럼 실행하지 않는다.

Training:

- `training_mode=controlled_generalization`
- F2 single 결과 검토 후 selected combination만 사용
- `augmentation=false`
- single 결과와 동일한 controlled setting 유지

Expected artifacts:

- `results/metrics/final_preprocessing_combination_results.csv` 또는 safe append equivalent
- `results/figures/final_preprocessing_*.png`

### F3. Final preprocessing selection

Purpose:

- final preprocessing policy 하나를 freeze한다.
- selection은 validation `Macro F1` 기준으로 수행한다.
- test `Macro F1`는 confirmation only로 사용한다.
- 점수 차이가 작으면 더 단순하고 해석 가능한 preprocessing을 우선한다.

Expected artifact:

- `docs/final_preprocessing_decision.md`

### F4. Low-data robustness

Purpose:

- final preprocessing이 고정된 상태에서 model robustness를 비교한다.

Setting:

- `preprocessing = final selected preprocessing`
- `models = benchmark top3 models from F1` 기본
- benchmark top5는 model 수와 시간 여유가 충분하고 사용자가 명시적으로 원할 때만 optional extension으로 사용
- `real_ratio = 1.0, 0.5, 0.25, 0.1`
- `augmentation=false`
- `training_mode=controlled_generalization`
- `validation/test unchanged`
- only train split reduced with deterministic `stratified sampling`

Expected artifacts:

- `results/metrics/final_low_data_results.csv`
- `results/figures/final_low_data_*.png`

### F5. Augmentation recovery

Purpose:

- 같은 final preprocessing과 같은 model set에서 train-only augmentation이 low-data 성능을 얼마나 회복시키는지 본다.

Setting:

- `preprocessing = final selected preprocessing`
- `models = same benchmark top3/top5 set used in F4`
- `real_ratio = 0.5, 0.25, 0.1`
- `augmentation=true`
- augmentation must be train-only
- `validation/test`는 절대 augmentation하지 않음
- F4 no-augmentation 결과와 직접 비교

Expected artifacts:

- `results/metrics/final_augmentation_results.csv`
- `results/figures/final_augmentation_*.png`

### F6. Final report

Purpose:

- official final workflow output만 사용해 final report를 작성한다.
- old prototype output은 필요하면 development/prototype check로만 언급한다.

Expected artifacts:

- `reports/final_report.md`
- `reports/final_report.pdf` if needed later

## Official Final Workflow Rule

- F1은 `original_epoch`를 사용한다.
- F2/F3/F4/F5는 `controlled_generalization`을 사용한다.
- F2 single preprocessing comparison은 benchmark rank 1 model을 사용한다.
- F2에서는 single preprocessing comparison이 combination comparison보다 반드시 먼저 수행되어야 한다.
- F2-combination은 single result review 이후에 selected combination만 소수 실행한다.
- F4/F5는 benchmark top3 model set을 기본으로 사용하고, top5는 optional extension이다.
- final preprocessing은 official low-data와 augmentation 실험 전에 반드시 freeze되어야 한다.
- old prototype output은 final evidence가 아니다.

## Prototype Results vs Final Results

기존 `M2/M3/M4/M5` smoke/prototype output은 codebase 검증과 계획 수립에 실제로 도움이 되었다. 하지만 revised final workflow에서는 fresh official result file을 새로 생성해 사용하며, prototype output과 final output을 섞지 않는다. 향후 구현에서는 `final_` prefix output filename을 우선 사용해 혼동을 줄인다.

| Existing artifact | Status | Future use |
|---|---|---|
| `dry_run_results.csv` | prototype pipeline validation | debugging 및 sanity-check reference |
| `preprocessing_ablation_results.csv` | preliminary preprocessing comparison among limited candidates | F2 candidate planning reference |
| `baseline_results_original_epoch.csv` | useful benchmark prototype, may be rerun as final benchmark | F1 rerun or output separation reference |
| `low_data_results.csv` | preliminary low-data result under `per_sample_zscore` | F4 metric schema / plotting reference |
| `augmentation_smoke_test_results.csv` | smoke test only, not final evidence | F5 pipeline validation reference |
| `preliminary_report.md` | draft, not final evidence | report structure template only |

## Sequential Dependency Rule

- F1이 benchmark rank 1 model과 benchmark top models를 정한다.
- F2 single이 individual preprocessing candidate를 비교한다.
- F2-combination은 single 결과를 검토한 뒤 selected combination만 추가 비교한다.
- F3이 final preprocessing을 freeze한다.
- F4는 frozen preprocessing과 benchmark top3/top5 model set을 사용한다.
- F5는 F4 no-augmentation baseline과 같은 model set을 사용해 augmentation recovery를 비교한다.
- F6는 official final workflow output만 사용해 report를 만든다.

이 규칙은 서로 다른 setting을 섞어 final conclusion처럼 서술하는 문제를 방지한다.

## Preprocessing Candidate Pool

### 1. `none / raw`

- normalization을 적용하지 않는다.
- preprocessing effect가 실제로 존재하는지 보여 주는 baseline이다.

### 2. `train_global_zscore`

- selected train split에서만 mean/std를 추정한다.
- 같은 statistics를 `train/val/test`에 적용한다.
- `validation/test` leakage를 피할 수 있다.

### 3. `per_sample_zscore`

- 각 sample을 독립적으로 normalize한다.
- sample-level amplitude variation을 줄일 수 있다.
- limitation은 absolute amplitude information을 일부 제거할 수 있다는 점이다.

### 4. `Min-Max Normalization`

- 값을 고정된 범위로 rescale한다.
- gradient 안정성과 convergence를 개선할 가능성이 있다.
- limitation은 noise 제거가 아니라 scale 변경이라는 점이다.

### 5. `Robust Scaling`

- median/IQR 기반 scaling을 사용한다.
- outlier에 상대적으로 강건하다.
- limitation은 CSI degradation의 주원인이 fluctuation/noise라면 효과가 제한적일 수 있다.

### 6. `Savitzky-Golay Smoothing`

- local temporal pattern을 보존하면서 high-frequency noise를 줄일 가능성이 있는 smoothing candidate다.
- revised workflow에서 테스트되기 전에는 superiority를 주장하지 않는다.

### 7. `Moving Average Smoothing`

- 가장 단순한 smoothing baseline이다.
- noise를 줄일 수 있지만 activity pattern을 blur할 수 있다.

### 8. `train_featurewise_zscore`

- selected train split에서 feature별 mean/std를 계산해 적용한다.
- global z-score보다 feature-aware한 scaling이다.

### 9. `per_sample_featurewise_zscore`

- sample 내부에서 feature별로 time axis 기준 normalization을 수행한다.
- feature-wise temporal scale variation을 줄일 수 있다.
- limitation은 유용한 amplitude cue까지 제거할 수 있다는 점이다.

### 10. `optional clipping / winsorization`

- simple하고 safe하게 지원되는 경우에만 optional candidate로 고려한다.
- outlier 관찰이 명확하지 않으면 기본 candidate pool에는 넣지 않는다.

## Final Preprocessing Implementation Checklist

- [x] Implement `Min-Max Normalization`.
- [x] Implement `Robust Scaling`.
- [x] Implement `Savitzky-Golay Smoothing`.
- [x] Implement `Moving Average Smoothing`.
- [x] Implement `train_featurewise_zscore`.
- [x] Implement `per_sample_featurewise_zscore`.
- [x] Implement a small set of preprocessing combinations.
- [x] Ensure train-statistics-based preprocessing fits only on train split.
- [x] Ensure deterministic smoothing is applied consistently to `train/val/test`.
- [x] Ensure augmentation remains train-only and separate from preprocessing.
- [x] Run expanded preprocessing comparison infrastructure implemented.
- [x] Select final preprocessing helper by validation `Macro F1` implemented.
- [x] Confirm with test `Macro F1` logic documented in selection helper.
- [ ] Official F1 benchmark full run completed.
- [ ] Benchmark rank 1 model selected.
- [ ] Single-method F2 run completed with benchmark rank 1 model.
- [ ] Promising preprocessing combinations selected.
- [ ] Combination F2 run completed.
- [ ] Final preprocessing selected.
- [ ] `docs/final_preprocessing_decision.md` created from full results.

## F2 Implementation Checklist

- [x] preprocessing helper implemented
- [x] expanded preprocessing runner created
- [x] single-method comparison mode added
- [x] combination comparison mode added
- [x] smoke-test mode added
- [x] final-prefixed output path added
- [x] figure regeneration integration added
- [x] report builder integration added
- [ ] official F1 benchmark full run completed
- [ ] benchmark rank 1 model selected
- [ ] single-method F2 run completed with benchmark rank 1 model
- [ ] promising preprocessing combinations selected
- [ ] combination F2 run completed
- [ ] final preprocessing selected
- [ ] `docs/final_preprocessing_decision.md` created from full results

## 현재 상태 요약

| Stage | Prototype status | Final workflow status |
|---|---|---|
| Benchmark comparison | prototype run exists | official F1 rerun pending |
| Preprocessing comparison | limited three-method result exists | F2 infrastructure implemented, official run pending |
| Low-data robustness | preliminary `per_sample_zscore` run exists | final rerun after preprocessing decision |
| Augmentation recovery | smoke test exists | final run pending |
| Report | preliminary report exists | final report pending |

## 운영 원칙

- 기존 artifact는 삭제하지 않는다.
- prototype output을 official final evidence로 간주하지 않는다.
- `original_baseline/`는 수정하지 않는다.
- preprocessing best claim은 F2/F3 official result 전에는 하지 않는다.
- 특히 `Savitzky-Golay Smoothing`은 revised workflow에서 테스트되기 전까지 best라고 주장하지 않는다.
- train-statistics transform은 selected train split에만 fit한다.
- deterministic smoothing은 `train/val/test`에 일관되게 적용한다.
- augmentation은 preprocessing과 분리된 train-only operation으로 유지한다.

## 다음 구현 우선순위

다음 단계는 official F1 benchmark full run을 완료해 benchmark rank 1 model과 benchmark top3 model set을 확정하는 것이다. 그 다음 순서로 F2 single preprocessing comparison, selected combination comparison, F3 decision, F4 low-data robustness, F5 augmentation recovery를 실행한다.
