# Wi-Fi CSI 기반 행동 인식에서 Low-data Generalization과 전처리/모델 구조의 영향 분석

## 1. 기본 정보

- 제목: Wi-Fi CSI 기반 행동 인식에서 Low-data Generalization과 전처리/모델 구조의 영향 분석
- 소속:
- 학번:
- 이름:
- Dataset: `UT-HAR`
- Task: `7-class Human Activity Recognition`

## 2. 문제 정의 및 동기

Wi-Fi CSI 기반 `Human Activity Recognition`은 여전히 `generalization failure` 문제를 크게 겪는다. CSI signal은 `environment`, `subject`, `device position`, `room layout`, `noise` 변화에 민감하고, 같은 행동이라도 수집 환경이 달라지면 분포가 쉽게 바뀔 수 있다. 따라서 특정 환경에서 높은 성능이 나왔다고 해도, 새로운 환경에서 같은 수준의 성능을 유지하는 것은 어렵다.

또한 새로운 공간이나 배치마다 충분한 `labeled data`를 다시 수집하는 것은 비용이 크다. 그래서 본 프로젝트는 real labeled data가 줄어드는 상황에서 성능 저하를 얼마나 줄일 수 있는지에 초점을 둔다. 구체적으로는 preprocessing, model choice, augmentation이 `low-data generalization`에 어떤 영향을 주는지 분석하려고 한다. 현재 preliminary report에서는 `data check / EDA`, `preprocessing ablation`, `full-data baseline`까지를 정리하고, 이후 `low-data robustness`와 `augmentation recovery` 실험을 이어서 수행할 계획이다.

## 3. Dataset 설명

본 프로젝트는 `UT-HAR` dataset을 사용한다. 입력 feature shape은 `X_train=(3977, 250, 90)`, `X_val=(496, 250, 90)`, `X_test=(500, 250, 90)`이며, label은 `0..6` 범위의 `7-class classification` task이다. 파일 확장자는 `.csv`이지만 실제 저장 형식은 NumPy binary이므로 `np.load`로 불러와야 한다.

참고 figure:

- `results/figures/class_distribution.png`
- `results/figures/sample_csi_heatmap.png`
- `results/figures/sample_csi_lineplot.png`

class distribution은 완전히 균등하지 않다. 따라서 단순 `Accuracy`만 보면 특정 class의 성능 저하를 놓칠 수 있어 `Macro F1`를 주요 metric으로 사용한다. 또한 각 sample이 `time x CSI feature` 구조를 가지므로, heatmap은 전체 행렬 구조를 보여주고 line plot은 일부 feature의 시간축 변화를 보여주는 보완적 시각화가 된다.

## 4. 전처리

전처리는 `none`, `train_global_zscore`, `per_sample_zscore` 세 가지를 비교했다.

- `none/raw`: 입력값을 그대로 사용
- `train_global_zscore`: `X_train`에서만 mean/std를 구한 뒤 동일한 통계량을 `train/val/test`에 적용
- `per_sample_zscore`: 각 sample 내부의 `time x feature` 값을 독립적으로 정규화

controlled setting은 다음과 같다.

- model=`GRU`
- `real_ratio=0.25`
- `augmentation=false`
- `seed=42`
- `epochs=30`
- `selected_by=validation Macro F1`

| preprocessing | val_macro_f1 | test_macro_f1 |
|---|---:|---:|
| per_sample_zscore | 0.8851 | 0.8351 |
| train_global_zscore | 0.8569 | 0.7983 |
| none | 0.6918 | 0.6305 |

현재 결과에서는 `per_sample_zscore`가 validation과 test `Macro F1` 모두에서 가장 높게 나타났다. 따라서 main experiments의 기본 preprocessing policy로 `per_sample_zscore`를 선택했다. 다만 `per_sample_zscore`는 absolute amplitude information을 일부 제거할 수 있다는 점은 주의해야 한다. 그럼에도 현재 controlled setting에서는 가장 좋은 일반화 성능을 보여 주었기 때문에, 이후 main experiment에서는 이 정책을 유지하는 것이 타당하다.

참고 figure:

- `results/figures/preprocessing_ablation_macro_f1.png`
- `results/figures/preprocessing_ablation_val_test_macro_f1_zoomed.png`

## 5. 모델 선택

본 프로젝트는 leaderboard-style benchmark가 아니다. 모델을 무작정 많이 늘리는 대신, 해석 가능한 비교를 위해 대표적인 네 가지 구조만 선택했다.

- `CNN`: local time-subcarrier pattern baseline
- `GRU`: lightweight sequence baseline
- `LSTM`: representative recurrent sequence baseline
- `CNN+GRU`: local feature extraction + temporal modeling hybrid

이 네 모델은 `CNN-only`, `RNN-only`, `hybrid` 구조를 모두 포함하면서도 비교 결과를 설명하기 쉽다. `Transformer` 계열은 흥미로운 future work이지만, 현재 단계에서는 low-data robustness와 interpretability가 핵심이므로 main scope에 넣지 않았다.

## 6. Full-data Baseline 실험

현재 full-data baseline 설정은 다음과 같다.

- `training_mode=original_epoch`
- models=`CNN`, `GRU`, `LSTM`, `CNN_GRU`
- original baseline epoch mapping: `CNN=200`, `GRU=200`, `LSTM=200`, `CNN_GRU=200`
- `real_ratio=1.0`
- `augmentation=false`
- `preprocessing=per_sample_zscore`
- `seed=42`
- `device=cuda`

현재 `results/metrics/baseline_results_original_epoch.csv`에서 읽은 결과는 다음과 같다.

| model | best_epoch | val_macro_f1 | test_macro_f1 | val_accuracy | test_accuracy |
|---|---:|---:|---:|---:|---:|
| CNN | 200 | 0.9937 | 0.9708 | 0.9940 | 0.9820 |
| GRU | 74 | 0.9920 | 0.9622 | 0.9940 | 0.9740 |
| LSTM | 160 | 0.9852 | 0.9620 | 0.9899 | 0.9740 |
| CNN_GRU | 164 | 0.9914 | 0.9628 | 0.9940 | 0.9760 |

참고 figure:

- `results/figures/baseline_original_epoch_val_test_macro_f1_zoomed.png`
- `results/figures/baseline_original_epoch_test_macro_f1_zoomed.png`
- `results/figures/baseline_original_epoch_macro_f1_gap.png`

예비 해석으로는 full-data setting에서는 모든 모델이 매우 높은 성능을 보인다. 현재 결과 기준으로는 `CNN`이 `test Macro F1`와 `test Accuracy`에서 가장 강하게 보인다. 다만 validation score는 거의 포화 수준인 반면 test score는 그보다 낮아, validation-test gap을 어떻게 해석할지 보고서에서 반드시 논의해야 한다. 또한 이 결과만으로 최종 결론을 내리기보다, 이후 `real_ratio`를 줄였을 때 어떤 모델이 덜 무너지는지가 더 중요한 질문이다.

## 7. 현재까지의 해석

현재까지의 결과는 전처리 선택이 실제로 중요하다는 점을 보여 준다. `per_sample_zscore`는 low-data controlled setting에서 가장 좋은 `Macro F1`를 보였고, 따라서 main experiments의 기본 policy로 정당화된다. 반면 full-data setting은 상대적으로 쉬운 조건이어서 모든 모델이 강한 성능을 보인다. 즉 full-data에서는 모델 차이가 크지 않게 보일 수 있으며, 진짜 핵심 비교는 `real_ratio`가 `0.5`, `0.25`, `0.1`으로 줄어들 때 어떤 모델이 가장 덜 성능이 저하되는가에 있다.

## 8. 한계점

- 현재까지는 `UT-HAR` 단일 dataset만 사용했다.
- full-data baseline은 `200 epochs`를 사용하므로 overfitting 가능성을 함께 논의해야 한다.
- 아직 `M4`와 `M5` 결과가 없다.
- augmentation 효과는 아직 평가하지 않았다.
- 현재 결과는 사실상 `single seed`에 기반한다.

## 9. 향후 계획

- `M4 Low-data robustness`: `real_ratio=1.0, 0.5, 0.25, 0.1`
- `M5 Augmentation recovery`: `train-only augmentation`
- best model에 대한 `confusion matrix` 분석
- report 및 presentation refinement

## 10. 실행 명령 정리

```bash
python experiments/01_check_data.py
```

```bash
python experiments/06_run_preprocessing_ablation.py --epochs 30 --real-ratio 0.25 --seed 42 --batch-size 64
```

```bash
python experiments/07_run_full_baseline_all_models.py --training-mode original_epoch --seed 42 --batch-size 64 --preprocessing per_sample_zscore
```

```bash
python experiments/08_regenerate_figures.py
```
