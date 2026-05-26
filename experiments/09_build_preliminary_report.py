from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_dir


REAL_RATIO_ORDER = [1.0, 0.5, 0.25, 0.1]
TIMESTEPS_PER_SAMPLE = 250


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _format_float(value: str | float | None, digits: int = 4) -> str:
    if value in ("", None):
        return "-"
    return f"{float(value):.{digits}f}"


def _ordered_low_data_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    order_index = {ratio: idx for idx, ratio in enumerate(REAL_RATIO_ORDER)}
    return sorted(
        rows,
        key=lambda row: (
            row["model"],
            order_index.get(float(row["real_ratio"]), 999),
        ),
    )


def build_frame_count_table(low_data_rows: list[dict[str, str]]) -> str:
    if not low_data_rows:
        return (
            "`results/metrics/low_data_results.csv`가 아직 없으므로 frame count 표는 "
            "공식 low-data run 이후 자동으로 채워질 예정이다."
        )

    grouped: dict[float, dict[str, str]] = {}
    for row in low_data_rows:
        ratio = float(row["real_ratio"])
        if ratio not in grouped:
            grouped[ratio] = row

    lines = [
        "| real_ratio | selected_train_size | timesteps_per_sample | total_train_csi_frames |",
        "|---:|---:|---:|---:|",
    ]
    for ratio in REAL_RATIO_ORDER:
        if ratio not in grouped:
            continue
        selected_train_size = int(grouped[ratio]["selected_train_size"])
        total_train_csi_frames = selected_train_size * TIMESTEPS_PER_SAMPLE
        lines.append(
            f"| {ratio:g} | {selected_train_size} | {TIMESTEPS_PER_SAMPLE} | {total_train_csi_frames} |"
        )
    return "\n".join(lines)


def build_duration_estimate_table(low_data_rows: list[dict[str, str]], fs: int = 100) -> str:
    if not low_data_rows:
        return (
            "`results/metrics/low_data_results.csv`가 아직 없으므로 100Hz 가정 시간 표는 "
            "공식 low-data run 이후 자동으로 채워질 예정이다."
        )

    grouped: dict[float, dict[str, str]] = {}
    for row in low_data_rows:
        ratio = float(row["real_ratio"])
        if ratio not in grouped:
            grouped[ratio] = row

    lines = [
        "| real_ratio | selected_train_size | estimated_duration_at_100Hz_seconds | estimated_duration_at_100Hz_minutes |",
        "|---:|---:|---:|---:|",
    ]
    for ratio in REAL_RATIO_ORDER:
        if ratio not in grouped:
            continue
        selected_train_size = int(grouped[ratio]["selected_train_size"])
        estimated_seconds = selected_train_size * TIMESTEPS_PER_SAMPLE / fs
        estimated_minutes = estimated_seconds / 60.0
        lines.append(
            f"| {ratio:g} | {selected_train_size} | {estimated_seconds:.1f} | {estimated_minutes:.1f} |"
        )
    return "\n".join(lines)


def _build_preprocessing_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "`results/metrics/preprocessing_ablation_results.csv`가 없어 preprocessing 표를 만들지 못했다."

    ordered = sorted(rows, key=lambda row: float(row["val_macro_f1"]), reverse=True)
    lines = [
        "| preprocessing | val_macro_f1 | test_macro_f1 |",
        "|---|---:|---:|",
    ]
    for row in ordered:
        lines.append(
            f"| {row['preprocessing']} | {_format_float(row['val_macro_f1'])} | "
            f"{_format_float(row['test_macro_f1'])} |"
        )
    return "\n".join(lines)


def _build_final_preprocessing_table(rows: list[dict[str, str]]) -> str:
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["val_macro_f1"]),
            float(row["test_macro_f1"]),
        ),
        reverse=True,
    )
    lines = [
        "| preprocessing_group | model | preprocessing | val_macro_f1 | test_macro_f1 | val_accuracy | test_accuracy |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in ordered:
        lines.append(
            f"| {row['preprocessing_group']} | {row['model']} | {row['preprocessing']} | "
            f"{_format_float(row['val_macro_f1'])} | {_format_float(row['test_macro_f1'])} | "
            f"{_format_float(row['val_accuracy'])} | {_format_float(row['test_accuracy'])} |"
        )
    return "\n".join(lines)


def _build_final_preprocessing_section(
    rows: list[dict[str, str]],
    decision_exists: bool,
) -> str:
    if not rows:
        return """## 6. Final Workflow - Expanded Preprocessing Comparison

F2 expanded preprocessing comparison has not been run yet.
"""

    ordered = sorted(rows, key=lambda row: float(row["val_macro_f1"]), reverse=True)
    best_row = ordered[0]
    decision_note = (
        "`docs/final_preprocessing_decision.md` exists."
        if decision_exists
        else "`docs/final_preprocessing_decision.md` does not exist yet."
    )
    return f"""## 6. Final Workflow - Expanded Preprocessing Comparison

The table below summarizes official revised-workflow F2 outputs only. This section does not use `preprocessing_ablation_results.csv` as final evidence.

{_build_final_preprocessing_table(rows)}

- best preprocessing by validation `Macro F1`: `{best_row['preprocessing']}`
- preprocessing group: `{best_row['preprocessing_group']}`
- model: `{best_row['model']}`
- selection rule: validation `Macro F1`
- test `Macro F1` is used for confirmation only.
- decision document status: {decision_note}
"""


def _build_baseline_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return (
            "`results/metrics/baseline_results_original_epoch.csv`가 없어 benchmark 표를 만들지 못했다. "
            "F1 official rerun 이후 갱신이 필요하다."
        )

    lines = [
        "| model | best_epoch | val_macro_f1 | test_macro_f1 | val_accuracy | test_accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['best_epoch']} | {_format_float(row['val_macro_f1'])} | "
            f"{_format_float(row['test_macro_f1'])} | {_format_float(row['val_accuracy'])} | "
            f"{_format_float(row['test_accuracy'])} |"
        )
    return "\n".join(lines)


def _build_low_data_section(rows: list[dict[str, str]]) -> str:
    if not rows:
        return """## 7. Prototype Low-data Robustness Snapshot

`results/metrics/low_data_results.csv`가 아직 없으므로 본 섹션은 placeholder 상태다. revised workflow에서는 final preprocessing selection이 완료된 뒤 `real_ratio=1.0, 0.5, 0.25, 0.1`에 대해 official low-data robustness를 다시 실행해야 한다.
"""

    ordered = _ordered_low_data_rows(rows)
    lines = [
        "## 7. Prototype Low-data Robustness Snapshot",
        "",
        "본 섹션은 현재 저장된 `results/metrics/low_data_results.csv`를 요약하지만, 이는 revised workflow 기준 final evidence가 아니라 prototype/development snapshot이다. 특히 현재 run은 `per_sample_zscore` 기반이므로 F3에서 final preprocessing이 확정되면 F4 official low-data run을 다시 수행할 수 있다.",
        "",
        "| model | real_ratio | test_macro_f1 | macro_f1_drop | macro_f1_retention | test_accuracy | accuracy_drop | accuracy_retention |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ordered:
        lines.append(
            f"| {row['model']} | {_format_float(row['real_ratio'])} | "
            f"{_format_float(row['test_macro_f1'])} | {_format_float(row.get('macro_f1_drop'))} | "
            f"{_format_float(row.get('macro_f1_retention'))} | {_format_float(row['test_accuracy'])} | "
            f"{_format_float(row.get('accuracy_drop'))} | {_format_float(row.get('accuracy_retention'))} |"
        )

    ratio_01_rows = [row for row in rows if abs(float(row["real_ratio"]) - 0.1) < 1e-9]
    interpretation_lines = ["", "현재 prototype low-data 결과에서 읽을 수 있는 제한적 관찰:"]
    if ratio_01_rows:
        best_macro = max(ratio_01_rows, key=lambda row: float(row["test_macro_f1"]))
        valid_drop_rows = [row for row in ratio_01_rows if row.get("macro_f1_drop") not in ("", None)]
        interpretation_lines.append(
            f"- `real_ratio=0.1`에서 highest `test_macro_f1` model은 `{best_macro['model']}`이다."
        )
        if valid_drop_rows:
            smallest_drop = min(valid_drop_rows, key=lambda row: float(row["macro_f1_drop"]))
            interpretation_lines.append(
                f"- `real_ratio=0.1`에서 smallest `macro_f1_drop` model은 `{smallest_drop['model']}`이다."
            )
    else:
        interpretation_lines.append(
            "- `real_ratio=0.1` 결과가 없으므로 가장 극단적인 low-data 비교는 아직 확정적으로 해석할 수 없다."
        )

    interpretation_lines.extend(
        [
            "",
            "- 이 해석은 current prototype run에 대한 요약일 뿐이며, final report 결론으로 직접 사용하지 않는다.",
            "",
            "참고 figure:",
            "",
            "- `results/figures/low_data_macro_f1.png`",
            "- `results/figures/low_data_accuracy.png`",
            "- `results/figures/low_data_degradation_macro_f1.png`",
            "- `results/figures/low_data_degradation_accuracy.png`",
        ]
    )
    return "\n".join(lines + interpretation_lines) + "\n"


def _build_augmentation_section(rows: list[dict[str, str]]) -> str:
    if not rows:
        return """## 8. Augmentation Recovery Status

`results/metrics/augmentation_results.csv`가 아직 없으므로 본 섹션은 pending 상태다. revised workflow에서는 final preprocessing selection 이후 F4 no-augmentation baseline을 확정한 다음, `real_ratio=0.5, 0.25, 0.1`에서 train-only augmentation recovery를 official하게 다시 실행해야 한다.
"""

    ordered = _ordered_low_data_rows(rows)
    lines = [
        "## 8. Augmentation Recovery Status",
        "",
        "본 섹션은 현재 저장된 augmentation 결과를 요약하지만, final preprocessing과 official F4 baseline이 확정되기 전이라면 final evidence로 직접 사용하지 않는다.",
        "",
        "| model | real_ratio | no_aug_test_macro_f1 | test_macro_f1 | augmentation_gain_macro_f1 | no_aug_test_accuracy | test_accuracy | augmentation_gain_accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ordered:
        lines.append(
            f"| {row['model']} | {_format_float(row['real_ratio'])} | "
            f"{_format_float(row.get('no_aug_test_macro_f1'))} | {_format_float(row['test_macro_f1'])} | "
            f"{_format_float(row.get('augmentation_gain_macro_f1'))} | {_format_float(row.get('no_aug_test_accuracy'))} | "
            f"{_format_float(row['test_accuracy'])} | {_format_float(row.get('augmentation_gain_accuracy'))} |"
        )

    interpretation_lines = ["", "현재 augmentation 결과의 제한적 관찰:"]
    valid_gain_rows = [
        row for row in rows if row.get("augmentation_gain_macro_f1") not in ("", None)
    ]
    if valid_gain_rows:
        largest_positive = max(valid_gain_rows, key=lambda row: float(row["augmentation_gain_macro_f1"]))
        hurt_rows = [row for row in valid_gain_rows if float(row["augmentation_gain_macro_f1"]) < 0.0]
        interpretation_lines.append(
            f"- largest positive `augmentation_gain_macro_f1`는 `{largest_positive['model']}` at `real_ratio={float(largest_positive['real_ratio']):g}`이다."
        )
        if hurt_rows:
            hurt_descriptions = ", ".join(
                f"{row['model']}@{float(row['real_ratio']):g}"
                for row in sorted(
                    hurt_rows,
                    key=lambda row: (float(row["real_ratio"]), row["model"]),
                )
            )
            interpretation_lines.append(
                f"- augmentation이 악화된 case도 있으며 현재 결과에서는 {hurt_descriptions}가 추가 해석 대상이다."
            )
        else:
            interpretation_lines.append("- 현재 저장된 결과에서는 `augmentation_gain_macro_f1 < 0`인 case가 뚜렷하지 않다.")
    else:
        interpretation_lines.append(
            "- gain column이 아직 없거나 비어 있으므로 augmentation effect는 official rerun 이후 해석해야 한다."
        )

    interpretation_lines.extend(
        [
            "",
            "참고 figure:",
            "",
            "- `results/figures/augmentation_recovery_macro_f1.png`",
            "- `results/figures/augmentation_recovery_accuracy.png`",
            "- `results/figures/augmentation_gain_macro_f1.png`",
            "- `results/figures/augmentation_gain_accuracy.png`",
        ]
    )
    return "\n".join(lines + interpretation_lines) + "\n"


def build_preliminary_report(project_root: Path | None = None) -> Path:
    root = project_root or PROJECT_ROOT
    report_path = root / "reports" / "preliminary_report.md"
    ensure_dir(report_path.parent)

    preprocessing_rows = _read_csv_rows(
        root / "results" / "metrics" / "preprocessing_ablation_results.csv"
    )
    final_preprocessing_rows = _read_csv_rows(
        root / "results" / "metrics" / "final_preprocessing_results.csv"
    )
    baseline_rows = _read_csv_rows(
        root / "results" / "metrics" / "baseline_results_original_epoch.csv"
    )
    low_data_rows = _read_csv_rows(root / "results" / "metrics" / "low_data_results.csv")
    augmentation_rows = _read_csv_rows(root / "results" / "metrics" / "augmentation_results.csv")

    preprocessing_table = _build_preprocessing_table(preprocessing_rows)
    final_preprocessing_section = _build_final_preprocessing_section(
        final_preprocessing_rows,
        decision_exists=(root / "docs" / "final_preprocessing_decision.md").exists(),
    )
    baseline_table = _build_baseline_table(baseline_rows)
    low_data_section = _build_low_data_section(low_data_rows)
    augmentation_section = _build_augmentation_section(augmentation_rows)
    frame_count_table = build_frame_count_table(low_data_rows)
    duration_estimate_table = build_duration_estimate_table(low_data_rows, fs=100)
    baseline_device = baseline_rows[0]["device"] if baseline_rows else "unknown"

    report_text = f"""# Wi-Fi CSI 기반 HAR Preliminary Report

## 1. 문서 성격

이 문서는 현재 저장소에 존재하는 실험 결과를 정리한 `preliminary_report`이다. 기존 codebase는 계속 재사용하지만, 현재 저장된 `results/`와 `reports/` 산출물은 revised final project 관점에서 prototype/development output으로 취급한다. 따라서 아래 표와 해석은 구현 검증과 계획 수립에는 유용하지만, final report의 공식 결론으로 직접 사용하지 않는다.

## 2. 문제 정의와 데이터

Wi-Fi CSI 기반 `Human Activity Recognition`은 `environment`, `subject`, `device position`, `room layout`, `noise` 변화에 민감하다. 이 프로젝트는 `UT-HAR`에서 `real_ratio`를 줄였을 때 성능이 얼마나 유지되는지, 그리고 preprocessing, model choice, `train-only augmentation`이 그 일반화 성능에 어떤 영향을 주는지를 분석하려고 한다.

- Dataset: `UT-HAR`
- Input shape: `X_train=(3977, 250, 90)`, `X_val=(496, 250, 90)`, `X_test=(500, 250, 90)`
- Label range: `0..6`
- One sample: `250 timesteps x 90 CSI features`
- Benchmark code preserved under `original_baseline/`

참고 figure:

- `results/figures/class_distribution.png`
- `results/figures/sample_csi_heatmap.png`
- `results/figures/sample_csi_lineplot.png`

## 3. CSI frame budget 해석

본 저장소에서는 sample 수와 함께 `CSI frame count`를 직접 기록한다. `real_ratio`가 줄어들면 selected train sample 수가 줄고, 각 sample이 `250 CSI frames`를 포함하므로 총 labeled training frame budget도 함께 줄어든다.

### Total CSI frame count by real_ratio

{frame_count_table}

### 100Hz 가정 시 예시 시간 길이

{duration_estimate_table}

위 시간 환산은 `sampling_rate=100Hz` 가정에 기반한 보조 설명일 뿐이며, 본 프로젝트의 주 분석 단위는 시간보다 `CSI frame count`와 `real_ratio`다.

## 4. Prototype Benchmark Snapshot

현재 benchmark prototype은 `training_mode=original_epoch`와 `baseline_results_original_epoch.csv`를 기준으로 정리할 수 있다. 이 결과는 useful benchmark prototype이지만, revised final workflow에서는 F1 official benchmark run으로 다시 생성하거나 `final_` prefix output으로 분리할 수 있다.

- `training_mode=original_epoch`
- `augmentation=false`
- `real_ratio=1.0`
- current preprocessing in the saved prototype run: `per_sample_zscore`
- device in saved CSV: `{baseline_device}`

{baseline_table}

이 표는 prototype benchmark ranking snapshot으로는 유용하지만, 이후 preprocessing decision과 low-data/augmentation 결론까지 대신해 주지는 않는다.

## 5. Prototype Preprocessing Snapshot

현재 preprocessing 결과는 제한된 candidate만 비교한 preliminary snapshot이다. saved comparison은 `none`, `train_global_zscore`, `per_sample_zscore`만 포함하며, revised workflow에서 요구하는 `Min-Max Normalization`, `Robust Scaling`, `Savitzky-Golay Smoothing`, `Moving Average Smoothing`, `train_featurewise_zscore`, `per_sample_featurewise_zscore`는 아직 official comparison에 포함되지 않았다.

controlled setting:

- model=`GRU`
- `real_ratio=0.25`
- `augmentation=false`
- `seed=42`
- selection metric=`validation Macro F1`

{preprocessing_table}

현재 저장된 limited candidate 비교에서는 `per_sample_zscore`가 가장 높은 validation `Macro F1`를 보인다. 그러나 이 결과는 어디까지나 limited candidate pool에서의 preliminary observation이다. 따라서 revised workflow에서는 expanded preprocessing comparison을 다시 수행하고, 그 결과가 나오기 전에는 `per_sample_zscore`나 `Savitzky-Golay Smoothing` 중 어느 것도 final best라고 주장하지 않는다.

참고 figure:

- `results/figures/preprocessing_ablation_macro_f1.png`
- `results/figures/preprocessing_ablation_val_test_macro_f1_zoomed.png`

{final_preprocessing_section}

{low_data_section}

{augmentation_section}

## 9. Revised Final Workflow and Remaining Work

- Existing results are currently treated as prototype/development outputs.
- The final report should be based on official revised workflow outputs.
- Expanded preprocessing comparison infrastructure is implemented, but official F2 results may still be pending.
- Existing `per_sample_zscore` result is preliminary among limited candidates only.
- M4 low-data results may need to be rerun after final preprocessing is selected.
- M5 full augmentation recovery remains pending for the revised workflow.
- Future result files should preferably use `final_`-prefixed filenames.

official revised workflow:

1. F1 benchmark model comparison with `original_epoch`
2. F2 expanded preprocessing comparison
3. F3 final preprocessing selection by validation `Macro F1`
4. F4 low-data robustness with selected preprocessing
5. F5 augmentation recovery against F4 no-augmentation baseline
6. F6 final report from official revised outputs only

## 10. 한계와 주의사항

- 현재 문서는 final report가 아니라 prototype snapshot 정리 문서다.
- saved preprocessing comparison은 candidate pool이 제한적이다.
- saved low-data 결과는 current preprocessing choice에 종속되어 있으므로 F3 이후 rerun 필요성이 있다.
- `augmentation_results.csv`가 없으면 augmentation 결론은 비워 둔다.
- 현재 결과 대부분은 single dataset, single seed 관점의 development evidence다.

## 11. 참고 명령

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
python experiments/10_run_low_data_robustness.py --seed 42 --batch-size 64 --preprocessing per_sample_zscore
```

```bash
python experiments/11_run_augmentation_recovery.py --seed 42 --batch-size 64 --preprocessing per_sample_zscore
```

```bash
python experiments/08_regenerate_figures.py
```
"""

    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def main() -> None:
    report_path = build_preliminary_report()
    print(f"Saved preliminary report: {report_path}")


if __name__ == "__main__":
    main()
