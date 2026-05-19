# Original Baseline Preservation

This directory preserves the original comparison baseline from `SenseFi / WiFi-CSI-Sensing-Benchmark`.

## Purpose

- Keep the original benchmark-style implementation separate from the new project-specific framework.
- Use this code only for comparison with the final project experiments.

## Copied Files

The full repository was not copied. Only the files needed to reproduce the original runnable pipeline were copied:

- `run.py`
- `dataset.py`
- `util.py`
- `UT_HAR_model.py`
- `NTU_Fi_model.py`
- `widar_model.py`
- `self_supervised_model.py`
- `self_supervised.py`
- `requirements.txt`
- `LICENSE`

## Compatibility Note

The original logic was preserved. Only `run.py` was minimally adjusted so its dataset root points to the new project path `../data/`.

## Reproduction Command

From the repository root:

```bash
python original_baseline/run.py --model GRU --dataset UT_HAR_data
```
