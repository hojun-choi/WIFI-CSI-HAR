"""Train-only augmentation utilities for low-data recovery experiments.

Keeping augmentation separate makes it easier to prove that validation and test
data remain untouched, which prevents evaluation leakage.
"""

# Rubric: train-only augmentation prevents evaluation leakage and supports a
# defensible comparison between augmented and non-augmented low-data settings.
