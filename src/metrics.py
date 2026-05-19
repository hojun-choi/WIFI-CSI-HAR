"""Project metrics utilities.

Macro F1 and Weighted F1 are planned alongside Accuracy because accuracy alone
can hide class-wise degradation in low-data settings.
"""

# Rubric: Macro F1 is tracked because Accuracy can hide class-wise degradation,
# especially when low-data sampling affects minority classes unevenly.
