# Usage

The main entry point is
{py:func}`bns_remnant_classifier.compute_bns_remnant_probs`, which runs
ClassifierA and/or ClassifierB on either posterior samples or a
low-latency point estimate.

```python
from bns_remnant_classifier import compute_bns_remnant_probs

probs_a, probs_b = compute_bns_remnant_probs(
    classifiers="AB",
    analysis="pos",
    par_file="path/to/posterior_samples.json",
)
```

See the {doc}`api` page for the full set of available functions and their
arguments.
