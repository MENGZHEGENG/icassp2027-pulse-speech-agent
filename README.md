# PULSE reproducibility repository

This repository contains the executable code, configuration, tests, and small
local examples for the PULSE spoken-agent study. It is separate from the
private project record and does not contain the manuscript, private speech
data, or internal review material.

## Scope

The repository supports local reproduction of the benchmark generator,
policy simulation, threshold sweep, record utilities, and optional dataset
preparation helpers. Public datasets and model checkpoints are obtained by
the user when needed; they are not stored here.

## Local checks

From the repository root:

```bash
make test
make generate-benchmark
make simulate-policy
make sweep-policy
```

The default commands use Python 3 and write generated outputs below
`data/` or `results/`. Paths can be overridden on the command line or with
environment variables documented in the scripts.

## Reproduction boundary

The code reports bounded, closed-world evaluations. Reproducing the paper's
private human-spoken safety gate requires access to the approved data and is
not implied by this repository.
