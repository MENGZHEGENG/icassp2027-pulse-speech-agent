PYTHON ?= python3
PYTHONPATH := src

BENCHMARK_RECORDS ?= data/ambiguous_tasks.jsonl
SYNTHETIC_RECORDS ?= data/task_hypotheses.jsonl
SIMULATION_SUMMARY ?= results/sim/policy_summary.json
SIMULATION_RECORDS ?= results/sim/policy_trials.jsonl
SWEEP_CSV ?= results/task_policy_sweep.csv
SWEEP_JSON ?= results/task_policy_sweep.json

.PHONY: test generate-benchmark synthesize-hypotheses simulate-policy sweep-policy download-librispeech build-librispeech download-fsc build-fsc

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q

generate-benchmark:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pulse.generate_benchmark --output $(BENCHMARK_RECORDS)

simulate-policy:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pulse.simulate_policy --output $(SIMULATION_SUMMARY) --records $(SIMULATION_RECORDS)

synthesize-hypotheses: generate-benchmark
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pulse.synthesize_task_hypotheses $(BENCHMARK_RECORDS) --output $(SYNTHETIC_RECORDS)

sweep-policy: synthesize-hypotheses
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pulse.sweep_task_policy $(SYNTHETIC_RECORDS) --output-csv $(SWEEP_CSV) --output-json $(SWEEP_JSON)

download-librispeech:
	PROJECT_DIR=$(CURDIR) bash scripts/download_librispeech_test_clean.sh

build-librispeech:
	PROJECT_DIR=$(CURDIR) bash scripts/download_librispeech_test_clean.sh

download-fsc:
	PROJECT_DIR=$(CURDIR) bash scripts/fetch_fluent_speech_commands.sh

build-fsc: download-fsc
