# Entry points for KAPPA. Every downstream target should depend on `data`,
# which is idempotent and cheap once the dataset is staged.

SHELL := /bin/bash
.DEFAULT_GOAL := help
ARGS ?=

.PHONY: help install data extract test smoke paths

help:  ## List available targets
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-10s %s\n", $$1, $$2}'

install:  ## Install the kappa package in editable mode, with data and dev tooling
	# [dev] provides pytest for `make test`; [train] pulls torch.
	python -m pip install -e '.[data,dev,train]'

data:  ## Fetch the archive (login node; extraction is a separate batch job)
	@scripts/get_data.sh --fetch-only $(ARGS)

extract:  ## Extract the fetched archive (submits a Slurm job; needs 32 cores)
	@sbatch slurm/extract.sbatch $(ARGS)

test:  ## Run the test suite
	@python -m pytest

smoke:  ## Prove the teacher/student wiring on synthetic data (seconds, no dataset)
	# Not an experiment: random init, one epoch, ~12 fake studies. The macro
	# figures it prints are noise. Run it after any change to the model, the
	# loss, the config schema or the loop.
	@python scripts/smoke_distill.py $(ARGS)

paths:  ## Print the resolved data layout
	@python -m kappa.data.paths
