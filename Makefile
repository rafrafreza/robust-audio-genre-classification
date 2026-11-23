# Usage: make train-ast DATA_ROOT=/path/to/messy_mashup
DATA_ROOT ?= data/messy_mashup
CKPT      ?= checkpoints
PY        ?= python

.PHONY: install test train-baseline train-cnn train-crnn train-ast train-all predict evaluate

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest

train-baseline:
	$(PY) -m src.train --model baseline --data-root $(DATA_ROOT) --out-dir $(CKPT)
train-cnn:
	$(PY) -m src.train --model cnn --data-root $(DATA_ROOT) --out-dir $(CKPT)
train-crnn:
	$(PY) -m src.train --model crnn --data-root $(DATA_ROOT) --out-dir $(CKPT)
train-ast:
	$(PY) -m src.train --model ast --data-root $(DATA_ROOT) --out-dir $(CKPT)

train-all: train-baseline train-cnn train-crnn train-ast

predict:
	$(PY) -m src.inference --checkpoint $(CKPT)/ast.pt --data-root $(DATA_ROOT) --out submission.csv

evaluate:
	$(PY) -m src.evaluate --data-root $(DATA_ROOT) \
		--checkpoints $(CKPT)/baseline.joblib $(CKPT)/cnn.pt $(CKPT)/crnn.pt $(CKPT)/ast.pt
