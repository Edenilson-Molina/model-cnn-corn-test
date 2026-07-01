PYTHON = python

.PHONY: setup data-collect data-split data-prepare train evaluate predict explain explain-batch pipeline clean all

setup:
	$(PYTHON) -m pip install -r requirements.txt

data-collect:
	$(PYTHON) script/collect_data.py

data-split:
	$(PYTHON) script/split_data.py

data-prepare: data-collect data-split

train:
	$(PYTHON) src/train.py

evaluate:
	$(PYTHON) src/evaluate.py

predict:
	$(PYTHON) src/predict.py --image $(IMG)

explain:
	$(PYTHON) src/explain.py --image $(IMG)

explain-batch:
	$(PYTHON) src/explain.py --n-images $(or $(N),3)

pipeline: train evaluate

clean:
	$(PYTHON) -c "import shutil, glob, os; [os.remove(f) for f in glob.glob('outputs/*')]; [os.remove(f) for f in glob.glob('models/*.pth')]"

all: clean pipeline
