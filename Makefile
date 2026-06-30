PYTHON = python

.PHONY: setup data-collect data-split data-prepare train evaluate predict pipeline clean all

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

pipeline: train evaluate

clean:
	$(PYTHON) -c "import shutil, glob, os; [os.remove(f) for f in glob.glob('outputs/*')]; [os.remove(f) for f in glob.glob('models/*.pth')]"

all: clean pipeline
