
.PHONY: train predict21 predict100

VENV?=.venv

init:
	python3 -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -U pip && pip install -r requirements.txt

train:
	python captcha_solver.py --train_input_dir data/input --train_output_dir data/output --model model/captcha_model.json --q 0.10

predict21:
	python captcha_solver.py --model model/captcha_model.json --predict data/input/input21.jpg --save predictions/out21.txt

predict100:
	python captcha_solver.py --model model/captcha_model.json --predict data/input/input100.jpg --save predictions/out100.txt
