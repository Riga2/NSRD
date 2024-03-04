@echo off
echo Running training on BistroX4...

set train_config=../configs/Bistro_train.txt

python main.py --config %train_config%

echo Training done.
pause
