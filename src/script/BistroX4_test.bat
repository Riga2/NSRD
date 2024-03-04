@echo off
echo Running test on BistroX4...

set folders=0 1 2 3
set test_config=../configs/Bistro_test.txt

for %%i in (%folders%) do (
    echo Testing folder %%i ...
    python main.py --config %test_config% --test_folder %%i
)

echo All tests executed.
pause
