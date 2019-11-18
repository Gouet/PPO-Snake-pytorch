call C:\Users\Victor\Anaconda3\Scripts\activate.bat
call conda activate GYM_ENV_RL

# --scenario=BipedalWalkerHardcore-v2
# --scenario=BipedalWalkerHardcore-v2

python train.py --eval --scenario=Snake --load-episode-saved 1001
pause