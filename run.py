from rl4co.tasks.train import train
import os
import time
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
if __name__ == "__main__":
    time_start = time.time()
    train()
    time_end = time.time()
    time_sum = time_end - time_start
    print(time_sum)
