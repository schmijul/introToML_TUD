import subprocess
import time
import numpy as np

def run_script(script, runs):
    times = []
    for _ in range(runs):
        start = time.time()
        subprocess.run(["python", script])
        end = time.time()
        times.append(end - start)
    return np.mean(times)

def compare_scripts(script1, script2, runs=5):
    time1 = run_script(script1, runs)
    print(f"Script {script1} average execution time over {runs} runs: {time1} seconds")

    time2 = run_script(script2, runs)
    print(f"Script {script2} average execution time over {runs} runs: {time2} seconds")

    print(f"\nScript {script1} was faster by {abs(time1 - time2)} seconds on average") if time1 < time2 else print(f"\nScript {script2} was faster by {abs(time1 - time2)} seconds on average")

if __name__ == "__main__":
    compare_scripts("numpy_script.py", "pytorch_script.py")
