"""
This is the master file that orchestrates the pipeline for a set of dates chosen and detailed in Chapter 6.2 of the main text.

This script executes the project stages in order for each specified date and optionally runs multiple dates on multiple cores in parallel to save time.

Pipeline stages:
---------------
01_ingest_spx.py
    Loads the raw SPX option data for a given date.
02_preprocess.py
    Cleans and transforms the raw data into model-ready inputs.
03_prior_model.py
    Fits and constructs the prior model.
04_gp_point_estimation.py
    Computes Gaussian-process point-estimate fits.
05_topology_scan.py
    Scan the surface for structural features / regime diagnostics and outputs an lml surface for diagnostics.
06_restart_sensitivity.py
    Assess sensitivity to initialisation/restarts of the model
07_gp_hmc.py
    Run Bayesian inference HMC-based calibration.
08_compare_results.py
    Compares the outputs of the competing model variants.
09_plots.py
    Generate figures and summary plots for the given date.

Notes
-----
- Each stage is executed as a separate Python process.
- stdout and stderr are captured so failures propagate cleanly.
- The date list can be extended to run multiple regimes in batch.
- Parallel execution is useful for research, but it should be tuned to the
  available machine resources. In our case, the number of cores that can be employed is around 5-6 without overheating.
"""

import os
import sys
import subprocess
import time
import threading
from concurrent.futures import ProcessPoolExecutor



def _spinner(message, stop_event):
    """
    Prints a spinner on the console.
    """
    dots = ["", ".", "..", "..."]
    i = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\r{message}{dots[i % 4]}")
        sys.stdout.flush()
        i += 1
        time.sleep(0.5)
    sys.stdout.write("\r" + " " * (len(message) + 3) + "\r")
    sys.stdout.flush()


def run_single_date(date, script_dir, python_cmd):
    """Run the full pipeline for one date.

    Parameters
    ----------
    date : str
        Target date in YYYY-MM-DD format.
    script_dir : str
        Directory containing the pipeline stage scripts.
    python_cmd : str
        Python executable used to launch the subprocesses.

    Returns
    -------
    bool
        True if all stages complete successfully, False otherwise.
    """
    stop_event = threading.Event()
    spinner_thread = threading.Thread(target=_spinner, args=(f"Starting {date}", stop_event))
    spinner_thread.start()

    try:
        subprocess.run([python_cmd, os.path.join(script_dir, "01_ingest_spx.py"), "--date", date], check=True,
                       capture_output=True)
        subprocess.run([python_cmd, os.path.join(script_dir, "02_preprocess.py"), "--date", date], check=True,
                       capture_output=True)
        subprocess.run([python_cmd, os.path.join(script_dir, "03_prior_model.py"), "--date", date], check=True,
                       capture_output=True)
        subprocess.run([python_cmd, os.path.join(script_dir, "04_gp_point_estimation.py"), "--date", date], check=True,
                       capture_output=True)
        subprocess.run([python_cmd, os.path.join(script_dir, "05_topology_scan.py"), "--date", date], check=True,
                       capture_output=True)
        subprocess.run([python_cmd, os.path.join(script_dir, "06_restart_sensitivity.py"), "--date", date], check=True,
                       capture_output=True)
        subprocess.run([python_cmd, os.path.join(script_dir, "07_gp_hmc.py"), "--date", date], check=True,
                       capture_output=True)
        subprocess.run([python_cmd, os.path.join(script_dir, "08_compare_results.py"), "--date", date], check=True,
                       capture_output=True)
        subprocess.run([python_cmd, os.path.join(script_dir, "09_plots.py"), "--date", date], check=True,
                       capture_output=True)
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error on {date}: {e}")
        # Decode and print the actual Python traceback from the failing script
        if e.stderr:
            print(f"Detailed Traceback:\n{e.stderr.decode('utf-8')}")
        if e.stdout:
            print(f"Standard Output before crash:\n{e.stdout.decode('utf-8')}")
        return False


    finally:
        stop_event.set()
        spinner_thread.join()
        print(f"Finished {date}")

def run_pipeline():
    """Run the pipeline for all configured dates."""
    regime_dates = [
        "2020-03-03",
        "2020-03-13",
        "2020-03-17",
        "2020-03-19",
        "2020-03-23",
        "2020-03-25",

        "2022-09-16",
        "2022-09-21",
        "2022-09-26",
        "2022-09-30",
        "2022-10-05",

        "2023-06-07",
        "2023-06-13",
        "2023-06-21",
        "2023-06-26",
        "2023-06-30",

        "2024-01-18",
        "2024-01-24",
        "2024-01-30",
        "2024-02-09",
        "2024-02-15",
    ]

    python_cmd = sys.executable
    script_dir = os.path.dirname(os.path.abspath(__file__))
    start_time = time.time()

    # Run several dates in parallel to reduce wall-clock time.
    # Adjust max_workers to match available CPU and memory.
    with ProcessPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(run_single_date, date, script_dir, python_cmd) for date in regime_dates]
        for future in futures:
            future.result()  # Wait for all to finish

    print(f"\nParallel Pipeline complete! Total time: {(time.time() - start_time) / 60:.2f} minutes.")


if __name__ == "__main__":
    run_pipeline()
