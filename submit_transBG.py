"""
Example submission script addapted from GraphINVENT (https://github.com/MolecularAI/GraphINVENT/tree/bdd69ffd11816f8781be9fc8f807750375f61809).
This can be used to train or sample from a TransBG model.
To run, type:
(transBG) ~/transBG$ python submit_transBG.py
"""
# load general packages and functions
import json
import sys
import os
from pathlib import Path
import subprocess
import time

#load parameters
from parameters import params

model_name = params["model_name"]

# define what you want to do for the specified job(s)
DATASET          = "QM9"               # dataset name in "./datasets/" #Note, add smaller version of mdqm9 for debugging
JOB_TYPE         = "energy"        # "likelihood", "energy" or "generate"
JOBDIR_START_IDX = 0                   # where to start indexing job dirs
N_JOBS           = 1                   # number of jobs to run per model
RESTART          = False               # whether or not this is a restart job
FORCE_OVERWRITE  = True                # overwrite job directories which already exist
JOBNAME          = f"{model_name}_{JOB_TYPE}"  # used to create a sub directory

# if running using SLURM sbatch, specify params below
USE_SLURM = True                        # use SLURM or not
RUN_TIME  = "5-00:00:00"               # hh:mm:ss
MEM_GB    = 32                          # required RAM in GB
run_in = 'gpu'                       # run in gpu or core (cpu)

# for SLURM jobs, set partition to run job on (preprocessing jobs run entirely on
# CPU, so no need to request GPU partition; all other job types benefit from running
# on a GPU)
if run_in == 'gpu':
    PARTITION = "gpu"
    CPUS_PER_TASK = 4
else:
    PARTITION = "core"
    CPUS_PER_TASK = 1

params["job_type"] = JOB_TYPE

# set paths here
HOME             = str(Path.home())
PYTHON_PATH      = "~/.conda/envs/transBG-env/bin/python"
#PYTHON_PATH      = f"{HOME}/miniconda3/envs/transBG-env/bin/python"
TRANSBG_PATH = "./"
DATA_PATH        = "./datasets/"

if run_in== 'gpu':
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"


def submit():
    """
    Creates and submits submission script. Uses global variables defined at top
    of this file.
    """
    check_paths()

    # create an output directory
    model_name = params["model_name"]
    job_type_folder = JOB_TYPE + "_trainings" 
    model_output_path = f"./output_files/{job_type_folder}/{model_name}" 
    tensorboard_path    = f"./output_files/{job_type_folder}/{model_name}/tensorboard"    

    print(f"* Creating output directory {model_output_path}/", flush=True)
    os.makedirs(model_output_path, exist_ok=True)
    os.makedirs(tensorboard_path, exist_ok=True)

    # submit `N_JOBS` separate jobs
    jobdir_end_idx = JOBDIR_START_IDX + N_JOBS
    for job_idx in range(JOBDIR_START_IDX, jobdir_end_idx):

        # specify and create the job subdirectory if it does not exist
        params["job_dir"]         = f"{model_output_path}/job_{job_idx}/"
        params["tensorboard_dir"] = f"{tensorboard_path}/job_{job_idx}/"

        # create the directory if it does not exist already, otherwise raises an
        # error, which is good because *might* not want to override data our
        # existing directories!
        os.makedirs(params["tensorboard_dir"], exist_ok=True)
        try:
            job_dir_exists_already = bool(
                JOB_TYPE in ["generate", "test"] or FORCE_OVERWRITE
            )
            os.makedirs(params["job_dir"], exist_ok=job_dir_exists_already)
            print(
                f"* Creating model subdirectory {model_output_path}/job_{job_idx}/",
                flush=True,
            )
        except FileExistsError:
            print(
                f"-- Model subdirectory {model_output_path}/job_{job_idx}/ already exists.",
                flush=True,
            )
            if not RESTART:
                continue

        # write the `input.json` file
        write_input_json(params_dict=params, filename="input.json")

        # write `submit.sh` and submit
        if USE_SLURM:
            print("* Writing submission script.", flush=True)
            write_submission_script(job_dir=params["job_dir"],
                                    job_idx=job_idx,
                                    job_type=params["job_type"],
                                    max_n_nodes=params["max_n_nodes"],
                                    runtime=RUN_TIME,
                                    mem=MEM_GB,
                                    ptn=PARTITION,
                                    cpu_per_task=CPUS_PER_TASK,
                                    python_bin_path=PYTHON_PATH)

            print("* Submitting job to SLURM.", flush=True)
            subprocess.run(["sbatch", params["job_dir"] + "submit.sh"], check=True)
        else:
            print("* Running job as a normal process.", flush=True)
            subprocess.run(["ls", f"{PYTHON_PATH}"], check=True)
            subprocess.run([f"{PYTHON_PATH}",
                            f"{TRANSBG_PATH}main.py",
                            "--job-dir",
                            params["job_dir"]],
                           check=True)

        # sleep a few secs before submitting next job
        print("-- Sleeping 2 seconds.")
        time.sleep(2)


def write_input_json(params_dict : dict, filename : str="input.json"):
    """
    Writes job parameters/hyperparameters in `params_dict` to CSV using the specified 
    `filename`.
    """
    json_path = params_dict["job_dir"] + filename

    with open(json_path, "w") as json_file:
        json.dump(params_dict, json_file)


def write_submission_script(job_dir : str, job_idx : int, job_type : str, max_n_nodes : int,
                            runtime : str, mem : int, ptn : str, cpu_per_task : int,
                            python_bin_path : str):
    """
    Writes a submission script (`submit.sh`).
    Args:
    ----
        job_dir (str)         : Job running directory.
        job_idx (int)         : Job idx.
        job_type (str)        : Type of job to run.
        max_n_nodes (int)     : Maximum number of nodes in dataset.
        runtime (str)         : Job run-time limit in hh:mm:ss format.
        mem (int)             : Gigabytes to reserve.
        ptn (str)             : Partition to use, either "core" (CPU) or "gpu" (GPU).
        cpu_per_task (int)    : How many CPUs to use per task.
        python_bin_path (str) : Path to Python binary to use.
    """
    submit_filename = job_dir + "submit.sh"
    model_name = params["model_name"]
    with open(submit_filename, "w") as submit_file:
        submit_file.write("#!/bin/bash\n")
        submit_file.write(f"#SBATCH --job-name={model_name}_{job_type}_{job_idx}\n")
        submit_file.write(f"#SBATCH --output={job_dir}output_{job_idx}\n")
        submit_file.write(f"#SBATCH --time={runtime}\n")
        submit_file.write(f"#SBATCH --mem={mem}g\n")
        submit_file.write(f"#SBATCH --partition={ptn}\n")
        submit_file.write("#SBATCH --nodes=1\n")
        submit_file.write(f"#SBATCH --cpus-per-task={cpu_per_task}\n")
        if ptn == "gpu":
            submit_file.write("#SBATCH --gres=gpu:volta:1\n")
        #submit_file.write("hostname\n")
        #submit_file.write("export QT_QPA_PLATFORM='offscreen'\n")
        submit_file.write("module load oelicense/1.0\n")
        submit_file.write(f"{python_bin_path} {TRANSBG_PATH}main.py --job-dir {job_dir}\n")
        #submit_file.write(f" > {job_dir}output.o${{SLURM_JOB_ID}}\n")


def check_paths():
    """
    Checks that paths to Python binary, data, and transBG are properly
    defined before running a job, and tells the user to define them if not.
    """
    for path in [PYTHON_PATH, TRANSBG_PATH, DATA_PATH]:
        if "path/to/" in path:
            print("!!!")
            print("* Update the following paths in `submit.py` before running:")
            print("-- `PYTHON_PATH`\n-- `TRANSBG_PATH`\n-- `DATA_PATH`")
            sys.exit(0)

if __name__ == "__main__":
    submit()