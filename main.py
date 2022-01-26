"""
Main function for running transBG jobs.
Examples:
--------
 * If you define an "input.json" with desired job parameters in job_dir/:
   (transBG) ~/transBG$ python main.py --job_dir path/to/job_dir/
 * If you instead want to run your job using the submission scripts:
   (transBG) ~/transBG$ python submit-fine-tuning.py

This script is addapted from https://github.com/MolecularAI/GraphINVENT/blob/bdd69ffd11816f8781be9fc8f807750375f61809/graphinvent/main.py
"""
# load general packages and functions
import datetime
import json
import torch

# load transBG-specific functions
from utils.load_parameters import load_parameters
from utils.command_line_args import args
from transBG import TransBG


def main():
    """
    Defines the type of job (preprocessing, training, generation, testing, or
    fine-tuning), writes the job parameters (for future reference), and runs
    the job.
    """
    _ = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # fix date/time

    with open(args.job_dir+"input.json") as json_file:
        params_dict = json.load(json_file)

    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    params = Struct(**params_dict)

    # create an instance of a transBG object
    conformer_gen = TransBG(params)
    conformer_gen.build_model()

    job_type = params.job_type
    print(f"* Run mode: '{job_type}'", flush=True)

    if job_type == "likelihood":
        # train model with only likelihood-based learning using all the molecules in the dataset
        conformer_gen.train_likelihood()

    elif job_type == "energy":
        # fine-tune the model with energy based learning using a smaller set of molecules (energy_train_indices)
        conformer_gen.model.load_state_dict(torch.load(params.pre_trained_model))
        if params.finetune_l:
            conformer_gen.finetune_likelihood()
        conformer_gen.train_energy()

    else:
        raise NotImplementedError("Not a valid `job_type`.")


if __name__ == "__main__":
    main()

