import argparse
import os

import sys
sys.path.append(".")

import diffuser.utils as utils
import yaml
from diffuser.utils.launcher_util import build_config_from_dict


def synthesize(Config):
    synthesizer = None
    Config.condition_guidance_w = getattr(Config, "condition_guidance_w", None)

    for load_step in Config.load_steps:
        ckpt_file_path = os.path.join(
            Config.log_dir, f"checkpoint/state_{load_step}.pt"
        )
        if not os.path.exists(ckpt_file_path):
            print(f"Checkpoint file {ckpt_file_path} not found. Skipping evaluation.")
            continue

        results_file_path = os.path.join(
            Config.log_dir,
            f"syn_results/step_{load_step}-ddim"
            if getattr(Config, "use_ddim_sample", False)
            else f"syn_results/step_{load_step}",
        )

        if Config.check_discrepancy:
            results_file_path = results_file_path + "-check_discrepancy"

        if Config.partially_noise:
            results_file_path = results_file_path + "-partially_noise"

        if Config.test_ret:
            results_file_path = results_file_path + f"-test_ret_{Config.test_ret}"

        if Config.include_labels:
            results_file_path = results_file_path + "-include_labels"

        if Config.use_composition_tech:
            results_file_path = results_file_path + "-use_composition_tech"

        if Config.condition_guidance_w is not None:
            results_file_path = results_file_path + f"-cg_{Config.condition_guidance_w}"
        
        if not Config.overwrite and os.path.exists(results_file_path):
            print(
                f"Results file {results_file_path} already exist. Skipping evaluation."
            )
            continue

        if synthesizer is None:
            synthesizer_config = utils.Config(
                Config.synthesizer, 
                n_gen=Config.n_gen,
                gen_batch_size=Config.gen_batch_size,
                check_discrepancy=Config.check_discrepancy,
                partially_noise=Config.partially_noise,
                times_of_regen_upper_limit=Config.times_of_regen_upper_limit,
                total_times_of_regen_upper_limit=Config.total_times_of_regen_upper_limit,
                recon_threshold=Config.recon_threshold,
                max_path_length_stitch=getattr(Config, "max_path_length_stitch", None),
                each_ig_step_num=Config.each_ig_step_num,
                test_ret=Config.test_ret,
                include_labels=Config.include_labels,
                verbose=False,
                use_composition_tech=Config.use_composition_tech,
                use_legal_model=Config.use_legal_model,
            )
            synthesizer = synthesizer_config()
            synthesizer.init(
                log_dir=Config.log_dir,
                condition_guidance_w=Config.condition_guidance_w,
                use_ddim_sample=Config.use_ddim_sample,
                n_ddim_steps=Config.n_ddim_steps,
                test_ret=Config.test_ret,
                include_labels=Config.include_labels,
            )

        synthesizer.synthesize(load_step=load_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    parser.add_argument("-g", "--gpu", help="gpu id", type=int, default=0)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string, Loader=yaml.SafeLoader)
    Config = build_config_from_dict(exp_specs)

    synthesize(Config)
