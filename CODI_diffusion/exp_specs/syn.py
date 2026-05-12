import os
import yaml
import subprocess
import copy
import time
import datetime


# YAML file path
YAML_FILE_PATH = 'syn_spread.yaml'
model = ''

test_ret_lst = [0]
use_label = True
GPU_IDS = [0]
PARAMETER_COMBINATIONS = []
if not use_label:
    partially_noise_lst = [True]
    for i, test_ret in enumerate(test_ret_lst):
        for j, partially_noise in enumerate(partially_noise_lst):
            gpu_id = GPU_IDS[(i * len(partially_noise_lst) + j) % len(GPU_IDS)]
            PARAMETER_COMBINATIONS.append({
                'log_dir': model,
                'test_ret': test_ret,
                'partially_noise': partially_noise,
                'include_labels': False,
                'use_composition_tech': False,
                'gpu_id': gpu_id
            })
else:
    use_composition_tech_lst = [True]
    for i, test_ret in enumerate(test_ret_lst):
        for j, use_composition_tech in enumerate(use_composition_tech_lst):
            gpu_id = GPU_IDS[(i * len(use_composition_tech_lst) + j) % len(GPU_IDS)]
            PARAMETER_COMBINATIONS.append({
                'log_dir': model,
                'test_ret': test_ret,
                'partially_noise': False,
                'include_labels': True,
                'use_composition_tech': use_composition_tech,
                'gpu_id': gpu_id,
            })

# Ensure output directory exists
OUTPUT_DIR = 'syn_tmp'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_yaml(file_path):
    """Load YAML file content"""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def save_yaml(data, file_path):
    """Save data to YAML file"""
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

def run_experiment(params, yaml_content_backup):
    """Run single experiment (in separate process)"""
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        gpu_id = params['gpu_id']
        
        # Create temporary YAML file for each experiment
        temp_yaml_path = f"{OUTPUT_DIR}/syn_{timestamp}_{gpu_id}.yaml"
        
        # Create output log files for each experiment
        stdout_log_path = f"{OUTPUT_DIR}/syn_{timestamp}_{gpu_id}_stdout.log"
        stderr_log_path = f"{OUTPUT_DIR}/syn_{timestamp}_{gpu_id}_stderr.log"
        
        # Copy backup content and modify
        data = copy.deepcopy(yaml_content_backup)
        
        # Modify variables section
        if 'variables' in data and 'log_dir' in data['variables']:
            data['variables']['log_dir'] = [params['log_dir']]
        
        # Modify constants section
        if 'constants' in data:
            if 'test_ret' in params:
                data['constants']['test_ret'] = params['test_ret']
            if 'partially_noise' in params:
                data['constants']['partially_noise'] = params['partially_noise']
            if 'include_labels' in params:
                data['constants']['include_labels'] = params['include_labels']
            if 'use_composition_tech' in params:
                data['constants']['use_composition_tech'] = params['use_composition_tech']
        
        # Save to temporary file
        save_yaml(data, temp_yaml_path)
        
        print(f"GPU {gpu_id}: Created temporary config file {temp_yaml_path}")
        print(f"GPU {gpu_id}: Parameters: {params}")
        print(f"GPU {gpu_id}: Stdout will be saved to {stdout_log_path}")
        print(f"GPU {gpu_id}: Stderr will be saved to {stderr_log_path}")
        
        # Run experiment command (non-blocking)
        command = f"python run_experiment.py -e {temp_yaml_path} -g {gpu_id}"
        
        print(f"GPU {gpu_id}: Starting experiment...")
        print(f"GPU {gpu_id}: Executing command: {command}")
        
        # Use subprocess.Popen for non-blocking execution, redirect output to files
        with open(stdout_log_path, 'w') as stdout_file, open(stderr_log_path, 'w') as stderr_file:
            process = subprocess.Popen(
                command, 
                shell=True, 
                executable="/bin/bash", 
                cwd='/home/madits',
                stdout=stdout_file,
                stderr=stderr_file
            )
        
        # Return immediately, do not wait for process to finish
        return process, gpu_id, temp_yaml_path, stdout_log_path, stderr_log_path
        
    except Exception as e:
        print(f"GPU {params['gpu_id']}: Error starting experiment: {e}")
        return None, params['gpu_id'], None, None, None

def monitor_processes(processes_info):
    """Monitor all process status"""
    while True:
        all_done = True
        active_processes = []
        
        for process_info in processes_info:
            process, gpu_id, temp_yaml_path, stdout_log_path, stderr_log_path = process_info
            
            if process.poll() is None:  # Process still running
                all_done = False
                active_processes.append(process_info)
                # Can add real-time log viewing here if needed
            else:  # Process finished
                return_code = process.returncode
                if return_code == 0:
                    print(f"GPU {gpu_id}: Experiment completed successfully")
                    print(f"GPU {gpu_id}: Output log: {stdout_log_path}")
                else:
                    print(f"GPU {gpu_id}: Experiment failed, return code: {return_code}")
                    print(f"GPU {gpu_id}: Please check error log: {stderr_log_path}")
        
        # Update process list, keep only active processes
        processes_info[:] = active_processes
        
        if all_done:
            break
        
        time.sleep(10)  # Check every 10 seconds

def main():
    """Main function: sequentially start experiments, 2 seconds interval between each"""
    print(f"Starting sequential execution of {len(PARAMETER_COMBINATIONS)} experiment configurations, 2 seconds interval between each")
    
    # Load original YAML file content as backup
    try:
        yaml_content_backup = load_yaml(YAML_FILE_PATH)
        print("Successfully loaded original YAML config file")
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        return
    
    # Sequentially start experiments, 2 seconds interval between each
    processes_info = []
    for params in PARAMETER_COMBINATIONS:
        try:
            # Start single experiment
            process, gpu_id, temp_yaml_path, stdout_log_path, stderr_log_path = run_experiment(params, yaml_content_backup)
            if process:
                processes_info.append((process, gpu_id, temp_yaml_path, stdout_log_path, stderr_log_path))
                print(f"✓ GPU {gpu_id}: Experiment started (PID: {process.pid})")
            
            time.sleep(10) # set long time to avoid gpu oom
                
        except Exception as e:
            print(f"GPU {params.get('gpu_id', 'unknown')}: Failed to start: {e}")
    
    print(f"\nAll experiments started, starting process monitoring...")
    print("Press Ctrl+C to stop monitoring (but started experiments will continue running)")
    print(f"All output logs saved in: {OUTPUT_DIR}")
    
    # Monitor process status
    try:
        monitor_processes(processes_info)
    except KeyboardInterrupt:
        print("\nUser interrupted monitoring, but started experiments will continue running in background")
        print("Use 'nvidia-smi' command to check GPU usage")
        print("Use 'ps aux | grep run_experiment.py' to check process status")
        print(f"Output logs saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()