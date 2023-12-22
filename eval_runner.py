import os
import argparse
import torch
from eval import create_env, Eval
import fnmatch

def is_cropped(case_name):
    return 'uncropped' not in case_name.lower()


def find_weight_file(base_weights_path, orig_dir_name, env_name):
    weights_dir = os.path.join(base_weights_path, orig_dir_name)
    env_name_lower = env_name.lower()
    for file_name in os.listdir(weights_dir):
        if fnmatch.fnmatch(file_name.lower(), f'*{env_name_lower}*.pth'):
            # Return the first match
            return os.path.join(weights_dir, file_name)
    raise FileNotFoundError(f"No weight file found for environment '{env_name}' in directory '{weights_dir}'")

def run_evaluation(env_name, seed, device, steps, epsilon, base_weights_path, case, cropenv, output_dir):
    orig_dir_name = case
    save_dir_name = f'{orig_dir_name}_no_greedy'
    savepath = os.path.join(os.getcwd(), output_dir, save_dir_name)
    weights = find_weight_file(base_weights_path, orig_dir_name, env_name)

    if not os.path.isdir(savepath):
        os.makedirs(savepath)

    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available. Switching to CPU ...')
        device = 'cpu'
    else:
        device = torch.device(device)

    k = 4
    env = create_env(env_name, k=k, seed=seed, crop=cropenv)

    print(f'Environment: {env_name}, Case: {orig_dir_name}, Cropped: {cropenv}')
    print('Device:', device)

    Eval(env, env_name, weights_path=weights, n_steps=steps, seed=seed, savepath=savepath, device=device, epsilon=epsilon)

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', default=10_000, type=int)
    parser.add_argument('--epsilon', default=0.0, type=float)
    parser.add_argument('--base_dir', default='results') 
    parser.add_argument('--output_dir', default='no_greedy_eval_results')
    args = parser.parse_args()

    env_list = ['BeamRider', 'Breakout', 'Enduro', 'Pong', 'Qbert', 'Seaquest', 'SpaceInvaders']
    
    # Iterate over each subdirectory in the base directory
    for case in os.listdir(args.base_dir):
        case_path = os.path.join(args.base_dir, case)
        if os.path.isdir(case_path):
            cropenv = is_cropped(case)
            for env_name in env_list:
                run_evaluation(env_name, args.seed, args.device, 
                               args.steps, args.epsilon, args.base_dir, case, 
                               cropenv, args.output_dir)