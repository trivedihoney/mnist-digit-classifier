import datetime 
import torch
import os
RANDOM_SEED = 42

def set_static_seed(seed : int = RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_model(model : torch.nn.Module, project : str):
    # Create dirs if they don't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists(f'models/{project}'):
        os.makedirs(f'models/{project}')
    model_name = model.__class__.__name__.lower()
    torch.save(model.state_dict(), f"models/{project}/{model_name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pth")

def get_next_experiment_number(runs_dir: str) -> int:
    if not os.path.exists(runs_dir):
        return 1
    existing_experiments = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
    experiment_numbers = [int(d.split('_')[1]) for d in existing_experiments if d.startswith('experiment_')]
    if not experiment_numbers:
        return 1
    return max(experiment_numbers) + 1