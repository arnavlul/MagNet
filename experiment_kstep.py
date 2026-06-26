import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from network_classes import LASympNet, GSympNet

class TrajectoryDataset(Dataset):
    """A dataset that returns sequences of length k for multi-step training"""
    def __init__(self, file_paths, k_steps=5, mean=None, std=None):
        self.k = k_steps
        
        all_seqs = []
        
        for file_path in file_paths:
            data = np.load(file_path)
            state_data = data[:, 1:] # Drop time
            state_data = state_data[:, [4, 5, 1, 2]] # P_theta, P_phi, theta, phi
            
            # Create sliding windows of size k+1 (1 input, k targets)
            for i in range(len(state_data) - self.k):
                all_seqs.append(state_data[i : i + self.k + 1])
                
        self.raw_data = torch.tensor(np.array(all_seqs), dtype=torch.float32)
        
        if mean is None or std is None:
            flat_data = self.raw_data[:, 0, :] # Use only starting states for stats
            self.mean = flat_data.mean(dim=0)
            
            true_std = flat_data.std(dim=0)
            prod_theta = true_std[0] * true_std[2]
            prod_phi = true_std[1] * true_std[3]
            c = torch.sqrt(prod_theta * prod_phi)
            
            S_Ptheta = torch.sqrt(c * true_std[0] / true_std[2])
            S_theta  = torch.sqrt(c * true_std[2] / true_std[0])
            S_Pphi = torch.sqrt(c * true_std[1] / true_std[3])
            S_phi  = torch.sqrt(c * true_std[3] / true_std[1])
            self.std = torch.tensor([S_Ptheta, S_Pphi, S_theta, S_phi], dtype=torch.float32)
        else:
            self.mean = mean
            self.std = std
            
        self.data = (self.raw_data - self.mean) / (self.std + 1e-8)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # returns (x_0, [x_1, x_2, ... x_k])
        return self.data[idx, 0], self.data[idx, 1:]

def train_experiment(args):
    net_type, mode, lr_strategy, epochs = args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    prefix = f"[{net_type} | {mode} | {lr_strategy}]"
    print(f"{prefix} Started Training...")
    
    # 1. Setup Data
    data_dir = Path("data_generation/resampled_orbits")
    all_files = list(data_dir.glob("*.npy"))
    train_files = all_files[:int(0.8 * len(all_files))]
    
    max_k = 5
    dataset = TrajectoryDataset(train_files, k_steps=max_k)
    # Using a smaller batch size because sequences take more memory
    loader = DataLoader(dataset, batch_size=8192, shuffle=True) 
    
    # Determine base learning rates based on network type
    high_lr = 0.005 if net_type == 'LA' else 0.001
    low_lr = 0.0005 if net_type == 'LA' else 0.0001
    
    initial_lr = high_lr if lr_strategy in ['high', 'schedule'] else low_lr
    
    # 2. Setup Model configurations scaled for 3D fusion physics generalization
    d = 2
    if net_type == 'LA':
        num_layers, num_sublayers = 25, 10
        model = LASympNet(d=d, num_layers=num_layers, num_sublayers_per_linear=num_sublayers).to(device)
        config = {'net_type': 'LA', 'd': d, 'num_layers': num_layers, 'num_sublayers': num_sublayers}
    else: # 'G'
        n, num_layers = 200, 25
        model = GSympNet(d=d, n=n, num_layers=num_layers).to(device)
        config = {'net_type': 'G', 'd': d, 'n': n, 'num_layers': num_layers}
        
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    criterion = nn.MSELoss()
    
    # Build markdown log entirely in memory to avoid race conditions
    md_lines = [
        f"\n### {net_type} SympNet | Mode: {mode} | LR Strategy: {lr_strategy}\n",
        "| Epoch | k-steps | Error (%) |\n",
        "|---|---|---|\n"
    ]
    
    for epoch in range(epochs):
        # Handle LR scheduling
        if lr_strategy == 'schedule' and epoch == epochs // 2:
            msg = f"--> Scheduling LR drop to {low_lr}"
            md_lines.append(f"\n**{msg}**\n\n| Epoch | k-steps | Error (%) |\n|---|---|---|\n")
            for param_group in optimizer.param_groups:
                param_group['lr'] = low_lr
                
        model.train()
        total_loss = 0.0
        total_percent_err = 0.0
        
        # Determine current k based on mode
        if mode == '5-step':
            current_k = 5
        else:
            # Scale the curriculum proportionally to the total number of epochs
            epochs_per_step = max(1, epochs // 5)
            current_k = min(5, (epoch // epochs_per_step) + 1)
        
        for batch_x, batch_targets in loader:
            batch_x = batch_x.to(device)
            batch_targets = batch_targets.to(device) # Shape: [batch, 5, 4]
            
            optimizer.zero_grad()
            
            loss = 0
            batch_percent_err = 0
            current_state = batch_x
            
            # Autoregressive Rollout during training!
            for step in range(current_k):
                next_state = model(current_state)
                loss += criterion(next_state, batch_targets[:, step, :])
                
                # Compute RMSE percentage error from the normalized MSE
                with torch.no_grad():
                    step_mse = criterion(next_state, batch_targets[:, step, :])
                    step_err = torch.sqrt(step_mse).item() * 100.0
                    batch_percent_err += step_err
                    
                current_state = next_state # Feed prediction back in
                
            loss = loss / current_k # Average loss across the sequence
            batch_percent_err = batch_percent_err / current_k
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_percent_err += batch_percent_err
            
        epoch_err = total_percent_err / len(loader)
        
        # Only print every 10 epochs to keep terminal pretty
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"{prefix} Epoch {epoch+1}/{epochs} (k={current_k}) | Error: {epoch_err:.4f}%")
            
        md_lines.append(f"| {epoch+1} | {current_k} | {epoch_err:.4f}% |\n")

    # 3. Save Model Weights and Config for rollout_test.py
    import json
    save_dir = Path("results/model_weights")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    save_prefix = f"{net_type}_{mode.replace('-', '_')}_{lr_strategy}_lr_weights"
    weights_path = save_dir / f"{save_prefix}.pth"
    config_path = save_dir / f"{save_prefix}_config.json"
    
    torch.save(model.state_dict(), weights_path)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"{prefix} Completed and Saved!")
    return "".join(md_lines)

if __name__ == "__main__":
    import multiprocessing
    # Clear the log file at the start of a new run
    log_file = Path("results/experiment_losses.md")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w") as f:
        f.write("# MagNet K-Step Experiment Log\n\n")

    epochs = 40
    tasks = []
    for lr_strat in ['schedule']:
        for net in ['LA', 'G']:
            for mode in ['5-step', 'curriculum']:
                tasks.append((net, mode, lr_strat, epochs))
                
    print(f"Starting {len(tasks)} experiments. Running 3 in parallel...\n")
    
    # Run 3 in parallel
    # Use spawn to be safe with PyTorch CUDA in multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    with multiprocessing.Pool(processes=3) as pool:
        results = pool.map(train_experiment, tasks)
        
    # Write all perfectly formatted markdown logs at the end
    with open(log_file, "a") as f:
        for res in results:
            f.write(res)
            
    print(f"\nAll {len(tasks)} experiments finished successfully! Check {log_file} for full markdown report.")
