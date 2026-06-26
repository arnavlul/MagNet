import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from network_classes import TokamakDataLoader, LASympNet, GSympNet
import json

def plot_rollout_results(t, true_states, pred_states, percent_errors, euclidean_errors=None):
    """
    Plots the percentage error, phase space projections, and a 3D geometric projection 
    in a single matplotlib window with subplots.
    """
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Percentage Error over steps
    ax1 = fig.add_subplot(2, 2, 1)
    steps = np.arange(len(t))
    ax1.plot(steps, percent_errors[:, 0], label=r'$P_\theta$')
    ax1.plot(steps, percent_errors[:, 1], label=r'$P_\phi$')
    ax1.plot(steps, percent_errors[:, 2], label=r'$\theta$')
    ax1.plot(steps, percent_errors[:, 3], label=r'$\phi$')
    if euclidean_errors is not None:
        ax1.plot(steps, euclidean_errors, label='Euclidean', color='k', linestyle=':')
    ax1.set_xlabel('Rollout Step')
    ax1.set_ylabel('Error (%)')
    ax1.set_title('Normalized Percentage Error over Time')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Poloidal Phase Space Projection (theta vs P_theta)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(true_states[:, 2], true_states[:, 0], 'k-', label='Ground Truth', linewidth=2)
    ax2.plot(pred_states[:, 2], pred_states[:, 0], 'r--', label='SympNet', linewidth=2)
    ax2.set_xlabel(r'Poloidal Angle $\theta$')
    ax2.set_ylabel(r'Poloidal Momentum $P_\theta$')
    ax2.set_title('Poloidal Phase Space Projection')
    ax2.legend()
    ax2.grid(True)

    # 3. Toroidal Phase Space Projection (phi vs P_phi)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(true_states[:, 3], true_states[:, 1], 'k-', label='Ground Truth', linewidth=2)
    ax3.plot(pred_states[:, 3], pred_states[:, 1], 'r--', label='SympNet', linewidth=2)
    ax3.set_xlabel(r'Toroidal Angle $\phi$')
    ax3.set_ylabel(r'Toroidal Momentum $P_\phi$')
    ax3.set_title('Toroidal Phase Space Projection')
    ax3.legend()
    ax3.grid(True)

    # 4. 3D Geometric Orbit Projection
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    # Map angles to a generic Cartesian torus for visualization (R0=3, r=1)
    R0, r = 3.0, 1.0
    
    X_true = (R0 + r * np.cos(true_states[:, 2])) * np.cos(true_states[:, 3])
    Y_true = (R0 + r * np.cos(true_states[:, 2])) * np.sin(true_states[:, 3])
    Z_true = r * np.sin(true_states[:, 2])
    
    X_pred = (R0 + r * np.cos(pred_states[:, 2])) * np.cos(pred_states[:, 3])
    Y_pred = (R0 + r * np.cos(pred_states[:, 2])) * np.sin(pred_states[:, 3])
    Z_pred = r * np.sin(pred_states[:, 2])
    
    ax4.plot(X_true, Y_true, Z_true, 'k-', label='Ground Truth')
    ax4.plot(X_pred, Y_pred, Z_pred, 'r--', label='SympNet')
    ax4.set_title('3D Geometric Tokamak Orbit')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()


class RolloutEvaluator:
    def __init__(self, data_dir="data_generation/resampled_orbits"):
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading dataset to extract physical normalizers...")
        data_manager = TokamakDataLoader(self.data_dir)
        _, _, train_dataset, _ = data_manager.get_loaders()
        self.mean = train_dataset.mean.to(self.device)
        self.std = train_dataset.std.to(self.device)
        self.std_np = train_dataset.x_raw.std(dim=0).numpy()

    def load_model(self, weights_path):
        weights_path = Path(weights_path)
        config_path = weights_path.with_name(weights_path.stem + "_config.json")
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"Loaded config: {config}")
            if config['net_type'] == 'LA':
                model = LASympNet(d=config['d'], num_layers=config['num_layers'], num_sublayers_per_linear=config['num_sublayers']).to(self.device)
            elif config['net_type'] == 'G':
                model = GSympNet(d=config['d'], n=config['n'], num_layers=config['num_layers']).to(self.device)
        else:
            print(f"No config file found at {config_path}. Falling back to hardcoded G-SympNet.")
            model = GSympNet(d=2, n=100, num_layers=10).to(self.device)
            
        model.load_state_dict(torch.load(weights_path, map_location=self.device))
        model.eval()
        return model

    def evaluate_trajectory(self, model, particle_file):
        data = np.load(particle_file)
        t = data[:, 0]
        state_data = data[:, 1:]
        # Rearranging data to fit the NN's format: [P_theta, P_phi, theta, phi]
        state_data = state_data[:, [4, 5, 1, 2]]
        true_states = torch.tensor(state_data, dtype=torch.float32).to(self.device)
        
        num_steps = len(true_states)
        pred_states = torch.zeros_like(true_states)
        pred_states[0] = true_states[0]
        
        current_state = (pred_states[0] - self.mean) / (self.std + 1e-8)
        
        with torch.no_grad():
            for i in range(1, num_steps):
                next_state = model(current_state.unsqueeze(0)).squeeze(0)
                pred_states[i] = (next_state * self.std) + self.mean
                current_state = next_state
                
        true_states_np = true_states.cpu().numpy()
        pred_states_np = pred_states.cpu().numpy()
        
        # Component-wise percentage errors
        percent_errors = (np.abs(true_states_np - pred_states_np) / (self.std_np + 1e-8)) * 100.0
        
        # Euclidean percentage error
        diff_norm = np.linalg.norm(true_states_np - pred_states_np, axis=1)
        std_norm = np.linalg.norm(self.std_np)
        euclidean_errors = (diff_norm / (std_norm + 1e-8)) * 100.0
        
        return t, true_states_np, pred_states_np, percent_errors, euclidean_errors


def main():
    DATA_DIR = "data_generation/resampled_orbits"
    TEST_FILE = "data_generation/resampled_orbits/particle_0000.npy" 
    
    print("Welcome to MagNet Extrapolation Tester")
    print("="*40)
    
    weights_path = Path("results/model_weights/G_trained_weights_1.pth")
    
    evaluator = RolloutEvaluator(DATA_DIR)
    
    print(f"\nLoading weights from {weights_path}...")
    model = evaluator.load_model(weights_path)
    
    print(f"\nLoading test particle: {TEST_FILE}")
    if not Path(TEST_FILE).exists():
        print(f"Error: Could not find {TEST_FILE}. Make sure the dataset is generated.")
        return
        
    print("Running auto-regressive extrapolation...")
    t, true_states, pred_states, percent_errors, euclidean_errors = evaluator.evaluate_trajectory(model, TEST_FILE)
    
    save_path = Path("results/rollout_predicted_path.npy")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, pred_states)
    print(f"Predicted trajectory saved to: {save_path}")
    
    print("Generating Rollout plots...")
    plot_rollout_results(t, true_states, pred_states, percent_errors, euclidean_errors)

if __name__ == "__main__":
    main()
