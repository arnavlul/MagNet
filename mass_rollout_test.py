import numpy as np
from pathlib import Path
import random
import os
from rollout_test import RolloutEvaluator

def main():
    print("Welcome to MagNet Mass Rollout Tester")
    print("="*40)
    
    DATA_DIR = Path("data_generation/resampled_orbits")
    if not DATA_DIR.exists():
        print(f"Error: Data directory {DATA_DIR} does not exist.")
        return
        
    all_particles = list(DATA_DIR.glob("*.npy"))
    if not all_particles:
        print(f"No .npy files found in {DATA_DIR}")
        return
        
    try:
        user_input = input(f"Enter the number of trajectories to test on (max {len(all_particles)}): ")
        num_traj = int(user_input)
        if num_traj < 1 or num_traj > len(all_particles):
            print(f"Invalid number of trajectories. Defaulting to 1.")
            num_traj = 1
    except ValueError:
        print("Invalid input. Defaulting to 1 trajectory.")
        num_traj = 1
        
    test_particles = random.sample(all_particles, num_traj)
    print(f"\nRandomly selected {num_traj} trajectories for testing.")
    
    WEIGHTS_DIR = Path("results/model_weights")
    all_weights = list(WEIGHTS_DIR.glob("*.pth"))
    
    RESULTS_DIR = Path("results/mass_rollout")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    md_file = RESULTS_DIR / "mass_rollout_summary.md"
    
    # Initialize summary table if it doesn't exist
    if not md_file.exists():
        with open(md_file, "w") as f:
            f.write("# MagNet Mass Rollout Testing Summary\n\n")
            f.write(f"Aggregated errors evaluated over **{num_traj} trajectories**.\n\n")
            f.write("| Model | P_theta (Mean/Med/Max) % | P_phi (Mean/Med/Max) % | theta (Mean/Med/Max) % | phi (Mean/Med/Max) % | Euclidean (Mean/Med/Max) % |\n")
            f.write("|---|---|---|---|---|---|\n")
            
    # Initialize the evaluator
    evaluator = RolloutEvaluator(str(DATA_DIR))
    
    for weight_path in all_weights:
        model_name = weight_path.stem
        print(f"\n[{model_name}] Starting Evaluation...")
        
        # Check if this model has already been processed (allows for pausing/resuming)
        error_save_path = RESULTS_DIR / f"{model_name}_errors.npz"
        if error_save_path.exists():
            print(f"  -> Found existing results at {error_save_path.name}, skipping.")
            continue
            
        model = evaluator.load_model(weight_path)
        
        all_percent_errors = []
        all_euclidean_errors = []
        
        for i, particle_file in enumerate(test_particles):
            print(f"  -> [{i+1}/{num_traj}] Running extrapolation on {particle_file.name}...")
            _, _, _, p_err, e_err = evaluator.evaluate_trajectory(model, particle_file)
            all_percent_errors.append(p_err)
            all_euclidean_errors.append(e_err)
            
        # Convert to numpy arrays
        # all_percent_errors shape: [num_traj, num_steps, 4]
        # all_euclidean_errors shape: [num_traj, num_steps]
        all_percent_errors = np.array(all_percent_errors)
        all_euclidean_errors = np.array(all_euclidean_errors)
        
        # Save raw errors to disk immediately as compressed arrays
        np.savez_compressed(error_save_path, percent_errors=all_percent_errors, euclidean_errors=all_euclidean_errors)
        print(f"  -> Raw arrays saved to {error_save_path.name}")
        
        # Calculate summary statistics across both trajectories and time steps
        mean_p = all_percent_errors.mean(axis=(0, 1))
        med_p = np.median(all_percent_errors, axis=(0, 1))
        max_p = all_percent_errors.max(axis=(0, 1))
        
        mean_e = all_euclidean_errors.mean()
        med_e = np.median(all_euclidean_errors)
        max_e = all_euclidean_errors.max()
        
        def format_stats(mean, med, maximum):
            return f"{mean:.2f} / {med:.2f} / {maximum:.2f}"
            
        # Append results to the markdown file immediately
        row = f"| {model_name} "
        for i in range(4):
            row += f"| {format_stats(mean_p[i], med_p[i], max_p[i])} "
        row += f"| {format_stats(mean_e, med_e, max_e)} |\n"
        
        with open(md_file, "a") as f:
            f.write(row)
            
        print(f"  -> Summary appended to {md_file.name}")
        
    print("\nMass testing complete! Check results/mass_rollout/mass_rollout_summary.md for the report.")

if __name__ == "__main__":
    main()
