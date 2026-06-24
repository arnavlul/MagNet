import numpy as np
from scipy.interpolate import CubicSpline
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_ORBITS_DIR = BASE_DIR / "data_generation" / "raw_orbits"
RESAMPLED_DIR = BASE_DIR / "data_generation" / "resampled_orbits"

def process_particle(file_path):
    # Load the raw orbit data: [t, s, theta, phi, vpar, p_theta, p_phi]
    data = np.load(file_path)
    
    t = data[:, 0]
    # Ensure t is strictly increasing (remove exact duplicates)
    _, unique_indices = np.unique(t, return_index=True)
    unique_indices.sort()
    data = data[unique_indices]
    
    t = data[:, 0]
    coords = data[:, 1:]
    
    # Check if there are enough points to fit a cubic spline
    if len(t) < 4:
        return False
        
    # We want a uniform time step across all particles.
    # Total orbit time is 1.0d-3.
    # Let's resample to 1000 uniform points.
    num_points = 1000
    t_uniform = np.linspace(t[0], t[-1], num_points)
    
    # Create cubic spline
    try:
        # Unwrap angles to avoid spurious derivatives at the 0/2pi boundary
        # coords indices: 0:s, 1:theta, 2:phi, 3:vpar, 4:p_theta, 5:p_phi
        coords[:, 1] = np.unwrap(coords[:, 1])
        coords[:, 2] = np.unwrap(coords[:, 2])
        
        cs = CubicSpline(t, coords, axis=0)
        coords_uniform = cs(t_uniform)
        
        # Wrap angles back to [0, 2pi)
        coords_uniform[:, 1] = np.mod(coords_uniform[:, 1], 2 * np.pi)
        coords_uniform[:, 2] = np.mod(coords_uniform[:, 2], 2 * np.pi)
    except Exception as e:
        print(f"Spline failed for {file_path.name}: {e}")
        return False
        
    # Stack [t, s, theta, phi, vpar, p_theta, p_phi]
    resampled_data = np.column_stack((t_uniform, coords_uniform))
    
    save_path = RESAMPLED_DIR / file_path.name
    np.save(save_path, resampled_data)
    return True

if __name__ == "__main__":
    import shutil
    
    if RESAMPLED_DIR.exists():
        shutil.rmtree(RESAMPLED_DIR)
    RESAMPLED_DIR.mkdir(parents=True, exist_ok=True)
    
    files = list(RAW_ORBITS_DIR.glob("*.npy"))
    if not files:
        print("No raw orbits found!")
        exit(1)
        
    print(f"Found {len(files)} raw orbits. Resampling using cubic splines...")
    
    success_count = 0
    for file_path in files:
        if process_particle(file_path):
            success_count += 1
            
    print(f"Successfully resampled {success_count}/{len(files)} particles.")
    print(f"Resampled uniform trajectories saved to: {RESAMPLED_DIR}")
