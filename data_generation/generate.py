import numpy as np
import subprocess
from pathlib import Path
import shutil
import os
import multiprocessing as mp
import tempfile

# Constants
BASE_DIR = Path(__file__).resolve().parent.parent
GORILLA_DIR = BASE_DIR / "GORILLA"
GORILLA_EXE = GORILLA_DIR / "BUILD" / "test_gorilla_main.x.exe"

def setup_base_inputs():
    """Sets up the initial inputs for generating orbits"""
    input_dir = BASE_DIR / "data_generation" / "base_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    
    import re
    gorilla_plot_inp = input_dir / "gorilla_plot.inp"
    # Ensure it exists (copy from GORILLA/INPUT if not)
    if not gorilla_plot_inp.exists():
        shutil.copy(GORILLA_DIR / "INPUT" / "gorilla_plot.inp", gorilla_plot_inp)
        
    with open(gorilla_plot_inp, 'r') as f:
        content = f.read()
        
    content = re.sub(r'i_orbit_options\s*=\s*[0-9]+', 'i_orbit_options = 3', content)
    content = re.sub(r'boole_full_orbit\s*=\s*\.[a-zA-Z]+\.', 'boole_full_orbit = .true.', content)
    content = re.sub(r'n_skip_full_orbit\s*=\s*[0-9]+', 'n_skip_full_orbit = 1', content)
    content = re.sub(r'total_orbit_time\s*=\s*[0-9.dD+-]+', 'total_orbit_time = 1.0d-3', content)
    
    with open(gorilla_plot_inp, 'w') as f:
        f.write(content)

def run_batch(batch_idx, batch_size=250):
    """Runs a batch of particles through a single Fortran execution to eliminate Grid overhead"""
    temp_path = BASE_DIR / "data_generation" / f"temp_batch_{batch_idx}"
    
    if temp_path.exists():
        shutil.rmtree(temp_path, ignore_errors=True)
    shutil.copytree(BASE_DIR / "data_generation" / "base_inputs", temp_path)
    
    # 1. Generate starting coordinates for this batch
    np.random.seed(42 + batch_idx)
    s_start = np.random.uniform(0.1, 0.9, batch_size)
    theta_start = np.random.uniform(0.0, 2*np.pi, batch_size)
    phi_start = np.random.uniform(0.0, 2*np.pi, batch_size)
    pitch_start = np.random.uniform(-0.9, 0.9, batch_size)
    
    # 2. Write them to orbit_start_sthetaphilambda.dat
    start_file = temp_path / "orbit_start_sthetaphilambda.dat"
    with open(start_file, "w") as f:
        for i in range(batch_size):
            f.write(f"{s_start[i]} {theta_start[i]} {phi_start[i]} {pitch_start[i]}\n")
            
    # 3. Run GORILLA
    env = os.environ.copy()
    env["PATH"] = r"C:\msys64\mingw64\bin;C:\msys64\usr\bin;" + env.get("PATH", "")
    env["OMP_NUM_THREADS"] = "1"  # Strict 1-core limit per batch
    
    try:
        subprocess.run(
            [str(GORILLA_EXE)],
            cwd=str(temp_path),
            env=env,
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Batch {batch_idx} Failed. STDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
        return False
        
    # 4. Parse the massive single output file
    orbit_file = temp_path / "full_orbit_plot_sthetaphi.dat"
    if not orbit_file.exists():
        return False
        
    try:
        raw_data = np.loadtxt(orbit_file, comments=['!', '#', 'C', 'c'])
        if raw_data.size == 0 or raw_data.ndim != 2:
            return False
            
        t_col = raw_data[:, 0]
        start_indices = np.where(np.diff(t_col, prepend=-1) < 0)[0]
        start_indices = np.append(start_indices, len(raw_data))
        
        saved_particles = 0
        for i in range(len(start_indices) - 1):
            p_idx = batch_idx * batch_size + i
            p_data = raw_data[start_indices[i]:start_indices[i+1]]
            
            _, unique_indices = np.unique(p_data[:, 0], return_index=True)
            p_data = p_data[unique_indices, :]
            
            if len(p_data) < 10:
                continue
                
            save_path = BASE_DIR / "data_generation" / "raw_orbits" / f"particle_{p_idx:04d}.npy"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, p_data)
            saved_particles += 1
            
        print(f"Batch {batch_idx}: saved {saved_particles} particles.")
        
    except Exception as e:
        print(f"Batch {batch_idx} exception parsing: {e}")
        return False
        
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)
        
    return True

def run_test_wrapper(batch_idx):
    return run_batch(batch_idx, batch_size=1000)

if __name__ == "__main__":
    setup_base_inputs()
    
    num_particles = 1000
    batch_size = 1000
    num_batches = 1
    num_cores = 1  # Strictly 1 core to cap RAM usage at ~4.6 GB total
    
    # Delete previous dummy raw orbits
    raw_dir = BASE_DIR / "data_generation" / "raw_orbits"
    if raw_dir.exists():
        shutil.rmtree(raw_dir, ignore_errors=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting data generation with {num_cores} core. 1 monolithic batch of {batch_size} particles...")
    
    start_time = __import__("time").time()
    
    # Run sequentially instead of using a pool, to avoid multiprocess overhead
    success = run_test_wrapper(0)
            
    print(f"\nGeneration complete in {__import__('time').time() - start_time:.1f}s!")
    
    # Count how many files were actually saved
    actual_files = len(list(raw_dir.glob("*.npy")))
    print(f"Successfully generated {actual_files}/{num_particles} viable particles.")
    print("All individual raw particle files are safely stored in: data_generation/raw_orbits/")
