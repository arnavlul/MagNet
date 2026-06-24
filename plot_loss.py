import numpy as np
import matplotlib.pyplot as plt

def plot_training_history(history_file="training_history.npy"):
    try:
        # Load the dictionary from the .npy file
        # allow_pickle=True is required to load python dictionaries saved via numpy
        history = np.load(history_file, allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading {history_file}: {e}")
        return

    epochs = range(1, len(history['train_loss']) + 1)

    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot 1: Training Loss (Normalized MSE) ---
    ax1.plot(epochs, history['train_loss'], 'k.-', label='Train Loss (Normalized MSE)')
    ax1.set_title("Training Convergence")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("MSE")
    ax1.grid(True)
    ax1.legend()

    # --- Plot 2: Testing Error (NRMSE Percentage) ---
    ax2.plot(epochs, history['test_P_theta'], 'b.-', label='P_theta Error (%)')
    ax2.plot(epochs, history['test_P_phi'], 'c.-', label='P_phi Error (%)')
    ax2.plot(epochs, history['test_theta'], 'r.-', label='Theta Error (%)')
    ax2.plot(epochs, history['test_phi'], 'm.-', label='Phi Error (%)')
    
    ax2.set_title("Physical Test Error (NRMSE)")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Deviation from Perfection (%)")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("training_plot.png", dpi=300)
    print("Plot saved to training_plot.png")
    plt.show()

if __name__ == "__main__":
    plot_training_history()
