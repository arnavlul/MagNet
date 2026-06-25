import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

class GradientModule(nn.Module):

    def __init__(self, d, n, mode='up'):

        super().__init__()

        self.d = d # Dimensions
        self.n = n # Width
        self.mode = mode

        self.K = nn.Parameter(torch.randn(n, d) * 0.1)
        self.a = nn.Parameter(torch.ones(n) * 0.1)
        self.b = nn.Parameter(torch.zeros(n))

    def forward(self, x):
        
        p, q = torch.split(x, self.d, dim=-1)

        if self.mode == 'up':
            p_new = p + torch.matmul((torch.sigmoid(torch.matmul(q, self.K.transpose(0, 1)) + self.b) * self.a), self.K)
            q_new = q
        elif self.mode == 'low':
            p_new = p
            q_new = q + torch.matmul((torch.sigmoid(torch.matmul(p, self.K.transpose(0,1)) + self.b) * self.a), self.K)
        
        return torch.cat([p_new, q_new], dim=-1)

class GSympNet(nn.Module):

    def __init__(self, d, n, num_layers):

        super().__init__()

        self.network = nn.ModuleList()

        for i in range(num_layers):
            mode = 'up' if i % 2 == 0 else 'low'

            self.network.append(GradientModule(d, n, mode))
            
    def forward(self, x):
        for layer in self.network:
            x = layer(x)
        return x



class TokamakDataset(Dataset):
    def __init__(self, file_paths, mean=None, std=None):

        super().__init__()
        
        if mean is not None and std is not None:
            self.mean = mean
            self.std = std


        all_x = []
        all_y = []

        for file_path in file_paths:
            data = np.load(file_path)

            # Columns are: t, s, theta, phi, vpar, p_theta, p_phi
            # Drop the time column
            state_data = data[:, 1:]

            # Rearranging data to fit the NN's format: [P_theta, P_phi, theta, phi]
            state_data = state_data[:, [4, 5, 1, 2]]


            x_particle = state_data[:-1] # Everything except last
            y_particle = state_data[1:]  # Everything except first

            all_x.append(x_particle)
            all_y.append(y_particle)

        self.x_raw = torch.tensor(np.vstack(all_x), dtype=torch.float32)
        self.y_raw = torch.tensor(np.vstack(all_y), dtype=torch.float32)
        
        # We need to share the mean/std between train and test
        if not hasattr(self, 'mean') or not hasattr(self, 'std'):
            self.mean = self.x_raw.mean(dim=0)
            
            # Extract true standard deviations
            true_std = self.x_raw.std(dim=0) # [std_Ptheta, std_Pphi, std_theta, std_phi]
            std_Ptheta = true_std[0]
            std_Pphi = true_std[1]
            std_theta = true_std[2]
            std_phi = true_std[3]
            
            # Optimal Canonical Scaling Factor (c)
            # We want S_Ptheta * S_theta = S_Pphi * S_phi = c
            prod_theta = std_Ptheta * std_theta
            prod_phi = std_Pphi * std_phi
            c = torch.sqrt(prod_theta * prod_phi)
            
            # Distribute the scaling symmetrically across the conjugate pairs
            S_Ptheta = torch.sqrt(c * std_Ptheta / std_theta)
            S_theta  = torch.sqrt(c * std_theta / std_Ptheta)
            
            S_Pphi = torch.sqrt(c * std_Pphi / std_phi)
            S_phi  = torch.sqrt(c * std_phi / std_Pphi)
            
            self.std = torch.tensor([S_Ptheta, S_Pphi, S_theta, S_phi], dtype=torch.float32)
            
        self.x = (self.x_raw - self.mean) / (self.std + 1e-8)
        self.y = (self.y_raw - self.mean) / (self.std + 1e-8)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    


if __name__ == "__main__":
    data_directory = Path("data_generation/resampled_orbits")

    all_files = list(data_directory.glob("*.npy"))
    np.random.shuffle(all_files)

    split_index = int(0.8 * len(all_files))
    train_files = all_files[:split_index]
    test_files = all_files[split_index:]

    print("Loading dataset...")
    train_dataset = TokamakDataset(train_files)
    test_dataset = TokamakDataset(test_files, mean=train_dataset.mean, std=train_dataset.std)

    train_loader = DataLoader(train_dataset, batch_size=32768, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32768, shuffle=False)


    print(f"Total training pairs loaded: {len(train_dataset)}")
    print(f"Total testing pairs loaded: {len(test_dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    model = GSympNet(d=2, n=100, num_layers=10).to(device)

    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr = 0.01)

    epochs = 40
    print("Starting Training")
    
    # Dictionary to save the training history
    history = {
        'train_loss': [],
        'test_P_theta': [],
        'test_P_phi': [],
        'test_theta': [],
        'test_phi': []
    }

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimiser.zero_grad()

            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)

            loss.backward()
            optimiser.step()

            total_loss += loss.item()
    
        avg_loss = total_loss / len(train_loader)

        model.eval()
        
        # Accumulate squared errors per component for RMSE
        total_squared_errors = torch.zeros(4, device=device)
        total_samples = 0

        with torch.no_grad():
            for test_x, test_y in test_loader:
                test_x, test_y = test_x.to(device), test_y.to(device)
                test_predictions =  model(test_x)
                
                # Denormalize to get the true physical error
                data_std = train_dataset.std.to(device)
                data_mean = train_dataset.mean.to(device)
                
                physical_predictions = (test_predictions * data_std) + data_mean
                physical_targets = (test_y * data_std) + data_mean

                # Sum up the squared errors for this batch across each component
                squared_errors = ((physical_predictions - physical_targets) ** 2).sum(dim=0)
                total_squared_errors += squared_errors
                total_samples += test_y.size(0)
        
        # Calculate Component-wise Root Mean Squared Error (RMSE)
        component_rmse = torch.sqrt(total_squared_errors / total_samples)
        
        # Divide by true physical Standard Deviation to get NRMSE Percentage
        true_stds = train_dataset.x_raw.std(dim=0).to(device)
        component_percentage_error = (component_rmse / true_stds) * 100.0

        print(f"Epoch [{epoch + 1}/{epochs}] | Train Loss (Normalized MSE): {avg_loss:.6f}")
        print(f"  Test Error (%): P_theta: {component_percentage_error[0]:.3f}% | P_phi: {component_percentage_error[1]:.3f}% | theta: {component_percentage_error[2]:.3f}% | phi: {component_percentage_error[3]:.3f}%")

        # Save to history
        history['train_loss'].append(avg_loss)
        history['test_P_theta'].append(component_percentage_error[0].item())
        history['test_P_phi'].append(component_percentage_error[1].item())
        history['test_theta'].append(component_percentage_error[2].item())
        history['test_phi'].append(component_percentage_error[3].item())

    # Save history and model
    np.save("G_training_history.npy", history)
    torch.save(model.state_dict(), "GSympNet_Tokamak.pth")
    print("Training Complete! Model & History Saved")