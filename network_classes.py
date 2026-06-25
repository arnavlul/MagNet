import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path


class LinearSubLayer(nn.Module):
    def __init__(self, d, mode='up'): # d = Dimension of half the state
        
        super().__init__()
        self.d = d
        self.mode = mode

        
        self.A = nn.Parameter(torch.randn(d, d) * 0.01) # * 0.01 to keep initial weights small
        self.b = nn.Parameter(torch.zeros(2 * d))

    def forward(self, x):
        # x is the batch of particle states: shape: [batch_size, 2d]
        
        # Split the states in half
        p, q = torch.split(x, self.d, dim=-1)

        # Symmetric Matrix S = A + A^T
        S = self.A + self.A.transpose(0, 1)

        # Split the bias into p and q components
        b_p, b_q = torch.split(self.b, self.d, dim=-1)

        if self.mode == 'up':
            p_new = p + torch.matmul(q, S) + b_p
            q_new = q + b_q
        elif self.mode == 'low':
            p_new = p + b_p
            q_new = q + torch.matmul(p, S) + b_q
        
        return torch.cat([p_new, q_new], dim=-1)

class LinearModule(nn.Module):
    def __init__(self, d, num_sublayers): 
        super().__init__()
        
        self.layers = nn.ModuleList()

        for i in range(num_sublayers):
            # If i is even: up / odd: low
            mode = 'up' if i % 2 == 0 else 'low'

            # Append the sublayer to network
            self.layers.append(LinearSubLayer(d, mode))
    
    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
        return x
    
class ActivationModule(nn.Module):
    def __init__(self, d, mode='up'):

        super().__init__()

        self.d = d
        self.mode = mode

        self.a = nn.Parameter(torch.ones(d) * 0.01)

    def forward(self, x):

        p, q = torch.split(x, self.d, dim=-1)

        if self.mode == 'up':
            p_new = p + self.a * torch.sigmoid(q)
            q_new = q
        elif self.mode == 'low':
            p_new = p
            q_new = q + self.a * torch.sigmoid(p)

        return torch.cat([p_new, q_new], dim=-1)

class LASympNet(nn.Module):
    
    def __init__(self, d, num_layers=3, num_sublayers_per_linear=2):

        super().__init__()

        self.network = nn.ModuleList()

        # Always start with a linear module
        self.network.append(LinearModule(d, num_sublayers_per_linear))

        for i in range(num_layers):
            
            # Alternate modes for activation layer
            act_mode = 'up' if i % 2 == 0 else 'low'

            self.network.append(ActivationModule(d, act_mode))

            self.network.append(LinearModule(d, num_sublayers_per_linear))

    def forward(self, x):
        for layer in self.network:
            x = layer(x)
        return x


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
        

class TokamakDataLoader:
    def __init__(self, data_directory, batch_size=32768, train_split=0.8):
        self.data_directory = Path(data_directory)
        self.batch_size = batch_size
        self.train_split = train_split
        
        all_files = list(self.data_directory.glob("*.npy"))
        np.random.shuffle(all_files)
        
        split_index = int(self.train_split * len(all_files))
        self.train_files = all_files[:split_index]
        self.test_files = all_files[split_index:]
        
    def get_loaders(self):
        train_dataset = TokamakDataset(self.train_files)
        test_dataset = TokamakDataset(self.test_files, mean=train_dataset.mean, std=train_dataset.std)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, test_loader, train_dataset, test_dataset


class NetworkTrainer:
    def __init__(self, model, train_loader, test_loader, train_dataset, device, lr=0.001, save_name="Model"):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_dataset = train_dataset
        self.device = device
        self.save_name = save_name
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.history = {
            'train_loss': [], 'test_P_theta': [], 'test_P_phi': [], 
            'test_theta': [], 'test_phi': []
        }
        
    def train(self, epochs=40):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            
            for batch_x, batch_y in self.train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(self.train_loader)
            
            self.model.eval()
            total_squared_errors = torch.zeros(4, device=self.device)
            total_samples = 0
            
            with torch.no_grad():
                for test_x, test_y in self.test_loader:
                    test_x, test_y = test_x.to(self.device), test_y.to(self.device)
                    test_predictions = self.model(test_x)
                    
                    data_std = self.train_dataset.std.to(self.device)
                    data_mean = self.train_dataset.mean.to(self.device)
                    
                    physical_predictions = (test_predictions * data_std) + data_mean
                    physical_targets = (test_y * data_std) + data_mean
                    
                    squared_errors = ((physical_predictions - physical_targets) ** 2).sum(dim=0)
                    total_squared_errors += squared_errors
                    total_samples += test_y.size(0)
                    
            component_rmse = torch.sqrt(total_squared_errors / total_samples)
            true_stds = self.train_dataset.x_raw.std(dim=0).to(self.device)
            component_percentage_error = (component_rmse / true_stds) * 100.0
            
            print(f"Epoch [{epoch + 1}/{epochs}] | Train Loss (Normalized MSE): {avg_loss:.6f}")
            print(f"  Test Error (%): P_theta: {component_percentage_error[0]:.3f}% | P_phi: {component_percentage_error[1]:.3f}% | theta: {component_percentage_error[2]:.3f}% | phi: {component_percentage_error[3]:.3f}%")
            
            self.history['train_loss'].append(avg_loss)
            self.history['test_P_theta'].append(component_percentage_error[0].item())
            self.history['test_P_phi'].append(component_percentage_error[1].item())
            self.history['test_theta'].append(component_percentage_error[2].item())
            self.history['test_phi'].append(component_percentage_error[3].item())
            
        np.save(f"{self.save_name}_history.npy", self.history)
        torch.save(self.model.state_dict(), f"{self.save_name}.pth")
        print("Training Complete! Model & History Saved")
    
