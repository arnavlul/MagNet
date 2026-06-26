import torch
from network_classes import TokamakDataLoader, NetworkTrainer, LASympNet, GSympNet

def main():
    # Hardcoded training directory
    DATA_DIR = "data_generation/resampled_orbits"
    
    print("Welcome to MagNet Dynamic Trainer")
    print("="*40)
    
    net_choice = input("Which network would you like to train? (LA/G): ").strip().upper()
    
    d = 2 # Fixed dimension for [P_theta, P_phi] / [theta, phi]
    
    if net_choice == 'LA':
        try:
            num_layers = int(input("Enter number of layers (depth) [default 25]: ") or 25)
            num_sublayers = int(input("Enter number of sublayers per linear layer [default 10]: ") or 10)
            lr = float(input("Enter learning rate [default 0.001]: ") or 0.001)
        except ValueError:
            print("Invalid input. Using defaults.")
            num_layers, num_sublayers, lr = 25, 10, 0.001
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LASympNet(d=d, num_layers=num_layers, num_sublayers_per_linear=num_sublayers).to(device)
        save_name = "LA"
        config = {'net_type': 'LA', 'd': d, 'num_layers': num_layers, 'num_sublayers': num_sublayers}
        
    elif net_choice == 'G':
        try:
            n = int(input("Enter width (n) [default 100]: ") or 100)
            num_layers = int(input("Enter number of layers (depth) [default 10]: ") or 10)
            lr = float(input("Enter learning rate [default 0.01]: ") or 0.01)
        except ValueError:
            print("Invalid input. Using defaults.")
            n, num_layers, lr = 100, 10, 0.01
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GSympNet(d=d, n=n, num_layers=num_layers).to(device)
        save_name = "G"
        config = {'net_type': 'G', 'd': d, 'n': n, 'num_layers': num_layers}
    else:
        print("Invalid choice. Exiting.")
        return
        
    try:
        epochs = int(input("Enter number of epochs [default 40]: ") or 40)
    except ValueError:
        epochs = 40
    
    print("\nLoading dataset...")
    data_manager = TokamakDataLoader(DATA_DIR)
    train_loader, test_loader, train_dataset, test_dataset = data_manager.get_loaders()
    
    print(f"Total training pairs loaded: {len(train_dataset)}")
    print(f"Total testing pairs loaded: {len(test_dataset)}")
    print(f"Training on {device}")
    print("Starting Training...\n")
    
    trainer = NetworkTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        train_dataset=train_dataset,
        device=device,
        lr=lr,
        save_name=save_name,
        config=config
    )
    
    trainer.train(epochs=epochs)

if __name__ == "__main__":
    main()
