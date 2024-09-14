"""
Experiment 2: 

Test how much training and test results vary for ResNet-18 on CIFAR10 when:
1. The dataset is randomly shuffled initially, but the order remains fixed for all runs.
2. The network is reinitialized for each run.

This experiment aims to isolate the effect of network initialization while keeping the data order constant.
"""
import os
import torch
import random 
import argparse
import numpy as np 
import torchvision 
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
import torchvision.transforms as transforms

import utils




def run_experiment(args):
    # Make output dir if doesnt exist 
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup hyperparameters 
    num_epochs = 10
    batch_size = 128
    learning_rate = 0.001
    # Setup data transformations, and datasets 
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Randomly shuffle the dataset once
    indices = list(range(len(train_dataset)))
    random.shuffle(indices)
    train_dataset = torch.utils.data.Subset(train_dataset, indices)
    
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Setup logging vars and load existing data if available
    train_loss_curves = []
    train_acc_curves = []
    test_accs = []
    test_losses = []

    file_paths = [
        f"{args.output_dir}/train_loss_curves.npy",
        f"{args.output_dir}/train_acc_curves.npy",
        f"{args.output_dir}/test_accs.npy",
        f"{args.output_dir}/test_losses.npy"
    ]

    if all(os.path.exists(path) for path in file_paths):
        train_loss_curves = list(np.load(file_paths[0], allow_pickle=True))
        train_acc_curves = list(np.load(file_paths[1], allow_pickle=True))
        test_accs = list(np.load(file_paths[2], allow_pickle=True))
        test_losses = list(np.load(file_paths[3], allow_pickle=True))

        # Cut to shortest length
        min_length = min(len(train_loss_curves), len(train_acc_curves), len(test_accs), len(test_losses))
        train_loss_curves = train_loss_curves[:min_length]
        train_acc_curves = train_acc_curves[:min_length]
        test_accs = test_accs[:min_length]
        test_losses = test_losses[:min_length]

    # Calculate remaining iterations
    remaining_iterations = args.number_of_runs - len(train_loss_curves)

    # Setup dataloaders (fixed for all iterations)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Go through remaining iterations
    for i in range(remaining_iterations):
        print(f"Running experiment {len(train_loss_curves)+1}/{args.number_of_runs}")

        # Set seed for network initialization
        torch.manual_seed(random.randint(0, 100000))

        # Initialize model 
        model = resnet18(num_classes=10)
        model = model.to(device)

        # Setup loss and optimizer 
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train for set number of epochs 
        single_run_train_loss = []
        single_run_train_accs = []

        for epoch in range(num_epochs):
            train_loss, train_acc = utils.train_epoch(model, train_loader, criterion, optimizer, device, epoch)
            single_run_train_loss.append(train_loss)
            single_run_train_accs.append(train_acc)
        train_loss_curves.append(single_run_train_loss)
        train_acc_curves.append(single_run_train_accs)
        
        # Now get test acc, loss 
        test_loss, test_acc = utils.test(model, test_loader, criterion, device)
        test_accs.append(test_acc)
        test_losses.append(test_loss)

        # Save all data after each iteration
        np.save(f"{args.output_dir}/train_loss_curves.npy", train_loss_curves)
        np.save(f"{args.output_dir}/train_acc_curves.npy", train_acc_curves)
        np.save(f"{args.output_dir}/test_accs.npy", test_accs)
        np.save(f"{args.output_dir}/test_losses.npy", test_losses)

        # Print out all sizes
        print(f"Train loss curves: {len(train_loss_curves)}")
        print(f"Train acc curves: {len(train_acc_curves)}")
        print(f"Test accs: {len(test_accs)}")
        print(f"Test losses: {len(test_losses)}")







if __name__ == '__main__':
    # Setup argparse
    parser = argparse.ArgumentParser(description='Experiment 2: ResNet-18 on CIFAR10 with fixed data order')
    parser.add_argument('--run_experiment', action='store_true', default=False, help='Run training')
    parser.add_argument('--plot', action='store_true', default=False, help='Plot results')
    parser.add_argument('--output_dir', type=str, default='exp2output', help='Directory to save output')
    parser.add_argument('--number_of_runs', type=int, default=1000, help='Number of runs for the experiment')
    args = parser.parse_args()
    if args.run_experiment:
        run_experiment(args)
        
    if args.plot:
        utils.plot_results(args)


























#