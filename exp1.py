"""
Experiment 1: Test variation in training and test results for ResNet-18 on CIFAR10

This experiment runs multiple training sessions with random initialization and data shuffling
to observe overall performance variance.

Key features:
- Random network initialization for each run
- Random data shuffling for each run
- Multiple complete training sessions (default: 1000)
"""
import os
import random
import argparse
from typing import List, Tuple

import torch
import numpy as np
import torchvision
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader

import utils


def run_experiment(args: argparse.Namespace) -> None:
    """
    Run the experiment with the specified parameters.

    Args:
        args (argparse.Namespace): Command-line arguments containing experiment parameters.
    """
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up hyperparameters
    num_epochs = 10
    batch_size = 128
    learning_rate = 0.001

    # Set up data transformations and datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Set up logging variables and load existing data if available
    train_loss_curves, train_acc_curves, test_accs, test_losses = utils.load_existing_data(args.output_dir)

    # Calculate remaining iterations
    remaining_iterations = args.number_of_runs - len(train_loss_curves)

    # Run remaining iterations
    for i in range(remaining_iterations):
        print(f"Running experiment {len(train_loss_curves)+1}/{args.number_of_runs}")

        # Set random seeds
        torch.manual_seed(random.randint(0, 100000))
        np.random.seed(random.randint(0, 100000))
        random.seed(random.randint(0, 100000))

        model = utils.initialize_model(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

        single_run_train_loss, single_run_train_accs = utils.train_model(
            model, train_loader, criterion, optimizer, device, num_epochs
        )

        train_loss_curves.append(single_run_train_loss)
        train_acc_curves.append(single_run_train_accs)

        test_loss, test_acc = utils.test(model, test_loader, criterion, device)
        test_accs.append(test_acc)
        test_losses.append(test_loss)

        utils.save_experiment_data(args.output_dir, train_loss_curves, train_acc_curves, test_accs, test_losses)
        utils.print_data_sizes(train_loss_curves, train_acc_curves, test_accs, test_losses)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment 1: ResNet-18 on CIFAR10')
    parser.add_argument('--run_experiment', action='store_true', help='Run training')
    parser.add_argument('--plot', action='store_true', help='Plot results')
    parser.add_argument('--output_dir', type=str, default='exp1output', help='Directory to save output')
    parser.add_argument('--number_of_runs', type=int, default=1000, help='Number of runs for the experiment')
    args = parser.parse_args()

    if args.run_experiment:
        run_experiment(args)
        
    if args.plot:
        utils.plot_results(args)