"""
Experiment 3: ResNet-18 on CIFAR10 with Fixed Network Initialization

This experiment tests how much training and test results vary for ResNet-18 on CIFAR10 when:
1. The network initialization is fixed for all runs.
2. The dataset is randomly shuffled for each run.

The aim is to isolate the effect of data order while keeping the network initialization constant.
"""
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models import resnet18

import utils


def run_experiment(args: argparse.Namespace) -> None:
    """
    Run the experiment with the specified parameters.

    Args:
        args (argparse.Namespace): Command-line arguments containing experiment parameters.
    """
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_epochs = 10
    batch_size = 128
    learning_rate = 0.001

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loss_curves, train_acc_curves, test_accs, test_losses = utils.load_existing_data(args.output_dir)

    remaining_iterations = args.number_of_runs - len(train_loss_curves)

    torch.manual_seed(42)  # Fixed seed for network initialization
    model = utils.initialize_model(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    initial_model_path = os.path.join(args.output_dir, 'initial_model_state.pth')

    for i in range(remaining_iterations):
        print(f"Running experiment {len(train_loss_curves)+1}/{args.number_of_runs}")

        train_dataset = utils.create_shuffled_dataset(full_train_dataset)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

        model.load_state_dict(torch.load(initial_model_path))

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
    import argparse
    parser = argparse.ArgumentParser(description='Experiment 3: ResNet-18 on CIFAR10 with fixed network initialization')
    parser.add_argument('--run_experiment', action='store_true', help='Run training')
    parser.add_argument('--plot', action='store_true', help='Plot results')
    parser.add_argument('--output_dir', type=str, default='exp3output', help='Directory to save output')
    parser.add_argument('--number_of_runs', type=int, default=1000, help='Number of runs for the experiment')
    args = parser.parse_args()

    if args.run_experiment:
        torch.manual_seed(42)
        initial_model = resnet18(num_classes=10)
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(initial_model.state_dict(), os.path.join(args.output_dir, 'initial_model_state.pth'))
        run_experiment(args)
        
    if args.plot:
        utils.plot_results(args)