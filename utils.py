"""

Hold some utilty methods for training, testing displaying results etc. 


"""
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random
from typing import List, Tuple
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18
import torchvision
from torch import nn, optim


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Initialize progress bar
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}", unit="batch")
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # Update progress bar
        progress_bar.set_postfix({
            'epoch': epoch,
            'loss': f'{total_loss / (batch_idx + 1):.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return avg_loss, accuracy

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    # Initialize progress bar
    progress_bar = tqdm(test_loader, desc="Testing", unit="batch")
    
    with torch.no_grad():
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{test_loss / (progress_bar.n + 1):.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    print(f"Test - Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return avg_loss, accuracy

def plot_results(args):
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Load data
    train_loss_curves = np.load(f"{args.output_dir}/train_loss_curves.npy")
    train_acc_curves = np.load(f"{args.output_dir}/train_acc_curves.npy")
    test_accs = np.load(f"{args.output_dir}/test_accs.npy")
    test_losses = np.load(f"{args.output_dir}/test_losses.npy")

    # Set XKCD style for all plots
    plt.xkcd()

    # Plot 1: Training Loss Curves
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(len(train_loss_curves)):
        ax.plot(range(1, len(train_loss_curves[i])+1), train_loss_curves[i], alpha=0.3)
    ax.set_title('Training Loss Curves', fontsize=16)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'train_loss_curves_xkcd.png'), dpi=300)
    plt.close()

    # Plot 2: Training Accuracy Curves
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(len(train_acc_curves)):
        ax.plot(range(1, len(train_acc_curves[i])+1), train_acc_curves[i], alpha=0.3)
    ax.set_title('Training Accuracy Curves', fontsize=16)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'train_acc_curves_xkcd.png'), dpi=300)
    plt.close()

    # Plot 3: Test Accuracy Histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(test_accs, kde=True, color='skyblue', ax=ax)
    ax.set_title('Test Accuracy Distribution', fontsize=16)
    ax.set_xlabel('Accuracy (%)', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'test_acc_histogram_xkcd.png'), dpi=300)
    plt.close()

    # Plot 4: Test Loss Histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(test_losses, kde=True, color='salmon', ax=ax)
    ax.set_title('Test Loss Distribution', fontsize=16)
    ax.set_xlabel('Loss', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'test_loss_histogram_xkcd.png'), dpi=300)
    plt.close()

    print(f"XKCD-style plots saved in {plots_dir}.")

    # Reset to default style
    plt.rcdefaults()

def load_existing_data(output_dir: str) -> Tuple[List, List, List, List]:
    """
    Load existing experiment data if available.

    Args:
        output_dir (str): Directory containing the data files.

    Returns:
        Tuple[List, List, List, List]: Loaded data or empty lists if no data is found.
    """
    file_paths = [
        f"{output_dir}/train_loss_curves.npy",
        f"{output_dir}/train_acc_curves.npy",
        f"{output_dir}/test_accs.npy",
        f"{output_dir}/test_losses.npy"
    ]

    if all(os.path.exists(path) for path in file_paths):
        train_loss_curves = list(np.load(file_paths[0], allow_pickle=True))
        train_acc_curves = list(np.load(file_paths[1], allow_pickle=True))
        test_accs = list(np.load(file_paths[2], allow_pickle=True))
        test_losses = list(np.load(file_paths[3], allow_pickle=True))

        min_length = min(len(train_loss_curves), len(train_acc_curves), len(test_accs), len(test_losses))
        return (
            train_loss_curves[:min_length],
            train_acc_curves[:min_length],
            test_accs[:min_length],
            test_losses[:min_length]
        )
    
    return [], [], [], []

def create_shuffled_dataset(full_dataset: torchvision.datasets.CIFAR10) -> Subset:
    """
    Create a shuffled subset of the full dataset.

    Args:
        full_dataset (torchvision.datasets.CIFAR10): The full CIFAR10 dataset.

    Returns:
        Subset: A shuffled subset of the full dataset.
    """
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    return Subset(full_dataset, indices)

def train_model(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, device: torch.device, num_epochs: int) -> Tuple[List[float], List[float]]:
    """
    Train the model for a specified number of epochs.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training data.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer for the model.
        device (torch.device): Device to train on.
        num_epochs (int): Number of epochs to train for.

    Returns:
        Tuple[List[float], List[float]]: Lists of training losses and accuracies for each epoch.
    """
    single_run_train_loss = []
    single_run_train_accs = []

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        single_run_train_loss.append(train_loss)
        single_run_train_accs.append(train_acc)

    return single_run_train_loss, single_run_train_accs

def save_experiment_data(output_dir: str, train_loss_curves: List[List[float]], 
                         train_acc_curves: List[List[float]], test_accs: List[float], 
                         test_losses: List[float]) -> None:
    """
    Save experiment data to files.

    Args:
        output_dir (str): Directory to save the data.
        train_loss_curves (List[List[float]]): Training loss curves.
        train_acc_curves (List[List[float]]): Training accuracy curves.
        test_accs (List[float]): Test accuracies.
        test_losses (List[float]): Test losses.
    """
    np.save(f"{output_dir}/train_loss_curves.npy", train_loss_curves)
    np.save(f"{output_dir}/train_acc_curves.npy", train_acc_curves)
    np.save(f"{output_dir}/test_accs.npy", test_accs)
    np.save(f"{output_dir}/test_losses.npy", test_losses)

def print_data_sizes(train_loss_curves: List[List[float]], train_acc_curves: List[List[float]], 
                     test_accs: List[float], test_losses: List[float]) -> None:
    """
    Print the sizes of experiment data.

    Args:
        train_loss_curves (List[List[float]]): Training loss curves.
        train_acc_curves (List[List[float]]): Training accuracy curves.
        test_accs (List[float]): Test accuracies.
        test_losses (List[float]): Test losses.
    """
    print(f"Train loss curves: {len(train_loss_curves)}")
    print(f"Train acc curves: {len(train_acc_curves)}")
    print(f"Test accs: {len(test_accs)}")
    print(f"Test losses: {len(test_losses)}")

def initialize_model(device: torch.device) -> nn.Module:
    """
    Initialize the ResNet-18 model and move it to the specified device.

    Args:
        device (torch.device): The device to move the model to.

    Returns:
        nn.Module: Initialized ResNet-18 model.
    """
    model = resnet18(num_classes=10)
    return model.to(device)

#