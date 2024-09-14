"""

Hold some utilty methods for training, testing displaying results etc. 


"""
import os
import torch
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt



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



















#