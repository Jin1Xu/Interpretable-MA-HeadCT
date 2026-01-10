import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import time
from datetime import datetime
import json
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, f1_score
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler

# Import data loader from previous file
from data_loader import create_dataloaders


def compute_metrics(outputs, labels):
    """Compute classification metrics"""
    # Get predictions
    _, preds = torch.max(outputs, 1)
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Calculate probabilities (for AUC)
    probs = torch.softmax(outputs, dim=1)
    probs_np = probs[:, 1].cpu().numpy()  # Probability of positive class

    # Calculate various metrics
    acc = accuracy_score(labels_np, preds_np)
    rec = recall_score(labels_np, preds_np, zero_division=0)
    pre = precision_score(labels_np, preds_np, zero_division=0)
    f1 = f1_score(labels_np, preds_np, zero_division=0)

    # Calculate AUC (ensure both classes have samples)
    try:
        auc = roc_auc_score(labels_np, probs_np)
    except ValueError:
        auc = 0.0

    return {
        'acc': acc,
        'recall': rec,
        'precision': pre,
        'f1': f1,
        'auc': auc,
        'preds': preds_np,
        'probs': probs_np
    }

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train one epoch"""
    model.train()
    running_loss = 0.0
    all_outputs = []
    all_labels = []

    for batch_idx, batch in enumerate(dataloader):
        inputs = batch['image'].to(device)
        labels = batch['label'].to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Mixed precision training
        if scaler is not None:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Collect statistics
        running_loss += loss.item() * inputs.size(0)
        all_outputs.append(outputs.detach())
        all_labels.append(labels.detach())

        # Print progress
        if (batch_idx + 1) % 10 == 0:
            print(f'  Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

    # Calculate epoch total loss and metrics
    epoch_loss = running_loss / len(dataloader.dataset)
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(all_outputs, all_labels)

    return epoch_loss, metrics

def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            all_outputs.append(outputs)
            all_labels.append(labels)

    # Calculate epoch total loss and metrics
    epoch_loss = running_loss / len(dataloader.dataset)
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(all_outputs, all_labels)

    return epoch_loss, metrics

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, is_best, checkpoint_dir):
    """Save checkpoint"""
    # Ensure directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }

    # Save current checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint saved to: {checkpoint_path}')

    # If it's the best model, save separately
    if is_best:
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(model.state_dict(), best_model_path)
        print(f'Best model saved to: {best_model_path}')

def train_model(main_folder, model, model_name, save_dir, num_epochs=50, batch_size=4,
                learning_rate=1e-4, weight_decay=1e-5, patience=10):

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader = create_dataloaders(
        main_folder=main_folder,
        batch_size=batch_size,
        num_workers=0,  # Set to 0 on Windows
        train_val_split=0.8
    )

    model = model.to(device)

    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    scaler = GradScaler() if torch.cuda.is_available() else None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(f"{save_dir}", f"{model_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to: {save_dir}")

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_recall': [],
        'val_recall': [],
        'train_precision': [],
        'val_precision': [],
        'train_f1': [],
        'val_f1': [],
        'train_auc': [],
        'val_auc': [],
    }

    # Early stopping settings
    best_val_auc = 0.0
    best_epoch = 0
    early_stop_counter = 0

    print("\nStarting training...")
    print("-" * 60)

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        print(f"\nEpoch {epoch}/{num_epochs}")

        # Training phase
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )

        # Validation phase
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_loss)

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_metrics['acc'])
        history['val_acc'].append(val_metrics['acc'])
        history['train_recall'].append(train_metrics['recall'])
        history['val_recall'].append(val_metrics['recall'])
        history['train_precision'].append(train_metrics['precision'])
        history['val_precision'].append(val_metrics['precision'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
        history['train_auc'].append(train_metrics['auc'])
        history['val_auc'].append(val_metrics['auc'])

        # Calculate time
        epoch_time = time.time() - start_time

        # Print results
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train Accuracy: {train_metrics['acc']:.4f}, Val Accuracy: {val_metrics['acc']:.4f}")
        print(f"Train F1: {train_metrics['f1']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        print(f"Train AUC: {train_metrics['auc']:.4f}, Val AUC: {val_metrics['auc']:.4f}")
        print(f"Time: {epoch_time:.2f} seconds")

        # Check if it's the best model
        is_best = val_metrics['auc'] > best_val_auc
        if is_best:
            best_val_auc = val_metrics['auc']
            best_epoch = epoch
            early_stop_counter = 0

            # Save best model checkpoint
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {
                    'train': train_metrics,
                    'val': val_metrics,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                },
                is_best=True,
                checkpoint_dir=save_dir
            )
        else:
            early_stop_counter += 1
            # Save checkpoint regularly (every 5 epochs)
            if epoch % 5 == 0:
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    {
                        'train': train_metrics,
                        'val': val_metrics,
                        'train_loss': train_loss,
                        'val_loss': val_loss
                    },
                    is_best=False,
                    checkpoint_dir=save_dir
                )

        # Early stopping check
        if early_stop_counter >= patience:
            print(f"\nEarly stopping triggered! No improvement in val AUC for {patience} epochs")
            print(f"Best epoch: {best_epoch}, Best val AUC: {best_val_auc:.4f}")
            break

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best epoch: {best_epoch}, Best val AUC: {best_val_auc:.4f}")
    print("=" * 60)

    # Save final model
    final_model_path = os.path.join(save_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")

    # Save training configuration
    config = {
        'model_name': model_name,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'patience': patience,
        'best_epoch': best_epoch,
        'best_val_auc': best_val_auc,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'device': str(device),
        'timestamp': timestamp
    }

    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Training configuration saved to: {config_path}")

    return model, history, save_dir


def evaluate_model(model, dataloader, device):
    """Evaluate model performance"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    # Calculate various metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    }

    # Print confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    return metrics, all_preds, all_labels, all_probs


def print_model_summary(model, input_shape=(1, 128, 128, 128)):
    """Print model architecture summary"""
    print("=" * 60)
    print("Model Architecture Summary")
    print("=" * 60)

    # Create dummy input
    dummy_input = torch.randn(1, *input_shape)

    # Print layer details
    total_params = 0
    print(f"{'Layer (type)':<30} {'Output Shape':<25} {'Param #':<15}")
    print("-" * 75)

    # Track input shape
    current_shape = list(input_shape)
    current_shape.insert(0, 1)  # Add batch dimension

    def count_parameters(module):
        return sum(p.numel() for p in module.parameters())

    # Iterate through model layers
    for name, module in model.named_children():
        if hasattr(module, 'forward'):
            # Get output shape
            with torch.no_grad():
                output = module(torch.randn(1, *current_shape[1:]))
                output_shape = list(output.shape)

            # Count parameters
            params = count_parameters(module)
            total_params += params

            # Print layer info
            print(f"{name:<30} {str(output_shape):<25} {params:,}")

            # Update current shape
            current_shape = output_shape

    print("-" * 75)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("=" * 60)