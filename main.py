import torch
from train import train_model,evaluate_model
from model.calss2 import create_model
from data_loader import create_dataloaders
from config import calss2 as config
import os

if __name__ == "__main__":
    # Set parameters
    main_folder = config.dataset
    save_dir = config.save_dir
    model_name = config.model_name
    model = create_model(config)
    num_epochs = config.num_epochs
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    weight_decay = config.weight_decay
    patience = config.patience

    print("Training configuration:")
    print(f"  Data path: {main_folder}")
    print(f"  Model: {model_name}")
    print(f"  Number of epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Early stopping patience: {patience}")

    # Train model
    model, history, save_dir = train_model(
        main_folder=main_folder,
        model=model,
        model_name=model_name,
        save_dir = save_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience
    )

    print(f"\nTraining completed! Results saved to: {save_dir}")

    # Reload best model for evaluation
    print("\nLoading best model for evaluation...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Recreate data loaders
    train_loader, val_loader = create_dataloaders(
        main_folder=main_folder,
        batch_size=batch_size,
        num_workers=0,
        train_val_split=0.8
    )
    best_model = create_model(config)
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))
    best_model = best_model.to(device)

    # Evaluate training set and validation set
    print("\nTraining set evaluation results:")
    train_metrics, _, _, _ = evaluate_model(best_model, train_loader, device)
    for metric_name, value in train_metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    print("\nValidation set evaluation results:")
    val_metrics, _, _, _ = evaluate_model(best_model, val_loader, device)
    for metric_name, value in val_metrics.items():
        print(f"  {metric_name}: {value:.4f}")