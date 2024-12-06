import torch
import numpy as np
from solar_pinn_ideal import SolarPINN, PINNTrainer
from tqdm import tqdm

def generate_training_data(n_samples=1000, validation_split=0.2, return_full=False, batch_size=32):
    """Generate synthetic training data for ideal clear sky conditions."""
    # Generate random input parameters with normalization
    latitude = (torch.rand(n_samples) * 180 - 90).requires_grad_()  # -90 to 90 degrees
    longitude = (torch.rand(n_samples) * 360 - 180).requires_grad_()  # -180 to 180 degrees
    time = (torch.rand(n_samples) * 24).requires_grad_()  # 0 to 24 hours
    slope = (torch.rand(n_samples) * 90).requires_grad_()  # 0 to 90 degrees
    aspect = (torch.rand(n_samples) * 360).requires_grad_()  # 0 to 360 degrees

    # Normalize inputs to [-1, 1] range for better training
    x_data = torch.stack([
        latitude / 90,  # Normalize latitude
        longitude / 180,  # Normalize longitude
        time / 24,  # Normalize time
        slope / 90,  # Normalize slope
        aspect / 360  # Normalize aspect
    ], dim=1).float()

    # Generate target values using ideal clear sky model
    y_data = []
    physics_model = SolarPINN()  # Use the same model for generating ideal data
    
    # Process data in batches
    physics_model.train()  # Ensure model is in training mode
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        with torch.set_grad_enabled(True):
            batch = x_data[i:batch_end]
            if len(batch) < 2:  # Handle last incomplete batch
                batch = torch.cat([batch, batch], dim=0)  # Duplicate the sample
                irradiance = physics_model(batch)[:1]  # Take only the first prediction
            else:
                irradiance = physics_model(batch)
            y_data.append(irradiance)
    
    y_data = torch.cat(y_data).float().requires_grad_()
    
    if return_full:
        return x_data, y_data
    
    # Split into train and validation sets
    n_val = int(n_samples * validation_split)
    indices = torch.randperm(n_samples)
    
    train_indices = indices[:-n_val]
    val_indices = indices[-n_val:]
    
    return (x_data[train_indices], y_data[train_indices], 
            x_data[val_indices], y_data[val_indices])

def create_latitude_strata(x_data, n_strata=5):
    """Create stratified folds based on latitude ranges."""
    latitudes = x_data[:, 0] * 90  # Denormalize latitudes
    bins = torch.linspace(latitudes.min(), latitudes.max(), n_strata + 1)
    strata_indices = []
    
    for i in range(n_strata):
        stratum_mask = (latitudes >= bins[i]) & (latitudes < bins[i + 1])
        strata_indices.append(torch.where(stratum_mask)[0])
    
    return strata_indices

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Import physics validator
    from physics_validator import SolarPhysicsIdeal, calculate_metrics
    
    # Generate dataset with specified size
    print("Generating training data...")
    x_data, y_data = generate_training_data(n_samples=1000, validation_split=0.2, return_full=True)
    
    # Create stratified folds
    n_folds = 5
    strata_indices = create_latitude_strata(x_data, n_strata=n_folds)
    
    # Training parameters
    n_epochs = 1000  # Increased epochs as per manager's request
    batch_size = 32  # Reduced batch size for better updates
    best_models = []
    patience = 20
    
    print(f"Starting training with {len(x_data)} samples...")
    print(f"Training parameters:")
    print(f"- Epochs: {n_epochs}")
    print(f"- Batch size: {batch_size}")
    print(f"- Number of folds: {n_folds}")
    
    # K-fold cross validation
    for fold in range(n_folds):
        print(f"\nTraining Fold {fold + 1}/{n_folds}")
        
        # Create train/val split for this fold
        val_indices = strata_indices[fold]
        train_indices = torch.cat([strata_indices[i] for i in range(n_folds) if i != fold])
        
        x_train, y_train = x_data[train_indices], y_data[train_indices]
        x_val, y_val = x_data[val_indices], y_data[val_indices]
        
        print(f"Train samples: {len(x_train)}, Validation samples: {len(x_val)}")
        
        # Initialize new model and trainer for this fold
        model = SolarPINN()
        trainer = PINNTrainer(model, learning_rate=0.001, min_lr=0.0001)
        n_batches = len(x_train) // batch_size
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Create progress bar for epochs
        epoch_pbar = tqdm(range(n_epochs), desc=f"Fold {fold + 1}")
        
        for epoch in epoch_pbar:
            # Training
            model.train()
            epoch_loss = 0
            
            # Shuffle training data
            indices = torch.randperm(len(x_train))
            
            # Create progress bar for batches
            batch_pbar = tqdm(range(n_batches), leave=False, desc=f"Epoch {epoch + 1}")
            
            for i in batch_pbar:
                batch_indices = indices[i*batch_size:(i+1)*batch_size]
                x_batch = x_train[batch_indices]
                y_batch = y_train[batch_indices]
                
                # Ensure gradients are properly tracked
                x_batch.requires_grad_(True)
                y_batch.requires_grad_(True)
                
                loss = trainer.train_step(x_batch, y_batch)
                epoch_loss += loss
                
                # Update batch progress
                batch_pbar.set_postfix({'batch_loss': f'{loss:.4f}'})
            
            avg_train_loss = epoch_loss / n_batches
            
            # Validation
            model.eval()
            with torch.no_grad():
                try:
                    y_pred = model(x_val)
                    val_loss = torch.nn.functional.mse_loss(y_pred, y_val)
                    val_metrics = calculate_metrics(y_val, y_pred)
                    
                    # Update epoch progress bar
                    epoch_pbar.set_postfix({
                        'train_loss': f'{avg_train_loss:.4f}',
                        'val_loss': f'{val_loss:.4f}',
                        'r2': f'{val_metrics["r2"]:.4f}'
                    })
                    
                    # Print detailed metrics every epoch
                    if (epoch + 1) % 1 == 0:
                        current_lr = trainer.optimizer.param_groups[0]['lr']
                        print(f"\nEpoch [{epoch+1}/{n_epochs}]")
                        print(f"Learning Rate: {current_lr:.6f}")
                        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
                        print(f"Validation Metrics:")
                        print(f"MAE: {val_metrics['mae']:.2f} W/m²")
                        print(f"RMSE: {val_metrics['rmse']:.2f} W/m²")
                        print(f"R²: {val_metrics['r2']:.4f}")
                        
                        # Print sample predictions
                        sample_idx = torch.randint(0, len(x_val), (5,))
                        y_sample = model(x_val[sample_idx])
                        y_true = y_val[sample_idx]
                        print("\nSample Predictions vs True Values:")
                        for i in range(5):
                            print(f"Pred: {y_sample[i].item():.2f} W/m², True: {y_true[i].item():.2f} W/m²")
                    
                except Exception as e:
                    print(f"Validation error: {str(e)}")
                    val_loss = float('inf')
                    val_metrics = {'mae': float('inf'), 'rmse': float('inf'), 'r2': -float('inf')}
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model for this fold
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'scheduler_state_dict': trainer.scheduler.state_dict(),
                    'epoch': epoch,
                    'val_metrics': val_metrics,
                    'fold': fold
                }, f'best_solar_pinn_ideal_fold_{fold}.pth')
                best_models.append({
                    'fold': fold,
                    'metrics': val_metrics,
                    'model_path': f'best_solar_pinn_ideal_fold_{fold}.pth'
                })
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    # Print final results
    print("\nTraining completed!")
    print("\nBest model results for each fold:")
    for model_info in best_models:
        print(f"\nFold {model_info['fold'] + 1}:")
        print(f"R²: {model_info['metrics']['r2']:.4f}")
        print(f"RMSE: {model_info['metrics']['rmse']:.2f} W/m²")
        print(f"MAE: {model_info['metrics']['mae']:.2f} W/m²")

if __name__ == "__main__":
    main()
