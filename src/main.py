import argparse
import os
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging
import json # Added import
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm # For progress bars
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
DEFAULT_SMILES_MODEL = "seyonec/ChemBERTa-zinc-base-v1"
DEFAULT_PROTEIN_MODEL = "facebook/esm2_t6_8M_UR50D"
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 5
DEFAULT_LR = 1e-4
DEFAULT_OUTPUT_DIR = "results"

# --- Data Loading ---
def load_kiba_data(data_dir="/data") -> pd.DataFrame:
    """Loads the KIBA dataset directly from the mounted CSV file."""
    logging.info("Loading KIBA dataset from local CSV...")
    # Define the expected path within the container
    csv_path = os.path.join(data_dir, "KIBA.csv") 
    
    if not os.path.exists(csv_path):
        logging.error(f"KIBA dataset not found at expected path: {csv_path}")
        raise FileNotFoundError(f"KIBA dataset not found at expected path: {csv_path}")

    try:
        # Directly load the CSV using pandas
        df = pd.read_csv(csv_path)
        # Ensure the necessary columns are present
        df.rename(columns={'Ki , Kd and IC50  (KIBA Score)': 'interaction'}, inplace=True)
        df.rename(columns={'compound_iso_smiles': 'smiles'}, inplace=True)
        logging.info(f"Successfully loaded KIBA data from {csv_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading KIBA data from {csv_path}: {e}")
        raise

# --- Embedding Generation ---
def generate_embeddings(sequences, model_name, tokenizer, model, device, batch_size, desc="Sequences"):
    """Generates embeddings for a list of sequences (SMILES or Protein) using a transformer model."""
    logging.info(f"Generating {desc} embeddings using {model_name}...")
    model.eval() # Ensure model is in evaluation mode
    all_embeddings = []
    num_sequences = len(sequences)

    with torch.no_grad():
        for i in range(0, num_sequences, batch_size):
            batch_sequences = sequences[i:i+batch_size]
            # Tokenize the batch
            inputs = tokenizer(batch_sequences, padding=True, truncation=True, return_tensors="pt", max_length=512) # Added max_length
            inputs = {k: v.to(device) for k, v in inputs.items()} # Move inputs to device

            # Get model outputs
            outputs = model(**inputs)

            # Mean pooling: average token embeddings across sequence length
            # outputs.last_hidden_state shape: (batch_size, sequence_length, hidden_size)
            # We need to average over the sequence_length dimension (dim=1)
            # Attention mask is crucial to only average non-padding tokens
            attention_mask = inputs['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9) # Avoid division by zero
            mean_pooled_embeddings = sum_embeddings / sum_mask

            all_embeddings.append(mean_pooled_embeddings.cpu()) # Move back to CPU to accumulate

            if (i // batch_size) % 10 == 0: # Log progress every 10 batches
                 logging.info(f"  Processed batch {i // batch_size + 1} / { (num_sequences + batch_size - 1) // batch_size }")

    logging.info(f"Finished generating {desc} embeddings.")
    return torch.cat(all_embeddings, dim=0)

# --- PyTorch Dataset ---
class KIBADataset(Dataset):
    """PyTorch Dataset for KIBA embeddings and interaction scores."""
    def __init__(self, smiles_embeddings, protein_embeddings, interactions):
        # Ensure inputs are torch tensors
        self.smiles_embeddings = torch.tensor(smiles_embeddings, dtype=torch.float32)
        self.protein_embeddings = torch.tensor(protein_embeddings, dtype=torch.float32)
        self.interactions = torch.tensor(interactions, dtype=torch.float32)

        # Combine embeddings
        self.combined_embeddings = torch.cat((self.smiles_embeddings, self.protein_embeddings), dim=1)

        logging.info(f"Dataset created with {len(self.interactions)} samples.")
        logging.info(f"Combined embedding dimension: {self.combined_embeddings.shape[1]}")

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        return self.combined_embeddings[idx], self.interactions[idx]

# --- Model Definition ---
class RegressionMLP(nn.Module):
    """Simple Multi-Layer Perceptron for regression."""
    def __init__(self, input_dim, hidden_dim1=512, hidden_dim2=256, dropout_rate=0.2):
        super(RegressionMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim2, 1) # Output layer predicts a single value (interaction score)

        logging.info(f"MLP initialized with input_dim={input_dim}, hidden_dims=[{hidden_dim1}, {hidden_dim2}], dropout={dropout_rate}")

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# --- Training Loop ---
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Trains the model for one epoch."""
    model.train() # Set model to training mode
    total_loss = 0.0
    num_batches = len(dataloader)

    # Use tqdm for progress bar
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch_idx, (features, targets) in enumerate(progress_bar):
        features, targets = features.to(device), targets.to(device).unsqueeze(1) # Ensure target is [batch_size, 1]

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(features)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Update progress bar description
        progress_bar.set_postfix({'batch_loss': f'{loss.item():.4f}'})

        # Log batch loss occasionally
        if batch_idx % (num_batches // 10) == 0: # Log approx 10 times per epoch
            logging.debug(f"  Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / num_batches
    logging.info(f"Training Epoch Finished. Average Loss: {avg_loss:.4f}")
    return avg_loss

# --- Evaluation ---
def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on the given dataset."""
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    all_preds = []
    all_targets = []
    num_batches = len(dataloader)

    # Use tqdm for progress bar
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

    with torch.no_grad(): # Disable gradient calculations
        for features, targets in progress_bar:
            features, targets = features.to(device), targets.to(device).unsqueeze(1)

            # Forward pass
            outputs = model(features)

            # Calculate loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Store predictions and targets for
    avg_loss = total_loss / num_batches
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()

    # Calculate metrics
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_preds)
    ci = concordance_index(all_targets, all_preds)

    logging.info(f"Evaluation Finished. Avg Loss: {avg_loss:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, CI: {ci:.4f}")
    return avg_loss, rmse, r2, ci, all_targets, all_preds # Return targets and preds for plotting

# --- Plotting Functions ---
def plot_loss_curves(train_losses, val_losses, output_dir):
    """Plots training and validation loss curves."""
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(output_dir, "loss_curves.png")
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Loss curves plot saved to {save_path}")

def plot_predictions_vs_actual(targets, predictions, output_dir):
    """Plots predicted vs. actual values."""
    plt.figure(figsize=(8, 8))
    plt.scatter(targets, predictions, alpha=0.5)
    # Add identity line
    min_val = min(np.min(targets), np.min(predictions))
    max_val = max(np.max(targets), np.max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (y=x)')
    plt.title('Predicted vs. Actual Interaction Scores (Test Set)')
    plt.xlabel('Actual Interaction Score')
    plt.ylabel('Predicted Interaction Score')
    plt.grid(True)
    plt.legend()
    save_path = os.path.join(output_dir, "predictions_vs_actual.png")
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Predictions vs. Actual plot saved to {save_path}")

# --- Main Execution ---
def main(args):
    """Main function to run the workflow."""
    logging.info("Starting KIBA LLM prediction workflow.")
    logging.info(f"Arguments: {args}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    if not torch.cuda.is_available():
        logging.warning("CUDA not available, running on CPU. This will be slow.")

    # Load data
    df = load_kiba_data()

    # Load Models and Tokenizers
    logging.info(f"Loading SMILES tokenizer and model: {args.smiles_model}")
    smiles_tokenizer = AutoTokenizer.from_pretrained(args.smiles_model)
    smiles_model = AutoModel.from_pretrained(args.smiles_model).to(device)

    logging.info(f"Loading protein tokenizer and model: {args.protein_model}")
    protein_tokenizer = AutoTokenizer.from_pretrained(args.protein_model)
    protein_model = AutoModel.from_pretrained(args.protein_model).to(device)

    # --- Generate Embeddings ---
    smiles_embeds = generate_embeddings(
        df['smiles'].tolist(),
        args.smiles_model,
        smiles_tokenizer,
        smiles_model,
        device,
        args.batch_size,
        desc="SMILES"
    ).numpy() # Convert to numpy for sklearn split

    protein_embeds = generate_embeddings(
        df['target_sequence'].tolist(),
        args.protein_model,
        protein_tokenizer,
        protein_model,
        device,
        args.batch_size,
        desc="Protein"
    ).numpy() # Convert to numpy for sklearn split

    logging.info(f"Generated SMILES embeddings shape: {smiles_embeds.shape}")
    logging.info(f"Generated protein embeddings shape: {protein_embeds.shape}")

    # Target variable
    y = df['interaction'].values

    # Split data indices (using indices is often better for large datasets before creating tensors)
    indices = np.arange(len(df))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

    # Create Datasets and DataLoaders
    train_dataset = KIBADataset(
        smiles_embeddings=smiles_embeds[train_indices],
        protein_embeddings=protein_embeds[train_indices],
        interactions=y[train_indices]
    )
    test_dataset = KIBADataset(
        smiles_embeddings=smiles_embeds[test_indices],
        protein_embeddings=protein_embeds[test_indices],
        interactions=y[test_indices]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    logging.info(f"Created DataLoaders: Train batches={len(train_loader)}, Test batches={len(test_loader)}")

    # Initialize Model
    input_dim = train_dataset.combined_embeddings.shape[1]
    model = RegressionMLP(input_dim=input_dim).to(device)

    # Define Loss and Optimizer
    criterion = nn.MSELoss() # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    logging.info("Model, Loss function, and Optimizer initialized.")

    # --- Training & Evaluation Loop ---
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_rmses = []
    val_r2s = []
    val_cis = [] # Added CI tracking
    final_test_targets = None
    final_test_preds = None

    logging.info(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        logging.info(f"--- Epoch {epoch+1}/{args.epochs} ---")

        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        # Evaluation
        val_loss, val_rmse, val_r2, val_ci, test_targets, test_preds = evaluate(model, test_loader, criterion, device)
        val_losses.append(val_loss)
        val_rmses.append(val_rmse)
        val_r2s.append(val_r2)
        val_cis.append(val_ci) # Added CI tracking

        # Store predictions from the last epoch for plotting
        if epoch == args.epochs - 1:
            final_test_targets = test_targets
            final_test_preds = test_preds

        # Save best model (based on validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"New best model saved to {model_save_path} (Val Loss: {best_val_loss:.4f})")

    logging.info("Training complete.")

    # --- Result Presentation ---
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Saving results and plots to {args.output_dir}")

    # Save final metrics
    final_results = {
        "final_test_loss": val_losses[-1],
        "final_test_rmse": val_rmses[-1],
        "final_test_r2": val_r2s[-1],
        "final_test_ci": val_cis[-1], # Added CI
        "best_validation_loss": best_val_loss,
        "training_losses_per_epoch": train_losses,
        "validation_losses_per_epoch": val_losses,
        "validation_rmse_per_epoch": val_rmses,
        "validation_r2_per_epoch": val_r2s,
        "validation_ci_per_epoch": val_cis, # Added CI
        "config": vars(args) # Save arguments used for this run
    }
    results_path = os.path.join(args.output_dir, "final_metrics.json")
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    logging.info(f"Final metrics saved to {results_path}")

    # Generate and save plots
    plot_loss_curves(train_losses, val_losses, args.output_dir)
    if final_test_targets is not None and final_test_preds is not None:
        plot_predictions_vs_actual(final_test_targets, final_test_preds, args.output_dir)
    else:
        logging.warning("Could not generate prediction vs actual plot (no test predictions found).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an LLM-based model for KIBA drug-target affinity prediction.")
    parser.add_argument("--smiles_model", type=str, default=DEFAULT_SMILES_MODEL, help="Hugging Face model name/path for SMILES embeddings.")
    parser.add_argument("--protein_model", type=str, default=DEFAULT_PROTEIN_MODEL, help="Hugging Face model name/path for protein sequence embeddings.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for embedding generation and training.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate for the optimizer.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save results and model checkpoints.")
    # Add more arguments as needed (e.g., seed, specific model hyperparameters)

    args = parser.parse_args()
    main(args)
