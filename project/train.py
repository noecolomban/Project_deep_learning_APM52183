from models.siamese_net import Siamese_Model
from models.base_model import Simple_Node_Embedding
import os
from pathlib import Path
import math
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from models import get_model, get_model_gen
from toolbox.losses import triplet_loss
from toolbox import metrics
from loaders.data_generator import QAP_Generator
from loaders.siamese_loaders import siamese_loader
from toolbox.metrics import all_losses_acc, accuracy_linear_assignment
from toolbox.utils import check_dir


def train_one_epoch(train_loader, model, optimizer, criterion, device):
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0
    n_batches = 0
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        raw_scores = model(data)
        loss = criterion(raw_scores, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches

def evaluate_accuracy(test_loader, model, criterion, device):
    """Evaluate model and return mean accuracy per graph."""
    model.eval()
    all_acc = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            raw_scores = model(data)
            acc = accuracy_linear_assignment(raw_scores, target, aggregate_score=False)
            all_acc += acc
    return np.array(all_acc)

def create_and_train_model(graph_type, noise_model, n_vertices, edge_density, 
                           training_noise, num_train, num_val, n_epochs, 
                           batch_size, lr, device, path_dataset):
    """Create a new Siamese FGNN and train it from scratch."""
    
    # Same architecture as the pretrained model
    new_model = Siamese_Model(
        original_features_num=2,
        num_blocks=2,
        in_features=64,
        out_features=64,
        depth_of_mlp=3
    ).to(device)
    
    new_criterion = triplet_loss()
    optimizer = torch.optim.Adam(new_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=False)
    
    param_count = sum(p.numel() for p in new_model.parameters())
    print(f"  Model parameters: {param_count:,}")
    
    # Generate training data
    train_args = {
        'generative_model': graph_type,
        'noise_model': noise_model,
        'edge_density': edge_density,
        'n_vertices': n_vertices,
        'vertex_proba': 1.0,
        'noise': training_noise,
        'num_examples_train': num_train,
        'num_examples_val': num_val,
    }
    
    print(f"  Generating training data ({num_train} examples)...")
    gene_train = QAP_Generator('train', train_args, path_dataset)
    gene_train.load_dataset()
    train_loader = siamese_loader(gene_train, batch_size, gene_train.constant_n_vertices)
    
    print(f"  Generating validation data ({num_val} examples)...")
    gene_val = QAP_Generator('val', train_args, path_dataset)
    gene_val.load_dataset()
    val_loader = siamese_loader(gene_val, batch_size, gene_val.constant_n_vertices)
    
    # Training loop
    print(f"  Training for {n_epochs} epochs...")
    train_losses = []
    val_accuracies = []
    best_acc = 0
    best_state = None
    
    for epoch in range(n_epochs):
        loss = train_one_epoch(train_loader, new_model, optimizer, new_criterion, device)
        train_losses.append(loss)
        
        val_acc = evaluate_accuracy(val_loader, new_model, new_criterion, device)
        mean_val_acc = np.mean(val_acc)
        val_accuracies.append(mean_val_acc)
        
        scheduler.step(loss)
        
        if mean_val_acc > best_acc:
            best_acc = mean_val_acc
            best_state = {k: v.clone() for k, v in new_model.state_dict().items()}
        
        print(f"    Epoch {epoch+1}/{n_epochs} — Loss: {loss:.4f}, Val Acc: {mean_val_acc:.3f}")
    
    # Load best model
    if best_state is not None:
        new_model.load_state_dict(best_state)
    print(f"  Best validation accuracy: {best_acc:.3f}")
    
    return new_model, train_losses, val_accuracies