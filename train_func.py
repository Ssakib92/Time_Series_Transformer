# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 10:08:51 2022

@author: Sadman Sakib
"""
import torch
from tqdm import tqdm

def train_step(data_loader, model, loss_function, optimizer):
    
    num_batches = len(data_loader)
    total_loss = 0
    
    model.train()
    
    for X, y, y_prev in tqdm(data_loader, desc = 'training step', leave = False):
        output = model(X, y_prev)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / num_batches
    
    return train_loss


def test_step(data_loader, model, loss_function):

    num_batches = len(data_loader)
    total_loss = 0
    
    model.eval()
    
    with torch.no_grad():
        for X, y, y_prev in tqdm(data_loader, desc = 'validation step', leave = False):
            
            y_star = model(X, y_prev)
            loss = loss_function(y_star, y)
            
            total_loss += loss.item()
            
    test_loss = total_loss / num_batches
    
    return test_loss


def train(model: torch.nn.Module, 
          train_loader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_function: torch.nn.Module,
          epochs: int, 
          checkpoint_name: dict = {"model": "model.pth",
                                   "model_states": "model_states_dict.pth"},
          val_loader: torch.utils.data.DataLoader = None):
    
    best_loss = 10000
    # Create empty results dictionary
    results = {"train_loss": [],"val_loss": []}

        # Loop through training and testing steps for a number of epochs
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}..")
            
        train_loss = train_step(model=model,
                                    data_loader=train_loader,
                                    loss_function=loss_function,
                                    optimizer=optimizer)
            
        if val_loader is not None:
            val_loss = test_step(model=model, 
                                  data_loader=val_loader, 
                                  loss_function=loss_function)

        print(
        f"train_loss: {train_loss:.4f} | "
        f"val_loss: {val_loss:.4f} | ")
        
        if val_loss < best_loss:
            print(f"saving best checkpoint on: epoch {epoch + 1}")
            torch.save(model, checkpoint_name["model"])
            torch.save(model.state_dict(), checkpoint_name["model_states"])
            best_loss = val_loss

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)

        # Return the filled results at the end of the epochs
    return results
