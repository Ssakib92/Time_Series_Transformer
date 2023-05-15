# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 13:36:17 2022

@author: Sadman Sakib
"""
import torch
import numpy as np
import pandas as pd

def predict(model, X, y_prev, device, target_len=24):

    y_out = torch.tensor([]).to(device)
    X = X.to(device)
    y_prev = y_prev.to(device)
    
    model.eval()
    
    with torch.no_grad():
            
        encoder_outs = model.encode(X) 
        
        y_prev = y_prev[:,0].unsqueeze(-1)
        size = y_prev.shape[-1]
        trg_mask = torch.triu(torch.ones(size,size)).transpose(0,1).type(dtype = torch.uint8).to(device)
        
        for i in range(target_len):
            
            out = model.decode(y_prev , encoder_outs, trg_mask = trg_mask)
            print(out.shape)
            y_prev = out
            y_out = torch.cat((y_out, out), 1)
            
    return y_out
    

#  pred = np.concatenate((output[:,0], output[-1][1:],), axis=0)
            
# creating prediction df for plotting           
def create_pred_df(df, prediction, window):
    
    df['DateTime'] = df['Year'].map(str) + '-' + df['Month'].map(str) + '-' + df['Day'].map(str) + '-' + df['Hour'].map(str)
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d-%H')
    df = df.set_index('DateTime')
    df_pred = df[['Month', 'Day', 'GHI', 'location', 'sky_index']]
    df_pred = df_pred.rename(columns={"GHI": "actual"})
    df_pred = df_pred[8:]
    preds = prediction[0:-14]
    pred = np.concatenate((preds[:,0], preds[-1][1:],), axis=0)
    df_pred['prediction'] = pred
    
    return df_pred   
