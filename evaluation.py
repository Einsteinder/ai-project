import numpy as np
import tensorflow as tf


def accuracy(pred, gt):
    
    pred = pred>0.5
    gt = (gt == 1)
    
    acc = np.true_divide((pred == gt).sum(),gt.shape[0])
    
    return acc


def prec_rec (pred, gt):
    
    pred = pred>0.5
    gt = (gt == 1)
    
    #print pred, gt
    
    true_positive = np.logical_and(pred == True, gt == True).sum()
    false_positive = np.logical_and(pred == True, gt == False).sum()
    false_negative = np.logical_and(pred == False, gt == True).sum()
    true_negative = np.logical_and(pred == False, gt == False).sum()
    
    #print true_positive, false_positive, false_negative, true_negative
    
    if true_positive + false_positive == 0:
        prec = 1.0 
    else:
        prec = np.true_divide(true_positive, true_positive + false_positive) 
        
        
    if true_positive + false_negative == 0:
        rec = 1.0
    else:
        rec = np.true_divide(true_positive, true_positive + false_negative)
    
    if prec+rec == 0:
        f1 = 0.0
    else:    
        f1 = 2*prec*rec / (prec+rec)
    
    return prec, rec, f1


