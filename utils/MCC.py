import copy
import numpy as np
from scipy.optimize import linear_sum_assignment

def MCC(pred_latent : np.array, true_latent : np.array):
    '''
    Args:
    - pred_latent   : [ B, D ]
    - true_latent   : [ B, D ]
    '''
    # I added this part: the function should perhaps only return one result?
    batch_size      = pred_latent.shape[0]

    num_samples     = pred_latent.shape[0]
    dim             = pred_latent.shape[1]
    total_batches= int( num_samples / batch_size )  

    mcc_arr= []
    for batch_idx in range(total_batches):
        
        z_hat   = copy.deepcopy( pred_latent[ (batch_idx)*batch_size : (batch_idx+1)*batch_size ] )
        z       = copy.deepcopy( true_latent[ (batch_idx)*batch_size : (batch_idx+1)*batch_size ] )
        batch_idx += 1
        
        cross_corr= np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                cross_corr[i,j]= (np.cov( z_hat[:,i], z[:,j] )[0,1]) / ( np.std(z_hat[:,i])*np.std(z[:,j]) )

        cost= -1*np.abs(cross_corr)
        row_ind, col_ind= linear_sum_assignment(cost)
        score= 100*( -1*cost[row_ind, col_ind].sum() )/(dim)
        # print(-100*cost[row_ind, col_ind])
    
        mcc_arr.append(score)

        # I added this part: since we are returning only one element, there is no need to make it a list?
        mcc_arr = mcc_arr[0]

    return mcc_arr

pred_latent = np.random.randn(12, 4)
true_latent = np.random.randn(12, 4)
MCC(pred_latent, true_latent)