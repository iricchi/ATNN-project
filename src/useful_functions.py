import numpy as np
from more_itertools import consecutive_groups


def pearsonr(x):
    """
    Simply computes correlation between all pairs of lines in X.
    It is assumed that X is in the form variables x observations.
    If X is in observations x variables it should be transposed before being passed to this function!
    """
    return np.corrcoef(x, rowvar=True)

def sliding_window_corr(x, window_length, window_stride, window_function):
    """
    
    """
    half_window=window_length//2
    window_f = window_function(window_length) # This serves to taper the samples
    start = half_window
    end = x.shape[0] - half_window
    
    n_windows = (end-start)//window_stride + 1 # Check that this is correct
    
    print('There should be {} windows'.format(n_windows))
    print('Start: {} - End: {}'.format(start, end))
    
    correlations = np.zeros((n_windows,x.shape[1], x.shape[1]))
    for i in range(0, n_windows):
        # The window is centered at i*window_stride+start
        center = i*window_stride + start
        # Left side is center - half_window
        # Right side is center + half_window
        # We compute correlation of current sample tampered with window function
        correlations[i]=pearsonr(x[center-half_window:center+half_window].T*window_f)
    return correlations

def 

def filter_out_smallrestingphases(locs, thr):
    """
    If the window is smaller than a specifirf threshold thr,
    then consider only consecutive indexes
    outputing the indexes of interest only   
    """
    ioi = []
    for l in locs:
        if len(l) > thr:
            ioi.append(l)
    return ioi

def condition_window_corr(thr=5):
    
    locs = []
    for grp in consecutive_groups(locations):
        locs.append(list(grp))
        
    
            
    return correlations