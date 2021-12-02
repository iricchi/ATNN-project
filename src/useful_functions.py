import os
import numpy as np
from scipy.io import loadmat
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

def load_ground_truth(task, path_parent = './Data'):
    
    exp_par = loadmat(os.path.join(path_parent, '%s_vec_task.mat' % task))['%s_vec_task' % task]
    exp_par_names = loadmat('Data/%s_task_names.mat' % task)['task_names']

    # transfor variable so that they're usable
    exp_par = exp_par[0]+1 
    names_task = [name[0] for name in exp_par_names[0]]

    # define dictionary for simplicity
    task_dic = dict(zip(np.unique(exp_par), names_task))
    
    return exp_par, task_dic 

def extract_limits(exp_par, cur_label, task_dic, thr=5):
    """
    Given the experimental paradigm vector (exp_par) with its labels and 
    the current label under consideration, it extracts the limits of the 
    time windows.
    
    Inputs: 
    - exp_par (vector of the experimental paradigm)
    - cur_label (current label of task paradigm)
    - task_dic (dictionary of task paradigm labels)
    - thr (threshold for discarding time samples that do not have continuity - RS)
    
    Outputs:
    - dictionary with keys the tuple of start and end limits of the window and task 
    name as value
    
    """
    indexes = np.where(exp_par==cur_label,exp_par,False)
    locations = np.argwhere(indexes).T[0]

    locs = []
    for grp in consecutive_groups(locations):
        locs.append(list(grp))

    w_inds = []
    for l in locs:
        if len(l) > thr:
            w_inds.append((l[0],l[-1]))
            
    return dict(zip(w_inds, len(w_inds)*[task_dic[cur_label]]))

def condition_window_corr(x, task='M'):
    """
    Given x with functional activity and what type of task (to load data) 
    outputs windows and labels.
    """
    
    exp_par_vec, task_dic = load_ground_truth(task)
    
    # initialize dictionary
    dic_w = dict()
    
    for label in np.unique(exp_par_vec):
        dic_w.update(extract_limits(exp_par_vec, label, task_dic))
        
    # sort dictionary 
    dic_w = sorted(dic_w.items(), key=lambda t: t[0])
    
    n_windows = len(dic_w)
    correlations = np.zeros((n_windows,90, 90))
    labels = []
    
    for i in range(0, n_windows):
        # We compute correlation of current sample tampered with window function
        correlations[i]=pearsonr(x[dic_w[i][0][0]:dic_w[i][0][1]].T)
        labels.append(dic_w[i][1])

    return correlations, labels

