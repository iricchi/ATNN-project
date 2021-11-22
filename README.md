# ATNN-project
Advanced Topics in Network Neuroscience project.

## Description
- Data contains 1 specific subject data (RS and Motor task):
  - M_mat_task.mat contains the experimental paradigm in a matrix format (1s when the subject was performing a speciic epoch). This is visualized in Results/Motor_exp_paradigm.jpg
  - M_vec_task.mat is the same concept but in a vector format and numbers as labels
  - sub1_Motor.mat is the data aquired from sub1 during this task (AAL parcellation so 90 brain regions)
  - sub1_RS.mat is the Resting state - always same parcellation 
  - sub1_SC.mat is the corresponding SC of that subject ( * ).

( * ) Note: The structural and diffusion-weighted MRI data of each subject were downloaded from the HCP and were processed using MRtrix3 (http://www.mrtrix.org/). We used single shell (b=3000) multi-tissue to estimate the response function, while fiber orientation distribution functions were computed using constrained spherical deconvolution of order 8. Tractogram generation was performed using deterministic tractography with about 2x10^7 output streamlines, and was seeded from the white matter. Fiber densities were used as metric to define the individual SCs and were computed by dividing the total number of fibers that connect each of the pairwise regions of the AAL atlas with both the mean fiber length and the average of the sizes of the two regions considered.

- Each of us may have a specific folder if we want to write our own scripts
- common_scripts is going to be the main folder with shared scripts :)

### Keypoints
1. Considering healthy subjects (for now only data available already preprocessed), we can focus on subject specific (eg. 1 subject)
2. We consider the SC of that subject and run a community detection (e.g. Leuvain algo)
3. Keeping in mind the communites from Structure, we can compute FC dinamically (e.g. sliding windows) and see how graph metrics (TBD) change in time in task or Resting State


