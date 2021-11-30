import nilearn.plotting as plotting



def plot_markers_based_on_partition(coords, partition, cmap, output_name='community_example.html', outpath='Results/'):
    """
    Given markers (defined by their coordinates) as well as a color map function and a partition vector, plot in 
    interactive 3D plot markers in MNI space, overlaid on a glass brain.
    The visualization is saved in the Results subdirectory as an HTML file with the provided name.
    :param coords: 3D coordinates of each marker in MNI space (should be N x 3, where N the number of markers)
    :param partition: Nx1 vector of assignments, denoting for each marker its community
    :param cmap: Colormap function
    :param output_name: Name under which to save visualization. (Default: community_example.html)
    :return: 
    """
    # Plot first 90 region coordinates. Each node is colored according to its community value
    view = plotting.view_markers(coords, cmap(partition)) 
    view.save_as_html(outpath + output_name)
    view.open_in_browser()
    return view