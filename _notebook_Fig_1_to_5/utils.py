import os
import sys
import json
import numpy as np
import pandas as pd
import seaborn as sns
import glob
import re
import itertools
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import matplotlib as mpl
try:
    from matplotlib.cm import get_cmap
except ImportError:
    import matplotlib as mpl
    def get_cmap(name):
        return mpl.colormaps[name]
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state
from sklearn.metrics import r2_score
from scipy.stats import ttest_rel
from scipy.spatial import procrustes
from scipy import stats

import nibabel as nib
from nilearn.image import load_img, mean_img, new_img_like, math_img, resample_img, threshold_img,  resample_to_img, crop_img
from nilearn.maskers import NiftiLabelsMasker
from nilearn.masking import apply_mask
from nilearn.connectome import  ConnectivityMeasure
from nilearn import signal
from nilearn import datasets
from nilearn import plotting

from brainspace.utils.parcellation import reduce_by_labels
from brainspace.gradient import gradient, kernels, embedding
from brainspace.datasets import load_fsa5, load_conte69, load_parcellation
from brainspace.plotting import plot_hemispheres
from brainspace.utils.parcellation import map_to_labels

# read json file into dict()
def read_config(config_file):
    with open(config_file) as js_file:
        params = json.load(js_file)
    return params

def get_merge_labels_sc(params):
    # return the list of SC RoIs and dictionary mapping these RoIs to constituent labels from custom atlas
    # in cases where params['merge_labels'] is None, each RoI is directly the same label from the custom, fine parcellation
    # however if params['merge_labels'] is the path to a json file specifying the merging scheme, then the RoIs are merged accordingly with a single RoI potentially containing multiple labels from the custom atlas
    # in cases where params['merge_labels'] is True, then the merging scheme is simply applied to WM RoIs to combine all of them at a given level into a single WM RoI for that level
    sc_level = params["sc_level"]
    if sc_level == "none":
        sc_level = None
    atlas_custom_path = params["custom_sc_parc"]
    node_names = open(params["custom_sc_labels"],'r')
    node_names = node_names.read().splitlines()
    sc_rois = []
    merge_labels = {}
    if os.path.exists(params["merge_labels"]) and type(params["merge_labels"]) == str:
        asc_desc = read_config(params["merge_labels"])
    else:
        asc_desc = None
    for node_name in node_names:
        if sc_level is not None:
            if not node_name.startswith(sc_level):
                continue
        else:
            if not params["include_C1C2"]:
                if node_name.startswith('C1') or node_name.startswith('C2'):
                    continue
            if not params["include_T1"]:
                if node_name.startswith('T1'):
                    continue
        if not params["merge_labels"]:
            sc_rois.append(node_name)
            continue
        if node_name.split(' ')[1] == 'WM':
            key_wm = node_name.split(' ')[0]+' WM'
            if asc_desc is not None:
                for key_asc in asc_desc.keys():
                    if ' '.join(node_name.split(' ')[3:]) in asc_desc[key_asc]:
                        key_wm = key_wm + ' ' + key_asc
            try:
                merge_labels[key_wm].append(node_name)
            except:
                merge_labels[key_wm] = [node_name]
                sc_rois.append(key_wm)
        else:
            merge_labels[node_name] = [node_name]
            sc_rois.append(node_name)
    sc_rois = np.sort(sc_rois)
    return sc_rois, merge_labels

def extract_timecourse_rest(subject_list, session_nos, params, label_parc='brain_jubrain32'):
    """ Extract timecourses from resting-state fMRI data using specified parcellation and save them as csv files.

    Args:
        subject_list (list of str): List of subject IDs for which to extract timecourses
        session_nos (list of int): List of session numbers possible. e.g. [1,2]
        params (dict): Dictionary extracted from adjoined config.json file
        label_parc (str, optional): parcellation to use for time series extraction. Options are 'brain_jubrain32', 'brain_glasser360', 'brain_Schaefer{number rois}', 'brain_HarvardOxford', 'SC'. Defaults to 'brain_jubrain32'.
        The saved timecourses files are stored in f"{params['save_timecourse_rest']}/{label_parc}/" directory.

    Raises:
        Exception: If invalid parcellation_brain is specified in label_parc.

    Returns:
        None
    """
    path_main  = params["path_main"]
    save_timecourse_dir = params["save_timecourse_rest"]
    region_prefix = label_parc.split('_')[0][:2].upper()
        
    # load atlas, label names and mask for region based on parcellation choice
    if label_parc == 'SC':
        atlas_file = params["custom_sc_atlas"]
        atlas_labels_names = open(params["custom_sc_labels"],'r')
        atlas_labels_names = atlas_labels_names.read().splitlines()
        mask_img = params["mask_sc"]
        print(f"Extracting timecourses from SC fMRI using custom parcellation")

    elif label_parc == 'brain_glasser426':
        atlas_file = params["glasser_atlas"]
        atlas_labels = params["glasser_labels"]
        df_parc = pd.read_csv(atlas_labels, sep='\t',index_col=False)
        atlas_labels_names = df_parc['Label'].iloc[1:] # remove background as label
        mask_img = params["mask_brain"]
        print(f"Extracting timecourses from brain fMRI using extended Glasser parcellation including subcortical regions")
    
    elif label_parc == 'brain_jubrain32':
        atlas_file = params["jubrain_atlas"]
        atlas_labels = params["jubrain_labels"] 
        roi_names = open(atlas_labels,'r')
        atlas_labels_names = roi_names.read().splitlines()
        mask_img = params["mask_brain"]
        print(f"Extracting timecourses from brain fMRI using Julich cytoarchitectonic parcellation")

    elif label_parc.startswith('brain_Schaefer'):
        n_rois = int(label_parc.split('Schaefer')[1])
        atlas_dataset = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois, resolution_mm=2)
        atlas_file = atlas_dataset.maps
        try:
            atlas_labels_names = atlas_dataset.labels.astype(str)
        except:
            atlas_labels_names = atlas_dataset.labels
        mask_img = params["mask_brain"]
        print(f"Extracting timecourses from brain fMRI using Schaefer2018 parcellation  ({n_rois} RoIs)")

    elif label_parc == 'brain_HarvardOxford':
        atlas_dataset = datasets.fetch_atlas_harvard_oxford(atlas_name="sub-maxprob-thr50-2mm")
        atlas_file = atlas_dataset.maps
        atlas_labels_names = atlas_dataset.labels[1:] # remove Background as label
        mask_img = params['mask_brain']
        print(f"Extracting timecourses from brain fMRI using Harvard Oxford parcellation")

    else:
        raise Exception("Invalid parcellation_brain specified in label_parc")
    
    masker = NiftiLabelsMasker(
                labels_img=atlas_file,
                mask_img=mask_img,
                standardize="zscore_sample",
                standardize_confounds=True,
                memory="nilearn_cache",
                verbose=0,
            )

    # iterate over subjects and sessions to extract time series and save as csv
    for subject_id in subject_list:
        for session_no in session_nos:
            if label_parc.startswith('brain'):
                path_fmri = f"{path_main}{subject_id}/func/rest/session{session_no}/5_Coregistration/brain/fmri_stc_brain_moco_rest_denoised_reg.nii.gz"
                motionreg = f"{path_main}{subject_id}/func/rest/session{session_no}/4_Denoising/brain/Nuisance/nuisance.txt" 
            elif label_parc.startswith('SC'):
                path_fmri = f"{path_main}{subject_id}/func/rest/session{session_no}/8_SpinalCord/mfmri_stc_sc_rest_denoised_n.nii.gz"
                motionreg = f"{path_main}{subject_id}/func/rest/session{session_no}/4_Denoising/brain/Nuisance/nuisance.txt"
            
            
            if params["add_confounds"]:
                # add motion regressors during time series extraction if params["add_confounds"] is True
                time_series = masker.fit_transform(path_fmri, confounds=motionreg)
            else:
                time_series = masker.fit_transform(path_fmri)

            # time series --> dataframe --> save as csv
            nodes_timecourses = {}
            for i,label in enumerate(atlas_labels_names):
                nodes_timecourses[label] = time_series[:,i]
            nodes_timecourses = pd.DataFrame(nodes_timecourses)
            if not os.path.exists(f'{save_timecourse_dir}{label_parc}/'):
                os.makedirs(f'{save_timecourse_dir}{label_parc}/')
            nodes_timecourses.to_csv(f'{save_timecourse_dir}{label_parc}/{subject_id}_session{session_no}_{region_prefix}_nodes.csv', sep='\t')
            #print(f"Generated timecourses for ({subject_id},session{session_no}) using "+ label_parc + " atlas")

def sec_to_vol(onset, duration, tr):
    start = int(onset / tr)
    end = int((onset + duration) / tr)
    return np.arange(start, end)

def extract_FC(subject_list, session_nos, params, roi_paths, pca=False):
    """Extract Functional Connectivity matrix from fMRI timeseries using specified parcellation and save them as csv files.

    Args:
        subject_list (list of str): List of subject IDs for which to extract FC
        session_nos (list of int): List of session numbers possible. e.g. [1,2]
        params (dict): Dictionary extracted from adjoined config.json file
        roi_paths (dict of str:str): Dictionary of RoI names mapped to their corresponding timecourse paths 
        
    Returns:
        tuple: (list containing FC matrices for each (subject,session) pair, list of RoIs in the order that they appear on the FC matrix columns/rows)
    
    """
    FC_mats = []
    corr_meas = ConnectivityMeasure(kind=params["FC_measure"], standardize=True, discard_diagonal=True)
    
    for subject_id in subject_list:
        for session_no in session_nos:
            timecourse_list = []
            rois_list = []
            for roi, path in roi_paths.items():
                # load timecourse for 'roi' from 'path' for the given subject as a numpy array
                region_prefix = 'SC' if roi=='SC' else 'BR'
                timecourse = pd.read_csv(f'{path}{subject_id}_session{session_no}_{region_prefix}_nodes.csv',sep='\t', index_col = 0)
                # for SC RoIs, if params["merge_labels"] is True or is the path to a json file specifying the merging scheme, then merge the time courses of the RoIs according to the specified merging scheme before computing FC. 
                # This is to ensure that we can compute FC at the level of merged RoIs in SC which combine multiple labels from the custom atlas
                if region_prefix == 'SC':
                    sc_rois, merge_labels = get_merge_labels_sc(params)
                    sc_level_timecourse = np.zeros((len(timecourse),len(sc_rois)))
                    for r, roi in enumerate(sc_rois):
                        if not params["merge_labels"]:
                            sc_level_timecourse[:,r] = timecourse[roi].to_numpy()
                            continue
                        merge_tc = []
                        for node_name in merge_labels[roi]:
                            merge_tc.append(timecourse[node_name].to_numpy())                    
                        sc_level_timecourse[:,r] = np.mean(np.array(merge_tc),axis=0)
                    timecourse = pd.DataFrame(sc_level_timecourse, columns=sc_rois)
                
                if pca:
                    colnames = timecourse.columns
                    timecourse = timecourse.to_numpy()
                    # temp - apply pcaa
                    pca = PCA(n_components=min(timecourse.shape[0],timecourse.shape[1]))
                    pca.fit(timecourse)
                    # choose top eigenvectors that together explain 70% of EV
                    eig_70 = np.argwhere(np.cumsum(pca.explained_variance_ratio_)>0.7)[0,0]
                    # print(f'EV ratios explaining {100*np.sum(pca.explained_variance_ratio_[:eig_70+1])}%:{pca.explained_variance_ratio_[:eig_70+1]}')
                    pca = PCA(n_components=eig_70+1)
                    br_low =pca.fit_transform(timecourse)
                    timecourse = pca.inverse_transform(br_low)
                
                    rois_list.extend(colnames.tolist())
                    timecourse = timecourse.T
                else:
                    rois_list.extend(timecourse.columns.tolist())
                    timecourse = timecourse.to_numpy().T # rows are nodes and columns are time points
                timecourse_list.append(timecourse)
            # timecourse of all rois in roi_paths
            timecourse_all = np.concatenate(timecourse_list, axis=0)
            FC_mat = corr_meas.fit_transform([timecourse_all.T])[0] # compute FC
            FC_mats.append(FC_mat)
    #print(f'Generated {len(FC_mats)} FC matrices; shape={FC_mat.shape}')
    return FC_mats, rois_list
  
def fit_gradients(FC_mat, rois_incl, rois_list, rois_incl_y = None, n_components=10, approach='dm', kernel='cosine', sparsity=0.9, alpha=0.5, random_state=42):
    """Fit gradients on sub-FC of FC_mat, specified using RoIs in rois_incl. Uses BrainSpace GradientMaps module for gradient fitting.

    Args:
        FC_mat (numpy array): FC matrix on which to fit gradients, with rows and columns in the order of rois_list
        rois_incl (list of str): List of RoIs (subset of rois_list) to include in the sub-FC. The gradients fit are projected into this space of len(rois_incl) dimensions. By default, this is the list of RoIs to include in both x and y axes of the FC matrix, unless rois_incl_y is not None.
        rois_list (list of str): List of all RoIs corresponding to rows and columns of FC_mat
        rois_incl_y (list of str, optional): List of RoIs (subset of rois_list) to include in the y-axis of the sub-FC. This is relevant when the subFC is intended to be non-square with cortex X cortex + SC connections. Defaults to None.
        n_components (int, optional): Number of gradient components to fit. Defaults to 10 
        approach (str, optional): Approach to use for gradient fitting. Options are 'dm' for diffusion maps and 'pca' for principal component analysis. Defaults to 'dm'.
        kernel (str, optional): Kernel to use for diffusion maps. Options are 'pearson', 'spearman', 'cosine', 'normalized_angle', 'gaussian'. Defaults to 'cosine'.
        sparsity (float, optional): Sparsity to use for pre-sparsification of functional connectivity. Defaults to 0.9.
        alpha(float, optional): Alpha value for DM embedding. Defaults to 0.5.
        random_state (int, optional): Random state for reproducibility. Defaults to 42.

    Returns:
        tuple: (Gradients values in array of shape (len(rois_incl),n_components), 
        raw lambdas in array of shape (n_components,),
        sub-FC matrix on which gradients were fit in array of shape (len(rois_incl), len(rois_incl))) if rois_incl_y is None, and (len(rois_incl), len(rois_incl_y))) otherwise
    """
    if rois_incl_y is None:
        rois_incl_y = rois_incl
    FC_ = FC_mat[[rois_list.index(roi) for roi in rois_incl],:][:, [rois_list.index(roi) for roi in rois_incl_y]]    
    gm = gradient.GradientMaps(n_components=n_components, approach=approach, kernel=kernel, random_state=random_state)
    gm.fit(FC_, sparsity=sparsity, alpha=alpha)
    return gm.gradients_, 100*gm.lambdas_/np.sum(gm.lambdas_), FC_

def project_4D(g, rois_incl, rois_map, params, br_parc = 'Glasser426'):
    """Project the array to 4D brain and SC images according to the order in which RoIs appear (rois_incl) and the mapping of RoIs to SM/Thalamus or Insula or SC (rois_map). The projected brain and SC images are returned as separate nifti images.

    Args:
        g (numpy array): Array of shape (len(rois_incl),n_components) or (len(rois_incl),) to be projected to 4D. The order of RoIs in g is the same as the order of RoIs in rois_incl.
        rois_incl (list of str): List of RoIs corresponding to RoI labels in the order they appear in g
        rois_map (list of str): List of region names that each RoI in rois_incl belongs to, in the same order as rois_incl. Options for region names are 'Insula' for insular cortex and 'SM/Thalamus' for cortical and subcortical regions included in the brain parcellation. For SC, use levels like 'C5'. This mapping is needed to determine whether a given RoI should be projected to the brain image or the SC image, and also to combine the Julich and Glasser/Schaefer parcellations
        params (dict): Dictionary extracted from adjoined config.json file, needed to load the relevant brain and SC masks and atlases for projection
        br_parc (str, optional): parcellation used for brain RoIs. Options are 'Glasser426' and 'Schaefer{number of rois}'. Defaults to 'Glasser426'. This is needed to determine which brain atlas to use for projection of brain RoIs.

    Returns:
        tuple: 4D images of brain and SC respectively with the projected array. The SC image is returned as None if no SC RoIs are included in rois_incl
    """
    if len(g.shape)==1:
        g = g.reshape(-1,1)


    mask_br = load_img(params["mask_brain"])
    if br_parc == 'Glasser426':
        br_img = load_img(params["glasser_atlas"])
        br_labels = pd.read_csv(params["glasser_labels"], sep='\t',index_col=False)['Label'].iloc[1:].to_list()
    elif br_parc.startswith('Schaefer'):
        n_rois = int(br_parc.split('Schaefer')[1])
        atlas_dataset = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois, resolution_mm=2)
        br_img = load_img(atlas_dataset.maps)
        try:
            br_labels = list(atlas_dataset.labels.astype(str)) # in some nilearn versions the labels are byte characters that need to be converted to str
        except:
            br_labels = list(atlas_dataset.labels)

    ins_img = load_img(params["jubrain_atlas"])
    ins_labels = open(params["jubrain_labels"], 'r').read().splitlines()

    br_data = np.zeros((*mask_br.get_fdata().shape,g.shape[1]))

    sc_levels = [f'C{i}' for i in range(1,9)] + ['T1']
    mask_sc = load_img(params["mask_sc"])
    sc_img = load_img(params["custom_sc_atlas"])
    sc_labels = open(params["custom_sc_labels"],'r').read().splitlines()
    sc_rois, merge_labels_sc = get_merge_labels_sc(params)
    
    sc_data = np.zeros((*mask_sc.get_fdata().shape,g.shape[1]))

    for j, roi_ in enumerate(rois_map):
        if roi_ in sc_levels: 
            if not params["merge_labels"]:
                sc_rois = list(sc_rois)
                sc_data[sc_img.get_fdata()==sc_labels.index(rois_incl[j])+1,:] = g[j,:]
            else:
                for node_name in merge_labels_sc[rois_incl[j]]:
                    node_label = sc_labels.index(node_name)
                    sc_data[sc_img.get_fdata()==node_label+1,:] = g[j,:]
        elif roi_ == 'Insula':
            br_data[ins_img.get_fdata()==ins_labels.index(rois_incl[j])+1,:] = g[j,:]
        elif rois_incl[j] in br_labels:
            br_data[br_img.get_fdata()==br_labels.index(rois_incl[j])+1,:] = g[j,:]
    
    if all(roi in sc_levels for roi in rois_map):
        br_img = None
    else:
        br_data = np.squeeze(br_data)
        br_img = new_img_like(br_img, br_data)
    
    if any(roi in sc_levels for roi in rois_map):
        sc_data = np.squeeze(sc_data)
        sc_img = new_img_like(sc_img, sc_data)
    else:
        sc_img = None
    return br_img, sc_img

def align_procrustes(grad ,ref_grad, align_dims=2):
    """Perform procrustes alignment of grad to reference axis ref_grad for the top align_ndims dimensions

    Args:
        grad (array): gradient axis to be aligned, of shape (n_rois_incl, n_components) where n_components is the number of gradient components that were fit. The order of RoIs in grad should be the same as the order of RoIs in ref_grad.
        ref_grad (array): reference axis to which grad will be aligned using procrustes analysis
        align_dims (int, optional): Number of dimensions of the gradient that have to be aligned. Defaults to 2.

    Returns:
        tuple: (matrix of shape (n_rois_incl, align_dims) containing the normalized reference axis values for the top align_dims dimensions,
        matrix of shape (n_rois_incl, align_dims) containing the aligned gradient axis values for the top align_dims dimensions,
        disparity value indicating the quality of the alignment, with lower values indicating better alignment)

    """
    m1, m2, disparity = procrustes(ref_grad[:,:align_dims],grad[:,:align_dims])
    return m1, m2, disparity

def procrustes_matlab_style(X, Y, scaling=False, reflection=False):
    """
    MATLAB-like Procrustes analysis for X (ref) and Y (to align).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    n, m = X.shape

    # Center
    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)
    X0 = X - muX
    Y0 = Y - muY

    # Norms (Frobenius)
    ssX = np.sum(X0**2)
    ssY = np.sum(Y0**2)

    # Scale to unit norm (internal step, like MATLAB)
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)
    X0 /= normX
    Y0 /= normY

    # Cross-covariance
    A = X0.T @ Y0

    # SVD
    U, S, Vt = np.linalg.svd(A)
    V = Vt.T

    # Reflection control
    T = V @ U.T
    if not reflection:
        if np.linalg.det(T) < 0:
            V[:, -1] *= -1
            S[-1] *= -1
            T = V @ U.T

    # Scaling factor
    if scaling:
        b = np.sum(S) * normX / normY
    else:
        b = 1.0

    # Translation
    c = muX - b * muY @ T

    # Transform Y -> Z
    Z = b * Y @ T + c

    # Disparity
    d = 1.0 - np.sum(S)**2

    tform = {'T': T, 'b': b, 'c': c}
    return d, Z, tform

def align_procrustes_matlab(grad, ref_grad, align_dims=3):
    """
    Align 'grad' to 'ref_grad' using MATLAB-style Procrustes
    (Scaling=false, Reflection=false), after z-scoring columns.
    """
    ref_mat  = np.asarray(ref_grad)[:, :align_dims]
    grad_mat = np.asarray(grad)[:, :align_dims]

    # Z-score each column (SMC and SMC-spinal) before alignment
    ref_z = (ref_mat - ref_mat.mean(axis=0)) / ref_mat.std(axis=0, ddof=0)
    grad_z = (grad_mat - grad_mat.mean(axis=0)) / grad_mat.std(axis=0, ddof=0)

    d, Z, tform = procrustes_matlab_style(
        ref_z,
        grad_z,
        scaling=False,
        reflection=False
    )

    ref_aligned  = ref_z   # z-scored reference (unchanged by Procrustes)
    grad_aligned = Z       # z-scored, aligned spinal gradients
    disparity    = d

    return ref_aligned, grad_aligned, disparity

def project_brainspace_arr(g, rois_incl=None, fill = np.nan, nrois_schaefer=400):
    """Project the input array g into the full Schaefer brain, and then map to surface labels array compatible with plotting on brainspace. The output array can be saved to drive, for e.g. as a .npy file and then loaded locally via the mounted drive for direct plotting with brainspace.plotting.surface_plotting.plot_hemispheres().
    COMPATIBLE ONLY IF SCHAEFER PARCELLATION IS USED FOR THE GRADIENT.

    Args:
        g (numpy array): Array consisting of either the full or sub set of Schaefer RoIs in the order rois_incl. This is the array to be projected on to the full brain, filling empty values as fill variable, and then mapped to surface labels for surface plotting
        rois_incl (list of str, optional): List of Schaefer RoIs included in the g, this helps to project g back on to the full Schaefer atlas space by listing the RoI names in the order they appear in g. Defaults to None, for cases where g is already the nrois_schaefer long array with all Schaefer RoIs
        fill (float, optional): Value to fill in empty spaces (for e.g. rest of brain in cases where g is a subset of the Schaefer full brain, or otherwise any nan values). Defaults to nan
        nrois_schaefer (int, optional): Number of RoIs in the Schaefer parcellation used. Defaults to 400.

    Returns:
        numpy array: Numpy array of the values in g for each surface label, ready for plotting using brainspace's surface plotting function
    """
    if len(g.shape)==1:
        g = g.reshape(-1,1)
    schaefer_atlas = datasets.fetch_atlas_schaefer_2018(n_rois=nrois_schaefer,resolution_mm=2, verbose=0)
    schaefer_labels = list(schaefer_atlas.labels.astype(str))
    g_full = np.empty((len(schaefer_labels),g.shape[1]))    
    
    if g.shape[0] == len(schaefer_labels):
        g_full = g
    else:
        g_full[:] = fill
        for j, roi in enumerate(rois_incl):
            g_full[schaefer_labels.index(roi),:] = g[j,:]
    
    labeling = load_parcellation('schaefer', scale=nrois_schaefer, join=True)
    mask=labeling!=0
    g_vec = [None]*g.shape[1]
    for i in range(g.shape[1]):
        g_vec[i] = map_to_labels(g_full[:, i], labeling, mask=mask, fill=fill)
    g_vec =np.array(g_vec)
    return g_vec


if __name__ == '__main__':
    
    if len(sys.argv) < 3:
        config_file = "config.json" #use config file in pwd
    else:
        if '--config' not in sys.argv:
            print("ERROR: specify option --config with the corresponding configuration file")
            assert '--config' in sys.argv
        else:
            ind = sys.argv.index('--config') + 1
            config_file = sys.argv[ind]
    params = read_config(config_file)