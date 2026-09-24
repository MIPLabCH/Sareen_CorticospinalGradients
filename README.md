# Functional Gradients of the Neuraxis: Reorganization from Cortex to Spinal Cord

## Corticospinal Gradients – Code Overview

This repository contains the code, configuration files, and preprocessed inputs required to reproduce the analysis and results from the article "Functional Gradients of the Neuraxis: Reorganization from Cortex to Spinal Cord".

## 📑 Related paper:
Ekansh Sareen, Nivedya Nambiar, Boris Bernhardt, and Dimitri Van De Ville; "Functional Gradients of the Neuraxis: Reorganization from Cortex to Spinal Cord", 22 July 2026, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-10155262/v1]

## 💾 Data

To support reproducibility of the analyses and figures reported in this article, this repository includes preprocessed subject-level functional connectivity (FC) matrices used by the analysis code:
`_fcmatrices/` : Subject-level FC matrices for the full corticospinal ROI set; `_fcmatrices_33sp/` : Subject-level FC matrices for the somatomotor cortical parcels and 33 spinal cord ROIs; `_fcmatrices_8sp/` : Subject-level FC matrices for the somatomotor cortical parcels and 8 spinal cord ROIs. 

Each CSV file contains one participant-level ROI-by-ROI FC matrix. ROI labels are stored in both the row index and column headers, and matrices within a directory share an identical ROI ordering. The analysis notebooks load these matrices directly, compute group-average FC matrices, derive functional gradients, and generate the corresponding figures.

The raw MRI data and the full set of participant-level preprocessing derivatives are part of an ongoing data collection and are not currently publicly available. Once data acquisition, anonymisation, and curation are complete, the full anonymised dataset and relevant preprocessing derivatives will be deposited on the Open Science Framework (OSF) under an open licence.

The shared FC matrices are intended to enable reproduction of the analyses reported in this article. They are not intended for participant-level inference, re-identification attempts, or reuse beyond the scope permitted by the repository licence. For additional materials or questions regarding data access, please contact the corresponding author.

## Required toolboxes
- Spinal Cord Toolbox (SCT, version 7.1.0; De Leener et al., 2017)
- Oxford Center for fMRI of the Software Library (FSL, version 6.0)
- Nilearn toolbox (version 0.9.1)
- plotnine 0.15.8

## Repository structure

### `brsc_prpc/`
Preprocessing utilities for brain and spinal cord fMRI.

- `spine/`
  - `pipeline_spine.py`: spinal cord preprocessing (slice_timing, cropping image, motion correction, segmentation, registration, smoothing).
  - `config_spine.json`: configuration for spine preprocessing options.
- `brain/`
  - `pipeline_brain.py`: cortical preprocessing (slice timing correction, cropping image, motion correction, segmentation, registration, smoothing)
  - `config_brain.json`: configuration for cortical preprocessing.
 - `denoise/`
  - `pipeline_denoising.py`: fMRI denoising pipeline for brain and spine -- nuisance regression with motion regressors and framewise displacement, physiological denoising (heart rate, respiration, csf signal).
  - `config_denoise.json`: configuration for brain-specific denoising.
  - `config_denoise_spine.json`: configuration for spine‑specific denoising. 
- `templates/`
  - `rest_template.sh`: shell template for resting‑state preprocessing jobs (e.g., FEAT runs on a cluster).
- `fsftemplates/`
  - `template_rest.fsf`: FSL FEAT template for resting‑state preprocessing.
  - `template_noiseregression.fsf`: FSL FEAT template for noise regression.
- `atlas/`
  - `MNI152_T1_2mm_brain_mask.nii`: standard 2 mm MNI brain mask.
  - `PAM50_cord.nii.gz`: PAM50 spinal cord template. (De Leener, B., 2018)
  - `PAM50_t2.nii.gz`: PAM50 T2‑weighted spinal cord template.  (De Leener, B., 2018)


### `distance/`
Somatomotor trajectory tables used to define distances along the cortical (SMC) strip.

- `somatomotor_L_trajectory_table_al_start18_noTL.csv`: left‑hemisphere SMC trajectory (ROI labels + cumulative distance).
- `somatomotor_R_trajectory_table_al_start20_noTL.csv`: right‑hemisphere SMC trajectory (ROI labels + cumulative distance).

### `templates/`
Atlas and mask templates used downstream in the analysis.

- `atlas_custom/`
  - `atlas.nii.gz`: custom spinal cord atlas for FC/gradient analyses.
  - `info_label.txt`: label definitions for the custom atlas.
  - `merge_ascending_descending.json`: merged labels for ascending vs descending white matter tracts.
- `MNI152_T1_2mm_brain_mask.nii`: standard MNI brain mask (copy used by analysis scripts).
- `PAM50_cord.nii.gz`: PAM50 spinal cord mask (copy used by analysis scripts).
- `PAM50_t2.nii.gz`: PAM50 T2 template (copy used by analysis scripts).

### `functionalclass/`
Mapping Schaefer somatomotor parcels to Brainnetome‑derived functional classes.

- `reqs/`
  - `BNA_subregions.xlsx`: Brainnetome subregion metadata.
  - `BNA-maxprob-thr0-2mm.nii.gz`: Brainnetome atlas (max‑probability, 2 mm).
  - `Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz`: Schaefer 400‑parcel, 7‑network atlas in MNI152 2 mm.
  - `Schaefer2018_400Parcels_7Networks_order.txt`: Schaefer label list.
  - `Schaefer_SMC_parcellation.csv`: somatomotor strip parcellation and functional class mapping.
- `schaeffer_to_brainnetome.ipynb`: notebook to derive Schaefer–Brainnetome correspondences and somatotopic class assignments.
- `out/`: output directory for functional class mapping products (e.g., CSVs, intermediate tables).

### `output/`
Workspace for intermediate and final analysis outputs (e.g., FC matrices, gradients, clustering results). Figure notebooks write their results here.

### `notebook_Fig_1_to_5/`
Figure‑specific notebooks and shared utilities.

- `Figure_1_clean.ipynb`: reproduces all subpanels of Figure 1 (cortical FC, cortical gradients, explained variance, clustering, enrichment, gradient distribution boxplots).
- `Figure_2_clean.ipynb`: reproduces Figure 2 panels (corticospinal FC, corticospinal gradients on cortex, explained variance, gradient scatter plots, functional clustering on gradient space, reorganization comparison between cortical and corticospinal gradient on cortex with dispersion metric for each functional class).
- `Figure_3_clean.ipynb`: reproduces Figure 3 panels (CCA between gradients and SMC trajectory distance, null distributions, regression diagnostics).
- `Figure_41_clean.ipynb`, `Figure_42_clean.ipynb`: notebooks for Figure 4 panels for intrinsic spinal FC of 2 paracellation: 8-ROI and 33-roi (spinal FC, explained variance, spinal gradients (single-subject and group-level), gradient scatter plots, quatified intrinsic organization with Fisher's J and GM comapctenss scores).
- `Figure_5_clean.ipynb`: reproduces Figure 5 panels (corticospinal FC, explained variance, corticospinal gradients on spine, gradient scatter plots, reorganization comparison between spinal and corticospinal gradient on spine with dispersion metric).
- `utils.py`: shared helper functions (config, FC computation, gradient fitting, clustering, visualization utilities).
- `config-results.json`: central configuration for figure notebooks (paths, plotting options, analysis parameters).

### `FC`
- `_fcmatrices/`: Subject-level FC matrices for the full corticospinal ROI set.
- `_fcmatrices_33sp/`: Subject-level FC matrices for the somatomotor cortical parcels and 33 spinal cord ROIs.
- `_fcmatrices_8sp/`: Subject-level FC matrices for the somatomotor cortical parcels and 8 spinal cord ROIs.

## Usage overview

1. **Resting-state corticospinal fMRI preprocessing (optional if raw data available)**  
   Use the scripts in `brsc_prpc/brain/`, `brsc_prpc/spine/`, and `brsc_prpc/denoise/` together with the FSL templates and atlas files to preprocess raw fMRI data (if available) and extract parcel‑wise timecourses needed for FC computation. The FC matrices computed from the preprocessed and paracellated fMRI data are stored in `_fcmatrices/`,  `_fcmatrices_33sp/`, and `_fcmatrices_8sp/`.

2. **Functional class mapping**  
   Use `functionalclass/schaeffer_to_brainnetome.ipynb` and the resources in `functionalclass/reqs/` to derive somatotopic classes for the Schaefer somatomotor strip.

3. **Gradients, FC, and figures**  
   Run the notebooks in `notebook_Fig_1_to_5/` to compute FC matrices, gradients, clustering, CCA, and statistical tests, and to regenerate Figures 1–5 from the manuscript.

4. **Precomputed inputs**  
   The `_fcmatrices/`, `_fcmatrices_33sp/`, `_fcmatrices_8sp/`, `distance/`, and `templates/` folders provide all required inputs so that the figure notebooks can be executed without rerunning the full preprocessing pipeline.
