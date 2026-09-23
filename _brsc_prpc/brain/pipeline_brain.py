# -*- coding: utf-8 -*-
#!/usr/bin/python

""""
Example:
    python pipeline_brain.py --config config_brain.json --moco (will set motor correction to true)

"""
import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from glob import glob
import subprocess
from nilearn import image

class PreprocessingRS(object):
    def __init__(self, config_file):
        super(PreprocessingRS, self).__init__()
        self.config_file = config_file
        self.__init_configs()
        self.working_dir = os.getcwd()
        self.__read_configs(config_file)


    def __init_configs(self):
        """Initialze variables"""
        self.TR = 1.55
        self.fmriname = "fmri"  # this is the name of the fmri nifti to analyse
        self.slice_timing = False
        self.crop_image = False
        self.moco_brain = False
        self.segmentation = False
        self.anat2func = False
        self.smoothing = False

    def processes(self):

        if self.slice_timing:
            self.slice_timing_corr()
            os.chdir(self.working_dir)
        
        if self.crop_image:
            self.crop_img()
            os.chdir(self.working_dir)

        if self.moco_brain:
            self.mc_brain()
            os.chdir(self.working_dir)
        
        if self.segmentation:
            self.segment_anat()
            os.chdir(self.working_dir)

        if self.anat2func:
            self.coreg_anat2func()
            os.chdir(self.working_dir)
        
        if self.smoothing:
            self.smoothing_brain()
            os.chdir(self.working_dir)
        
    # Select or de-select the subjects for analysis  
    def select_subjects(self, subj_found, include=range(1, 20), exclude=[]):
        include_list = [f"S{i:02d}" for i in include if i not in exclude]
        return [s for s in include_list if s in subj_found]

    def __read_configs(self, input_file):
        """Recursively read configs from given JSON file."""
        with open(input_file) as js_file:
            params = json.load(js_file)

        for key in params:
            if hasattr(self, key) and isinstance(getattr(self, key), list):
                assert isinstance(params[key], list)
                getattr(self, key).extend(params[key])
            else:
                setattr(self, key, params[key])

        if self.condition == "rest":
            self.session = ["session1", "session2"]
        
        # assign the list of subjects variable
        self.list_subjects = glob(os.path.join(self.parent_path, self.data_root+'*'))
        subj_found = [os.path.basename(sub.rstrip('/')) for sub in self.list_subjects]
        self.list_subjects = self.select_subjects(subj_found, include=range(1, 20), exclude=[15])
        print(" ### Condition:", self.condition)
        print(" ### Session:", self.session)
        print(" ### Subject List:", self.list_subjects)

    def slice_timing_corr(self): 
        subj_paths = [os.path.join(self.parent_path, s, self.func, self.condition, ses) for s in self.list_subjects for ses in self.session]
        print("Subject paths currently working with:", subj_paths)

        start = time.time()
        print(" ### Info: slice timing correction ...") 
        for sub in subj_paths:
            self._correct_slice_time(sub, fmriname="fmri")
        print("### Info: Slice time correction done in %.3f s" %(time.time() - start ))

    def _correct_slice_time(self, sps, fmriname):
        stc_dir = sps+self.stc
        if not os.path.exists(stc_dir):
            os.mkdir(stc_dir)

        if not os.path.exists(stc_dir + f"{fmriname}_stc.nii.gz"):
            print(">>>>> slice timing correction started")
            # Define input and output names
            session_name = sps.split('/')[-1]
            dicom_path = os.path.join(
                '/'.join(sps.split('/')[:-3]),  # up to subject root
                self.dicoms,
                self.condition,
                session_name,
            )
            json_f = glob(os.path.join(dicom_path, '*.json'))[0]
            i_img = sps + f"/{fmriname}.nii.gz"
            o_img = stc_dir + os.path.basename(i_img.split(".")[0] + "_stc.nii.gz")
            with open(json_f) as g:
                params = json.load(g)
            tr=params["RepetitionTime"] # extract the time repetition value
            o_txt = os.path.join(os.path.dirname(json_f), os.path.basename(i_img).split("_")[0] + "_stc.txt") # output with slicetiming info
            stc_info=params["SliceTiming"] # provide info about interleave slice order

            with open(o_txt, 'w') as f:
                for item in stc_info:
                    item_tr=item/tr # Transform slicetiming in secs in st in TRs units
                    f.write('{}\n'.format(item_tr))
            f.close()
            
            # run slice timing correction:
            string="slicetimer -i "+ i_img + " -o " + o_img +" -r " + str(tr) + " --tcustom=" + o_txt
            print(string)
            os.system(string)
            print("Slice timing correction done")

            # Store original SliceTimings values from BIDS json sidecar for PNM denoising (performed later)
            pnm_txt = os.path.join(
                os.path.dirname(json_f), "pnm_slicetiming_seconds.txt"
            )
            with open(pnm_txt, "w") as f:
                f.write(" ".join(f"{t:.8f}" for t in stc_info) + "\n")

        elif os.path.exists(stc_dir + f"{fmriname}_stc.nii.gz"):
            print(">>>>> slice timing correction was already completed")
        
        if not os.path.exists(stc_dir + f"{fmriname}_mean.nii.gz"):
            print(">>>>> Computing the mean image")
            i_img = sps + f"/{fmriname}.nii.gz"
            o_img = stc_dir + os.path.basename(i_img.split(".")[0] + "_mean.nii.gz")
            string_bsc_mean = 'fslmaths '+ i_img +' -Tmean '+ o_img
            os.system(string_bsc_mean) 
    
    def crop_img(self):
        subj_paths_func = [os.path.join(self.parent_path, s, self.func, self.condition, ses) for s in self.list_subjects for ses in self.session]
        paths_anat = [os.path.join(self.parent_path, s, self.anat, 'session1') for s in self.list_subjects]
        print("Subject paths currently working with:", subj_paths_func)
        start = time.time()

        for sub in subj_paths_func:
            self._crop_func(sub, fmriname="fmri_stc")
        
        for sub in paths_anat:
            self._crop_anat(sub, mriname="t1")

        print("### Info: Cropping Done done in %.3f s" %(time.time() - start ))

    def _crop_func(self, sps, fmriname):
        ## func image
        i_img = sps + self.stc + f'{fmriname}.nii.gz'
        print(i_img)
        o_folder = sps + self.stc
        o_img_brain=o_folder + os.path.basename(fmriname.split(".")[0] + "_brain.nii.gz")
        o_img_sc=o_folder + os.path.basename(fmriname.split(".")[0] + "_sc.nii.gz")
        
        # For brain 
        if not os.path.exists(o_img_brain):
            string_br='sct_crop_image -i ' + i_img + ' -o '+ o_img_brain+' -zmin 35 -zmax -1'
            string_br_mean = 'fslmaths '+ o_img_brain +' -Tmean '+o_img_brain.split('.')[0] + '_mean.nii.gz'
            os.system(string_br) 
            os.system(string_br_mean)
        # For spinalcord
        if not os.path.exists(o_img_sc):   
            string_sc='sct_crop_image -i ' + i_img + ' -o '+ o_img_sc+' -zmin 0 -zmax 34'
            string_sc_mean = 'fslmaths '+ o_img_sc +' -Tmean '+o_img_sc.split('.')[0] + '_mean.nii.gz'
            os.system(string_sc) 
            os.system(string_sc_mean)
    
    def _crop_anat(self, sps, mriname):
        ## anat image
        i_img = os.path.join(sps, mriname, f'{mriname}.nii.gz')
        o_folder = os.path.join(sps, mriname)
        o_img_brain = o_folder + "/"  + os.path.basename(mriname.split(".")[0] + "_brain.nii.gz")
        o_img_sc = o_folder + "/" + os.path.basename(mriname.split(".")[0] + "_sc.nii.gz")
        if not os.path.exists(o_img_sc):
            # For brain
            string_br='sct_crop_image -i ' + i_img + ' -o '+ o_img_brain+' -zmin 160 -zmax -1'
            os.system(string_br) 
            # For spinalcord
            string_sc='sct_crop_image -i ' + i_img + ' -o '+ o_img_sc+' -zmin 0 -zmax 159'
            os.system(string_sc)
    
    def mc_brain(self):
        start = time.time()
        for ses in self.session:
            print("### Running motion correction for", ses)
            subj_paths = [os.path.join(self.parent_path, s, self.func, self.condition, ses) for s in self.list_subjects]
            print("Subject paths currently working with:", subj_paths)
            for sub in subj_paths:
                self._moco_br(sub, fmriname="fmri_stc_brain")
        print("### Info: Motion correction done in %.3f s" % (time.time() - start))
    
    def _moco_br(self, sps, fmriname, plot_show=True):
        o_folder = sps + self.moco_func
        if not os.path.exists(o_folder):
            os.mkdir(o_folder)
        if not os.path.exists(o_folder+ "brain/"):
            os.mkdir(o_folder + "brain/")
        
        i_img = sps + self.stc + fmriname + ".nii.gz"
        print(i_img)
        
        # Segmentation 
        mask_folder = sps + self.seg_func
        if not os.path.exists(mask_folder):
            os.mkdir(mask_folder)
        
        mask_img = mask_folder + os.path.basename(i_img).split(".")[0] + "_masked.nii.gz"
        moco_file = o_folder + "brain/" + os.path.basename(i_img).split(".")[0] + "_moco.nii.gz"
        moco_mean_file= o_folder+ "brain/" + os.path.basename(i_img).split(".")[0] + "_moco_mean.nii.gz"

        if not os.path.exists(moco_file):
            print(">>>>> Moco is running for the brain image of the sub")
    
            # 1. Create a binary mask before apply moco
            string_mask="bet "+i_img+ " "+mask_img+" -F" 
            os.system(string_mask)
            
            # 2. compute motion correction
            string_moco="mcflirt -in "+ mask_img+" -out "+moco_file+" -mats -plots"
            os.system(string_moco)
            
            #3. Calculate the mean image
            string_mean="fslmaths " +moco_file+ " -Tmean " + moco_mean_file
            os.system(string_mean)

            if plot_show==True:
                #4. Calculate Framewise displacement
                output_fd=o_folder + "brain/" + os.path.basename(i_img).split('.')[0] + "_FD_brain"
                if not os.path.exists(output_fd + ".txt"):
                    print('Compute framewise displacement')
                    string_fd="fsl_motion_outliers -i "+mask_img+" -o "+o_folder+"/brain/"+" -s "+ output_fd + ".txt -p "+ output_fd + ".png --fd"
                    print(string_fd)
                    os.system(string_fd)

                FD=pd.read_csv(output_fd + '.txt', delimiter=',',header=None)
                FD_mean = np.mean(FD[0])
                print('Mean FD for sub-' ": " + str(round(FD_mean,3)) + ' mm')
                
                #5. Plot motion parameters
                fig, axs = plt.subplots(2,1, figsize=(18, 6), facecolor='w', edgecolor='k')
                fig.tight_layout()
                fig.subplots_adjust(hspace = .5, wspace=.001)
                moco_params=pd.read_csv(moco_file + '.par',delimiter='  ',header=None,engine='python')
        
                axs[0].plot(moco_params.iloc[:,0:3]) 
                #axs[0].set_title(ID + " " + ses_name)
                axs[1].plot(moco_params.iloc[:,3:6])
                
                axs[0].set_ylabel("Rotation (rad)")
                axs[1].set_ylabel("Translation (mm)")
                axs[1].set_xlabel("Volumes")
                
                if not os.path.exists(o_folder + "/brain/moco_params.png"):
                    plt.savefig(o_folder + "/brain/moco_params.png")
                    np.savetxt(o_folder + '/brain/FD_mean.txt', [FD_mean])
    
    def segment_anat(self):
        subj_paths = [os.path.join(self.parent_path, s) for s in self.list_subjects]
        print("Subject paths currently working with:", subj_paths)
        start = time.time()
        for sub in subj_paths:
            self._seg_br(sub, mriname="t1_brain", img_type="anat")
        print("### Info: Segmentation done in %.3f s" % (time.time() - start))

    def _seg_br(self, sps, mriname, img_type):
        # Create output folder
        if img_type=="anat":
            o_folder = os.path.join(sps, self.anat, "session1", "t1/")
            i_img = os.path.join(sps, self.anat, "session1", "t1", mriname+ ".nii.gz")
        
        ## run segmentation using fsl 
        string_anat_seg="fsl_anat -i " + i_img + " -o " + o_folder + mriname
        print(string_anat_seg)
        os.system(string_anat_seg)

    ## [run the denoising scripts]

    def coreg_anat2func(self):
        subj_paths = [os.path.join(self.parent_path, s) for s in self.list_subjects]
        print("Subject paths currently working with:", subj_paths)
        start = time.time()
        for sub in subj_paths:
            for ses in self.session:  # loop over session1..session4
                self._coreganat2func(sub, ses, fmriname="fmri_stc_brain_moco", anatname="T1_biascorr_brain")
        print("### Info: Registration done in %.3f s" % (time.time() - start))

    def _coreganat2func(self, sps, ses, fmriname, anatname):
        func_dir = os.path.join(sps, self.func, self.condition, ses)
        coreg_dir = func_dir + self.func_coreg
        if not os.path.exists(coreg_dir):
            os.mkdir(coreg_dir)
        o_folder = coreg_dir + "brain/"
        if not os.path.exists(o_folder):
            os.mkdir(o_folder)

        func_img = func_dir + self.moco_func + "brain/" + fmriname + "_mean.nii.gz"
        anat_img = os.path.join(
            sps, self.anat, "session1", self.t1, "t1_brain.anat", anatname + ".nii.gz"
        )
        o_filename = o_folder + os.path.basename(fmriname).split(".")[0] + "_mean_reg.nii.gz"
        func2template = o_folder + "funct2template.mat"
        wm_file = os.path.join(
            sps, self.anat, "session1", self.t1, "t1_brain.anat", "T1_fast_pve_2.nii.gz"
        )
        warp_field = o_filename.split(".")[0] + ".mat"

        # Generate fields func↔anat
        if not os.path.exists(o_filename):
            string_flirt = (
                "flirt -in " + func_img +
                " -ref " + anat_img +
                " -o " + o_filename +
                " -omat " + warp_field +
                " -dof 6 -cost bbr -wmseg " + wm_file
            )
            print(string_flirt)
            os.system(string_flirt)
            inv_warp = warp_field.split(".mat")[0] + "_inv.mat"
            string_inv_warp = "convert_xfm -omat " + inv_warp + " -inverse " + warp_field
            os.system(string_inv_warp)

        # Concatenate to get func↔template
        if not os.path.exists(func2template):
            func2standard = o_folder + fmriname + "_mean_reg.mat"
            standard2template = os.path.join(
                sps, self.anat, "session1", self.t1, "t1_brain.anat", "T1_to_MNI_lin.mat"
            )
            string_convertxfm = (
                "convert_xfm -omat " + func2template +
                " -concat " + standard2template + " " + func2standard
            )
            os.system(string_convertxfm)

            inv_warp_2 = func2template.split(".mat")[0] + "_inv.mat"
            string_convertxfm_inv = (
                "convert_xfm -omat " + inv_warp_2 + " -inverse " + func2template
            )
            os.system(string_convertxfm_inv)

        standard_img = os.path.join(self.FSL_PATH, "data/standard/MNI152_T1_2mm")

        if self.condition == "rest":
            func_rest_img_full_reg = coreg_dir + "brain/" + fmriname + "_rest_denoised_reg.nii.gz"
            if not os.path.exists(func_rest_img_full_reg):
                func_img_full = func_dir + self.denoise_func + "brain/" + fmriname + "_rest_denoised.nii.gz"
                func2template_string = (
                    "flirt -ref " + standard_img +
                    " -in " + func_img_full +
                    " -applyxfm -init " + func2template +
                    " -out " + func_rest_img_full_reg
                )
                print(func2template_string)
                os.system(func2template_string)

    def smoothing_brain(self):
        print(" ### Info: Smoothing started ...")
        start = time.time()
        for s in self.list_subjects:
            for ses in self.session:
                sub = os.path.join(self.parent_path, s, self.func, self.condition, ses)
                print("Smoothing for:", sub)
                self._smooth_img(sub, fmriname="fmri_stc_brain_moco", fwhm=5, structure="brain")
        print("### Info: Smoothing done in %.3f s" % (time.time() - start))
            
    def _smooth_img(self, sps, fmriname, fwhm, structure):
            smoothing_dir = sps + self.func_smooth
            if not os.path.exists(smoothing_dir):
                os.mkdir(smoothing_dir)
            if not os.path.exists(smoothing_dir + structure):
                os.mkdir(smoothing_dir + structure)
            
            if self.condition == 'rest':
                func_img = f'{sps}/{self.func_coreg}/{structure}/{fmriname}_rest_denoised_reg.nii.gz'
                o_img = smoothing_dir + structure + '/' +os.path.basename(func_img).split('.')[0] + "_sm.nii.gz"
                if not os.path.exists(o_img):
                    smoothed_image=image.smooth_img(func_img, fwhm)
                    smoothed_image.to_filename(o_img)
                    string='fslmaths '+o_img+' -Tmean '+o_img.split('.')[0] + '_mean.nii.gz'
                    os.system(string)
                    # save smoothing parameters                    
                    with open(o_img.split('.')[0] + '.json', 'w') as f:
                        json.dump(fwhm, f) # save info
                    print("Smoothing done: " + os.path.basename(o_img))   
                return o_img
        
if __name__ == '__main__':
    
    if len(sys.argv) < 3:
        print("**************************")
        print("ERROR: no option has been specified. Please specify what process to run. If this\
               was done, specify the username in the following way --userid username")
        print("**************************")
        assert len(sys.argv) >= 3
    else:   
        if '--config' not in sys.argv:
            print("ERROR: specify option --config with the corresponding configuration file")
            assert '--config' in sys.argv
        else:
            ind = sys.argv.index('--config') + 1
            config_file = sys.argv[ind]
            print(" ### Info: reading config file: %s" % config_file)
    
    print(" ### Info: running Preprocessing ... ")  

    PR = PreprocessingRS(config_file)

    if '--slice_timing' in sys.argv:
        PR.slice_timing = True
    if '--crop_image' in sys.argv:
        PR.crop_image = True
    if '--moco_brain' in sys.argv:
        PR.moco_brain = True
    if '--segmentation' in sys.argv:
        PR.segmentation = True
    if '--anat2func' in sys.argv:
        PR.anat2func = True
    if '--smoothing' in sys.argv:
        PR.smoothing = True

    PR.processes()





