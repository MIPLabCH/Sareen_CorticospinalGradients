# -*- coding: utf-8 -*-
#!/usr/bin/python

"""
Usage:
python pipeline_spine.py --config config_spine.json
python pipeline_spine.py --config config_spine.json --anat_norm --moco --func_norm --normalization --smoothing


Order of the options (recommended to follow):
STEPS :
    0) Segmentation with SCT and labels generation (outside the pipeline, example for t1 image, change accordingly for t2 image)
		Run the following commands
        Generating file: labels.nii.gz
        sct_label_utils -i t2.nii.gz -create-viewer 1,2,3,4,5,6,7,8,9,10 -o labels.nii.gz
        Review the documentation from SCT to verify correct labelling
    
    Pre-processing Pipeline:
    1) anat_norm (register to template, anatomical normalization)
    2) moco (motion correction)
        This part is ran twice:
        [RUN-1] python pipeline_spine.py --config config_spine.json --moco
            Correct the generated mask_sc_raw.nii.gz and save it as mask_sc.nii.gz
        [RUN-2] python pipeline_spine.py --config config_spine.json --moco
            Generates the Segmentation folder with required mask_sc.nii.gz and mask_csf.nii.gz

    3) func_norm (functional normalization)
    4) normalization (apply normalization to template space)
    5) smoothing

WARNING1: if you don't specify the options --anat_norm, --moco, the flags used will be the ones
specified in the config file.
The options (eg --moco) are used to set some specific processes to true, otherwise you can
declare them on the config file.


"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
#import plotly.graph_objects as go
import time
from glob import glob
from scipy.io import loadmat
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
        self.fmriname = "fmri_stc_sc"
        self.crop_sc = False
        self.anat_norm = False
        self.moco = False
        self.func_norm = False
        self.mask_fname = "mask_sc"
        self.mask_csf_name = "mask_csf"
        self.csf_mask = False
        self.normalization = False
        self.smoothing = False

    def processes(self):
        if self.crop_sc:
            self.cord_segment()
            os.chdir(self.working_dir)

        if self.anat_norm:
            self.anat_seg_norm()
            os.chdir(self.working_dir)

        if self.moco:
            self.motor_correction()
            os.chdir(self.working_dir)

        if self.func_norm:
            self.func_normalize()
            os.chdir(self.working_dir)

        if self.normalization:
            self.normalize()

        if self.smoothing:
            self.apply_smoothing()

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

    def cord_segment(self):
        subj_paths = [os.path.join(self.parent_path, s, self.anat, "session1") for s in self.list_subjects]
        print("Subject paths currently working with:", subj_paths)
        start = time.time()
        for sub in subj_paths:
            self._cropsegment(sub, t1_name ="t1_sc", t2_name="t2")
        print("### Info: Spinal Cord Anatomical Segmentation took %.3f s" %(time.time()-start))
    
    def _cropsegment(self, sps, t1_name, t2_name):
         ## [With T1 image]
        if not os.path.exists(os.path.join(sps, self.t1 ,f'{t1_name}')):
            run_string_t1  = f'cd {sps}/{self.t1}; sct_deepseg spinalcord -i {t1_name}.nii.gz; mkdir {t1_name}; \
                cp {t1_name}.nii.gz {t1_name}; mv {t1_name}_seg.nii.gz {t1_name}_seg.json {t1_name}'
            print(run_string_t1)
            os.system(run_string_t1)
            print("### T1 Segmentation done!")
        if os.path.exists(os.path.join(sps, self.t1, f'{t1_name}')):
            print("### T1 Segmentation already performed")  

        ## [With T2 image]
        if not os.path.exists(os.path.join(sps, self.t2 ,f'{t2_name}_seg.nii.gz')):
            run_string_t2  = f'cd {sps}/{self.t2}; sct_deepseg spinalcord -i {t2_name}.nii.gz;'
            print(run_string_t2)
            os.system(run_string_t2)
            print("### T2 Segmentation done!")
        if os.path.exists(os.path.join(sps, self.t2, f'{t2_name}_seg.nii.gz')):
            print("### T2 Segmentation already performed")  
        
    def anat_seg_norm(self):
        subj_paths = [os.path.join(self.parent_path, s, self.anat, "session1") for s in self.list_subjects]
        start = time.time()
        for sub in subj_paths:
            self._register_to_template(sub, t1_name ="t1_sc", t2_name="t2")
        print("### Info: Normalization took %.3f s" %(time.time()-start))

    def _register_to_template(self, sps, t1_name, t2_name):
        # [With T1 image -- off as mostly for spine we register with T2]
        if not os.path.exists(os.path.join(sps, self.t1, f'{t1_name}/vertebral_labels/')):
            run_string_t1  = f'cd {sps}/{self.t1}/{t1_name}; sct_register_to_template -i {t1_name}.nii.gz -s {t1_name}_seg.nii.gz -ldisc labels.nii.gz -c t1 -ofolder vertebral_labels -qc qc\
                        -param step=1,type=seg,algo=centermassrot:step=2,type=im,algo=syn,iter=5,slicewise=1,metric=CC,smooth=0'
            print(run_string_t1)
            os.system(run_string_t1)
            print("### T1 anatomical normalization done!")
        if os.path.exists(os.path.join(sps, self.t1, f'{t1_name}/vertebral_labels/')):
            print("### T1 Normalization already performed")

        ## [With T2 image]
        if  os.path.exists(os.path.join(sps, self.t2 ,"vertebral_labels/")):
            run_string_t2  = f'cd {sps}/{self.t2}; sct_register_to_template -i {t2_name}.nii.gz -s {t2_name}_seg.nii.gz -ldisc labels.nii.gz -c t2 -ofolder vertebral_labels -qc qc\
                        -param step=1,type=seg,algo=centermassrot:step=2,type=im,algo=syn,iter=5,slicewise=1,metric=CC,smooth=0'
            print(run_string_t2)
            os.system(run_string_t2)
            print("### T2 anatomical normalization done!")
        if os.path.exists(os.path.join(sps, self.t2, "vertebral_labels/")):
            print("### T2 Normalization already performed")  

    def motor_correction(self):
        subj_paths = [os.path.join(self.parent_path, s, self.func, self.condition, ses) for s in self.list_subjects for ses in self.session]
        print("Subject paths currently working with:", subj_paths)
        start = time.time()
        print(" ### Info: Checking for Mask ...")        
        for sub in subj_paths:
            self._create_mask(sub,fmriname="fmri_stc_sc")
        print("### Info: Mask created in %.3f s" %(time.time() - start ))
        start = time.time()
        print(" ### Info: Starting moco and functional mask generation ...")        
        for sub in subj_paths:
            self._moco(sub, fmriname="fmri_stc_sc")
        print("### Info: Motor correction and functional mask generation done in %.3f s" %(time.time() - start ))
        
    def _create_mask(self, sps, fmriname):
        spine_directory = os.path.join(sps, self.spine_dir)
        if not os.path.exists(spine_directory):
            os.mkdir(spine_directory)
        i_img = sps + self.stc + f'{fmriname}.nii.gz'
        if not os.path.exists(os.path.join(sps, self.spine_dir, 'Mask', f'mask_{fmriname}.nii.gz')):  
            os.makedirs(os.path.join(sps, self.spine_dir, 'Mask'))
            run_string = f'cd {sps}/{self.spine_dir}; fslmaths {i_img} -Tmean {fmriname}_mean.nii.gz; mv {fmriname}_mean.nii.gz Mask; cd Mask; sct_get_centerline -i {fmriname}_mean.nii.gz -c t2;\
            sct_create_mask -i {fmriname}_mean.nii.gz -p centerline,{fmriname}_mean_centerline.nii.gz -size 30mm -o mask_{fmriname}.nii.gz;'
            print(run_string)
            os.system(run_string)
        else:
            print("### Info: Mask is already existing!")

    def _moco(self, sps, fmriname):
        i_img = sps + self.stc + f'{fmriname}.nii.gz'
        if not os.path.exists(os.path.join(sps, self.spine_dir, 'Moco')):
            run_string = f'cd {sps}/{self.spine_dir}; sct_fmri_moco -i {i_img} -m Mask/mask_{fmriname}.nii.gz -x spline -param poly=0,smooth=1,metric=MeanSquares,gradStep=1,sampling=0.2 -g 1 -r 1 -ofolder Moco; cp Moco/{fmriname}_moco.nii.gz m{fmriname}.nii.gz; cp Moco/{fmriname}_moco_mean.nii.gz m{fmriname}_mean.nii.gz'
            print(run_string)
            os.system(run_string)
        mask_sc_dir = os.path.join(sps, self.spine_dir, 'sct_deepseg')
        if not os.path.exists(mask_sc_dir):
            os.mkdir(mask_sc_dir)
        if not os.path.exists(os.path.join(sps, self.spine_dir, 'sct_deepseg/mask_sc_raw.nii.gz')):
            print(" Moco run 1: Generating mask_sc_raw, Correction required")
            run_mask_sc = f'sct_deepseg sc_epi -i ../m{fmriname}_mean.nii.gz -o mask_sc_raw.nii.gz'
            print("### sct_deepseg_sc - Generating mask")
            os.system(f'cd {mask_sc_dir}; {run_mask_sc}; cd ..; chmod a+w sct_deepseg;')

        ###################
        # [Perform manual correction on raw mask]
        #   Save the manually created mask as mask_sc.nii.gz 
        #   Re-run the script with --moco flag
        ##################

        if os.path.exists(os.path.join(sps, self.spine_dir, 'sct_deepseg/mask_sc.nii.gz')):
            print(" Moco run 2: Manually corrected mask_sc found")
            mask_csf_dir = os.path.join(sps, self.spine_dir, 'sct_propseg')
            if not os.path.exists(mask_csf_dir):
                os.mkdir(mask_csf_dir)
            run_mask_csf = f'sct_propseg -i ../m{fmriname}_mean.nii.gz -c t2s -o mask_spinal.nii.gz -CSF'
            print("### sct_propseg_csf - Generating spinal cord csf mask")
            os.system(f'cd {mask_csf_dir}; {run_mask_csf}; cd ..; chmod a+w sct_propseg;')

            os.system(f'cd {mask_csf_dir}; sct_maths -i m{fmriname}_mean_CSF_seg.nii.gz -o mask_csf.nii.gz -sub ../sct_deepseg/mask_sc.nii.gz;\
                    sct_maths -i mask_csf.nii.gz -o mask_csf.nii.gz -thr 1')
            
            segmentation_dir = os.path.join(sps, self.spine_dir, 'Segmentation')
            if not os.path.exists(segmentation_dir):
                os.mkdir(segmentation_dir)
                os.system(f'cd {segmentation_dir}; cd ..; chmod a+w Segmentation;')
            os.system(f'cd {segmentation_dir}; cp ../sct_deepseg/mask_sc.nii.gz ./; cp ../sct_propseg/mask_csf.nii.gz ./; fslcpgeom ../../{self.stc}/fmri_mean.nii.gz mask_csf.nii.gz')

    def func_normalize(self):
        anat_paths = [os.path.join(self.parent_path, s, self.anat, "session1")
                    for s in self.list_subjects]

        print(" ### Info: Functional Normalization starting...")
        start = time.time()
        for s, anat_p in zip(self.list_subjects, anat_paths):
            for ses in self.session:
                func_p = os.path.join(self.parent_path, s, self.func, self.condition, ses, self.spine_dir)
                print("Subject/session:", s, ses, "func_path:", func_p)
                self._register_multimodal(func_p, anat_p, fmriname="fmri_stc_sc")

        print("### Info: Functional Normalization done in %.3f s" % (time.time() - start))
        return

    def _register_multimodal(self, sps, aps, fmriname):
    
        #[T2 Normalization]
        Normalization_t2_dir = os.path.join(sps, 'Normalization_t2')
        if os.path.exists(Normalization_t2_dir):
            print("### T2 func_norm already performed")

        if not os.path.exists(Normalization_t2_dir):
            os.mkdir(Normalization_t2_dir)
        run_string_t2 = f'cd {Normalization_t2_dir}; sct_register_multimodal -i {self.pam50_template}\
            -iseg {self.pam50_template_cord} \
            -d  ../m{fmriname}_mean.nii.gz \
            -dseg ../Segmentation/{self.mask_fname}.nii.gz \
            -param step=1,type=seg,algo=centermass:step=2,type=seg,algo=bsplinesyn,metric=MeanSquares,slicewise=1,iter=3:step=3,type=im,algo=syn,metric=CC,iter=3,slicewise=1 \
            -initwarp {aps}/{self.t2}/vertebral_labels/warp_template2anat.nii.gz \
            -initwarpinv {aps}/{self.t2}/vertebral_labels/warp_anat2template.nii.gz \
            -owarp warp_template2fmri.nii.gz \
            -owarpinv warp_fmri2template.nii.gz'
        
        print(run_string_t2)
        os.system(run_string_t2)

    def normalize(self):
        subj_paths = [os.path.join(self.parent_path, s, self.func, self.condition, ses, self.spine_dir) for s in self.list_subjects for ses in self.session]
        print("### Info: Starting normalization ...") 
        start = time.time()
        for sub in subj_paths:
            self._normalization(sub,fmriname="fmri_stc_sc")
        print("### Info: Normalization to Template done in %.3f s" %(time.time() - start ))
    
    def _normalization(self, sps, fmriname):
        if self.condition == 'rest':
            run_string = f'cd {sps}; sct_apply_transfo -i m{fmriname}_rest_denoised.nii.gz -d {self.pam50_template} -w Normalization_t2/warp_fmri2template.nii.gz\
                        -x linear -o m{fmriname}_rest_denoised_n.nii.gz'
            print(run_string)
            os.system(run_string)

    def apply_smoothing(self):
        subj_paths = [os.path.join(self.parent_path, s, self.func, self.condition, ses, self.spine_dir) for s in self.list_subjects for ses in self.session]
        print("Subject paths currently working with:", subj_paths)
        print("### Info: Applying smoothing ...") 
        start = time.time()
        for sub in subj_paths:
            self._smooth_img(sub,fmriname="fmri_stc_sc", fwhm = [2,2,4])
        print("### Info: smoothing done in %.3f s" %(time.time() - start ))
        return
    
    def _smooth_img(self, sps, fmriname, fwhm):
            smoothing_dir = os.path.join(sps, self.func_smooth)
            if not os.path.exists(smoothing_dir):
                os.mkdir(smoothing_dir)
            
            if self.condition == 'rest':
                func_img = f'{sps}/m{fmriname}_rest_denoised_n.nii.gz'
                o_img = smoothing_dir + '/' + os.path.basename(func_img).split('.')[0] + "_sm.nii.gz"
                if not os.path.exists(o_img):
                    smoothed_image=image.smooth_img(func_img, fwhm)
                    smoothed_image.to_filename(o_img)
                    string='fslmaths '+o_img+' -Tmean '+o_img.split('.')[0] + '_mean.nii.gz'
                    os.system(string)                   
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
    
    print(" ### Info: Running Spincal Cord Preprocessing ... ")  

    PR = PreprocessingRS(config_file)
    if '--crop_sc' in sys.argv:
        print(" ### Info: Running T1 and T2 crop segmentation...")
        PR.crop_sc = True 
    if '--anat_norm' in sys.argv:
        print(" ### Info: Running anatomical to template normalization ...")
        PR.anat_norm = True 
    if '--moco' in sys.argv:
        print(" ### Info: Running motor correction and generating masks...")
        PR.moco = True 
    if '--func_norm' in sys.argv:
        print(" ### Info: Running functional to template normalization...")
        PR.func_norm = True
    if '--normalization' in sys.argv:
        print(" ### Info: Applying normalization (Native -> Template)...")
        PR.normalization = True
    if '--smoothing' in sys.argv:
        print(" ### Info: Applying smoothing (2x2x4)...")
        PR.smoothing = True

    PR.processes()





