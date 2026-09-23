# -*- coding: utf-8 -*-
#!/usr/bin/python

""""
Example:
    python pipeline_denoising.py --config config_denoise.json --pnm0 --pnm1 --pnm2 auto --pnm3 auto --denoising
    python pipeline_denoising.py --config config_denoise.json --pnm0
    python pipeline_denoising.py --config config_denoise.json --pnm1
    python pipeline_denoising.py --config config_denoise.json --pnm2 auto
    python pipeline_denoising.py --config config_denoise.json --pnm3 auto
    python pipeline_denoising.py --config config_denoise.json --denoising_brain
    python pipeline_denoising.py --config config_denoise.json --denoising_spine
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import time
from glob import glob
from joblib import Parallel, delayed
import subprocess
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks

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
        self.mask_fname = "mask_sc"
        self.mask_csf_name = "mask_csf"
        self.pnm0_preptxt = False   # step 0: prepare physio rec (txt file and filter)
        self.pnm1_stage1 = False   # first step: pnm_stage 1
        self.pnm2_peaks = False   # second step: check peaks of cardiac sig
        self.pnm3_gen_evs = False   # third step: generate evs 
        self.cof = 2  # default filter parameter for pnm0
        self.filter = False   # by default don't filter data
        self.mode = ''
        self.csf_mask = False  # default / in cervical should be true
        self.denoising_brain_rest = False
        self.denoising_spine_rest = False

    def processes(self):
        if self.pnm0_preptxt:
            self.prepare_physio()
            os.chdir(self.working_dir)

        if self.pnm1_stage1:
            self.pnm_stage1()
            os.chdir(self.working_dir)

        if self.pnm2_peaks:
            self.pnm_stage2()
            os.chdir(self.working_dir)

        if self.pnm3_gen_evs:
            self.generate_evs()
            os.chdir(self.working_dir)
        
        if self.denoising_brain_rest:
            self.apply_denoising_brain_rest()

        if self.denoising_spine_rest:
            self.apply_denoising_spine_rest()
        
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
                # extend if parameter is a list
                assert isinstance(params[key], list)
                # setattr(self, key, getattr(self, key).extend(params[key]))
                getattr(self, key).extend(params[key])
            else:
                # initialize or overwrite value
                setattr(self, key, params[key])
        if self.task == "rest":
            self.session = ["session1", "session2"]
        elif self.task == "task":
            self.session = ["session1", "session2"]
        
         # assign the list of subjects variable
        self.list_subjects = glob(os.path.join(self.parent_path, self.data_root+'*'))
        subj_found = [os.path.basename(sub.rstrip('/')) for sub in self.list_subjects]
        self.list_subjects = self.select_subjects(subj_found, include=range(1, 20), exclude=[15])
        print(" ### Condition:", self.task)
        print(" ### Session:", self.session)
        print(" ### Subject List:", self.list_subjects)
        
    def prepare_physio(self):
        subj_paths = [os.path.join(self.parent_path, s, self.physio, self.task, ses) for s in self.list_subjects for ses in self.session]
        print("Subject paths currently working with:", subj_paths)
        print(" ### Info: Converting text file ...") 
        start = time.time()
        for sub in subj_paths:
            self._convert_txt_filt(sub)
        print("### Info: File conversion done in %.3f s" %(time.time() - start ))
        return

    def __get_mat_info(self,sub):
        matstructfile = glob(os.path.join(sub, 'S*.mat'))[0]
        print(matstructfile)
        matstruct = loadmat(matstructfile)
        data = matstruct['data']
        FS = 1/matstruct['isi']*1000
        isi = matstruct['isi']
        return matstructfile, data, FS[0][0]

    def _convert_txt_filt(self, sub):

        os.chdir(sub)   
        matstructfile, data, FS = self.__get_mat_info(sub)
        if self.filter:
            print("### Filtering the cardiac signal...")
            cof = self.cof
            Wn = (cof*2)/FS
            if Wn > 1.0:
                Wn = 0.99
            B,A = butter(3,Wn,'low')
            data[:,int(self.pnm_columns['cardiac'])-1] = filtfilt(B,A,data[:,int(self.pnm_columns['cardiac'])-1])

            data = np.array(data)
        file_name_sub = matstructfile.split('/')[-5]
        file_name_session = matstructfile.split('/')[-2]
        np.savetxt(file_name_sub + '_' + file_name_session + '.txt', data, fmt='%.4f', delimiter='\t')

    def pnm_stage1(self):
        subj_paths = [os.path.join(self.parent_path, s, self.physio, self.task, ses) for s in self.list_subjects for ses in self.session]
        print("Subject paths currently working with:", subj_paths)

        print("### Info: Physiological preparation ...") 
        start = time.time()
        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="multiprocessing")(delayed(self._fsl_pnm_stage1)(sub)\
                 for sub in subj_paths)

        print("### Info: Physiological preparation done in %.3f s" %(time.time() - start ))
        return

    def _fsl_pnm_stage1(self, sps):

        subj_name = sps.split('/')[-4]
        session_name = sps.split('/')[-1]
        subjname = subj_name+'_'+session_name
        print(subjname)
        os.chdir(sps) 
        _, _,  FS = self.__get_mat_info(sps)

        run_string = 'cd %s; %sbin/fslFixText ./%s.txt ./%s_input.txt; %sbin/pnm_stage1 -i ./%s_input.txt -o %s -s %s --tr=%s \
        --smoothcard=0.3 --smoothresp=0.1 --resp=%s --cardiac=%s --trigger=%s' % (sps,
                                                                                  self.FSL_PATH, 
                                                                                  subjname,
                                                                                  subjname,
                                                                                  self.FSL_PATH,
                                                                                  subjname,
                                                                                  subjname,                                                                 
                                                                                  str(FS),
                                                                                  str(self.TR),  
                                                                                  self.pnm_columns['resp'],
                                                                                  self.pnm_columns['cardiac'],  
                                                                                  self.pnm_columns['trigger'])                                                                               

        print(run_string)
        os.system(run_string)

    def pnm_stage2(self):
        subj_paths = [os.path.join(self.parent_path, s, self.physio, self.task, ses) for s in self.list_subjects for ses in self.session]
        print("Subject paths currently working with:", subj_paths)
        print("### Info: Physiological preparation ...") 
        start = time.time()
        if self.mode == 'auto':
            Parallel(n_jobs=self.n_jobs,
                     verbose=100,
                     backend="multiprocessing")(delayed(self._check_peaks_car_persub)(sub)\
                     for sub in subj_paths)
        else:
            # manual check of the peaks
            print(" ### SUBJECT %s chosen for single caridac peaks identification! " % self.mode)
            self._check_peaks_car_persub(self.mode, auto=False)

        print("### Info: Physiological preparation done in %.3f s" %(time.time() - start ))
        return

    def _check_peaks_car_persub(self, sps, auto=True):

        if auto:
            sub_path = sps
            subj_name = sps.split('/')[-4]
            session_name = sps.split('/')[-1]
            subname = subj_name+'_'+session_name
        else:
            sub_path = os.path.join(self.parent_path, sps, self.physio)
            subname = sps
        os.chdir(sps)
        _, _, FS = self.__get_mat_info(sps)
        print(f"### Info: FS = {FS}")

        seq_input = np.loadtxt(os.path.join(sub_path, subname+'_input.txt'))
        seq_card = np.loadtxt(os.path.join(sub_path, subname+'_card.txt'))
        
        seq_card = (np.round(seq_card*FS)).astype(int)

        # Find triggers 
        col_tr = int(self.pnm_columns["trigger"])-1   # in python -1 indexing
        print(col_tr)
        triggers = np.where(seq_input[:,col_tr]==5)[0]
        col_car = int(self.pnm_columns["cardiac"])-1 
        card_signal = seq_input[triggers[0]:,col_car]/10

        indices = find_peaks(card_signal)[0]
        
        auto_detect = indices/FS
        np.savetxt(os.path.join(sub_path, subname+'_card_auto.txt'), auto_detect, delimiter='\n', fmt='%.3f') 

    def generate_evs(self):
        subj_paths = [os.path.join(self.parent_path, s, self.physio, self.task, ses) for s in self.list_subjects for ses in self.session]
        print("Subject paths currently working with:", subj_paths)
        print("### Info: Generate EVS ...") 
        start = time.time()
        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="multiprocessing")(delayed(self._fsl_pnm_evs)(sub, fmriname="fmri")\
                 for sub in subj_paths)

        print("### Info: EVS generation done in %.3f s" %(time.time() - start ))
        return 

    def _fsl_pnm_evs(self, sub, fmriname):
        subj_name = sub.split('/')[-4]      # S04
        session_name = sub.split('/')[-1]   # session1, session2, ...
        subname = subj_name + '_' + session_name
        print(subname)
        out_evs = subname
        evs_names = f"{out_evs}_ev0"
        auto_mode = '_auto' if self.mode == 'auto' else ''

        if len(fmriname.split('_')) != 1:
            sliceorder = ""
        else:
            sliceorder = "--sliceorder=up"

        if not os.path.exists(sub + "/bsc_pnmregs/"):
            os.mkdir(sub + "/bsc_pnmregs/")
        
        if not glob(os.path.join(sub, "bsc_pnmregs", "*ev00*.nii.gz")):
            # build func and CSF paths using the *single* session_name
            func_img = f"../../../{self.func}/{self.task}/{session_name}/{fmriname}.nii.gz"
            if self.csf_mask:
                csf_mask_path = (
                    f"../../../{self.func}/{self.task}/{session_name}/"
                    f"{self.spine_dir}/Segmentation/{self.mask_csf_name}.nii.gz"
                )
                pnm_txt_path = (
                    f"../../../{self.dicoms}/{self.task}/{session_name}/pnm_slicetiming_seconds.txt"
                )
                run_string = (
                    f'cd {sub}; {self.FSL_PATH}bin/pnm_evs '
                    f'-i {func_img} '
                    f'-c {subname}_card{auto_mode}.txt -r {subname}_resp.txt '
                    f'-o {sub}/bsc_pnmregs/{out_evs}_ '
                    f'--tr={self.TR} --oc=4 --or=4 --multc=2 --multr=2 '
                    f'--csfmask={csf_mask_path} '
                    f'--slicetiming={pnm_txt_path} --slicedir=z'
                )
            else:
                run_string = (
                    f'cd {sub}; {self.FSL_PATH}bin/pnm_evs '
                    f'-i {func_img} '
                    f'-c {subname}_card{auto_mode}.txt -r {subname}_resp.txt '
                    f'-o {sub}/bsc_pnmregs/{out_evs}_ '
                    f'--tr={self.TR} --oc=4 --or=4 --multc=2 --multr=2 '
                    f'--slicetiming={pnm_txt_path} --slicedir=z'
                )
            print(run_string)
            os.system(run_string)

        ## Cropping regs
        if not os.path.exists(sub + "/brain_pnmregs"):
            os.mkdir(sub + "/brain_pnmregs")
            run_stringperm_b = f'cd {sub}; chmod a+w brain_pnmregs/'
            os.system(run_stringperm_b)
        if not os.path.exists(sub + "/spine_pnmregs"):
            os.mkdir(sub + "/spine_pnmregs")
            run_stringperm_sp = f'cd {sub}; chmod a+w spine_pnmregs/'
            os.system(run_stringperm_sp)
        
        if not glob(os.path.join(sub, "brain_pnmregs", "*ev00*.nii.gz")):
            for i in range(1,33):
                ## BRAIN
                file_name_brain = subname+f"_ev{i:03d}.nii.gz"
                inreg_full_brain = sub + "/bsc_pnmregs/" + file_name_brain
                outreg_crop_brain = sub + "/brain_pnmregs/" + subname+f"_ev{i:03d}_brain.nii.gz"
                crop_string_brain = f"sct_crop_image -i {inreg_full_brain} -o {outreg_crop_brain} -zmin 35 -zmax -1"
                print(crop_string_brain)
                os.system(crop_string_brain)

        if not glob(os.path.join(sub, "spine_pnmregs", "*ev00*.nii.gz")):    
            for i in range(1,34):   
                ## SPINE
                file_name_spine = subname+f"_ev{i:03d}.nii.gz"
                inreg_full_spine = sub + "/bsc_pnmregs/" + file_name_spine
                outreg_crop_spine = sub + "/spine_pnmregs/" + subname+f"_ev{i:03d}_spine.nii.gz"
                crop_string_spine = f"sct_crop_image -i {inreg_full_spine} -o {outreg_crop_spine} -zmin 0 -zmax 34"
                os.system(crop_string_spine)
        
        # Saving the regressor_list.txt for denoising
        if not os.path.exists(sub + "/brain_pnmregs/"+subname+"_evlist.txt"):
            string_evlist_brain = f'ls -1 `{self.FSL_PATH}bin/imglob -extensions {sub}/brain_pnmregs/{evs_names}*` > {sub}/brain_pnmregs/{out_evs}_brain_evlist.txt'
            print(string_evlist_brain)
            os.system(string_evlist_brain)

        if not os.path.exists(sub + "/spine_pnmregs/"+subname+"_evlist.txt"):
            string_evlist_spine = f'ls -1 `{self.FSL_PATH}bin/imglob -extensions {sub}/spine_pnmregs/{evs_names}*` > {sub}/spine_pnmregs/{out_evs}_spine_evlist.txt'
            print(string_evlist_spine)
            os.system(string_evlist_spine)
    
    def apply_denoising_brain_rest(self):
        subj_paths = [os.path.join(self.parent_path, s, self.func, self.task, ses) for s in self.list_subjects for ses in self.session]
        print("Subject paths currently working with:", subj_paths)
        print("### Info: Generate denoising regressors ...") 
        start = time.time()
        for sub in subj_paths:
            self._fsl_feat_regressors_brain_rest(sub, fmriname="fmri_stc_brain_moco", structure="brain")

        print("### Info: Denoising done in %.3f s" %(time.time() - start ))
        return 
    
    def _fsl_feat_regressors_brain_rest(self, sps, fmriname, structure):
        brain_of = sps + self.denoise_func + "brain"
        if not os.path.exists(sps + "/" + self.denoise_func):
            os.mkdir(sps + "/" + self.denoise_func)
            os.mkdir(brain_of)
        
        subj_name = sps.split('/')[-4]
        session_name = sps.split('/')[-1]
        subname = subj_name+'_'+session_name
        print(subname)

        physiopath = os.path.join(
            '/'.join(sps.split('/')[:-3]),
            self.physio,
            self.task,
            session_name,
            self.pnmregs,
        )
        out_reg = "regressors_evlist"
        noise_reg_out = "noise_regression_rest"
        nuisance_txt = 'nuisance'
        moco_params_name = "fmri_stc_brain_moco"   
        moco = "moco"
        add_outname = ""   
        add_outliers = ''
        add_moco     = ''
        OUTLYN = "0"
        
        ## Adding PNM regressors
        if (structure == "brain" and "pnm" in self.denoising_regs and not os.path.exists(os.path.join(brain_of, f"{out_reg}_brain.txt"))):
            print("### Info: adding PNM regressors ...")
            run_string = f'cp {physiopath}/{subname}_brain_evlist.txt {brain_of}/{out_reg}_brain.txt;'
            print(run_string)
            os.system(run_string)
        
        if "pnm" not in self.denoising_regs and "csf" not in self.denoising_regs:
            pnmpaths = ''
        else:
            pnmpaths = os.path.join(brain_of,f"{out_reg}_{structure}.txt")
        ## Adding motion regressor based on FD metric
            if "moco" in self.motionregs:
                print("### Info: adding moco params regressors ...")
                OUTLYN = "1"
                if not os.path.exists(os.path.join(brain_of, f"/Nuisance/{moco}_nohdr.txt")):
                    os.makedirs(os.path.join(brain_of, 'Nuisance'), exist_ok=True)
                    run_string = f"cd {brain_of}; tail -n +1 ../..{self.moco_func}{structure}/{moco_params_name}.nii.gz.par > Nuisance/{moco}_nohdr.txt"
                    print(run_string)
                    os.system(run_string)
                add_outname += "_moco"
                add_moco = f"Nuisance/{moco}_nohdr.txt"
            
            ## Combination of regressors    
            if len(add_moco)!= 0 and len(add_outliers) == 0:
                run_string = f"cd {brain_of}; paste -d '\t' {add_moco} > Nuisance/{nuisance_txt}.txt"
                print(run_string)
                os.system(run_string)

            if len(add_moco)==0 and len(add_outliers)==0:
                nuispath = ''
            else:
                nuispath = os.path.join(brain_of,f"Nuisance/{nuisance_txt}.txt")
        
        if os.uname()[1] in ['srv5','srv6','srv7']:
            add_paths = 'FSLDIR=../fsl; . ${FSLDIR}/etc/fslconf/fsl.sh; PATH=${FSLDIR}/bin:${PATH}; export FSLDIR PATH;'
            fsl_feat = 'feat'
            tmpout = '> ../tmp/out_feat.txt'           
        
        ###################
        #[E] Noise regression 
        ##################
        if not os.path.exists(sps + self.denoise_func + structure + "/noise_regression_rest.feat"):
            os.system(f'export DIREC={sps}{self.denoise_func}{structure}; \
                        export FSL_TEMP={self.fsl_template_rest}; \
                        export FEATOUTPUTNAME={noise_reg_out};\
                        export OUTDIR=$DIREC/{noise_reg_out};\
                        export PNMPATHS={pnmpaths};\
                        export OUTLYN={OUTLYN};\
                        export OUTLPATH={nuispath};\
                        export DATAPATH={sps}/{self.moco_func}/{structure}/{fmriname}.nii.gz;\
                        bash {self.working_dir}/../_templates/rest_template.sh')
            run_string_denoise = f'{add_paths} cd {sps}{self.denoise_func}{structure}; {fsl_feat} {noise_reg_out}.fsf {tmpout}; cp {noise_reg_out}.feat/stats/res4d.nii.gz {fmriname}_rest_denoised_womean.nii.gz;\
                                fslcpgeom {sps}{self.moco_func}{structure}/{fmriname}.nii.gz {fmriname}_rest_denoised_womean.nii.gz'
            print("### Info: running denoising...")          
            print(run_string_denoise)
            os.system(run_string_denoise)
        
        # [For adding the mean back to denoised image]
        if not os.path.exists(os.path.join(sps, self.denoise_func, structure, f"{fmriname}_rest_denoised.nii.gz")):
            run_string_addmean = f'cd {sps}{self.denoise_func}{structure}; fslmaths {fmriname}_rest_denoised_womean.nii.gz -add {noise_reg_out}.feat/mean_func {fmriname}_rest_denoised.nii.gz'
            print(run_string_addmean)
            os.system(run_string_addmean)
            
    def apply_denoising_spine_rest(self):
        subj_paths = [os.path.join(self.parent_path, s, self.func, self.task, ses) for s in self.list_subjects for ses in self.session]
        print("Subject paths currently working with:", subj_paths)
        print("### Info: Generate denoising regressors ...") 
        start = time.time()
        for sub in subj_paths:
            self._fsl_feat_regressors_spine_rest(sub, fmriname="fmri_stc_sc", structure="spine")
        print("### Info: Denoising done in %.3f s" %(time.time() - start ))
        return 
    
    def _fsl_feat_regressors_spine_rest(self, sps, fmriname, structure):
        spine_of = os.path.join(sps , self.spine_dir, self.denoise_spine)
        print(spine_of)
        if not os.path.exists(spine_of):
            os.mkdir(spine_of)
        
        subj_name = sps.split('/')[-4]
        session_name = sps.split('/')[-1]
        subname = subj_name+'_'+session_name
        print(subname)

        physiopath = os.path.join(
            '/'.join(sps.split('/')[:-3]),
            self.physio,
            self.task,
            session_name,
            self.pnmregs,
        )
        print(physiopath)
        out_reg = "regressors_evlist"
        noise_reg_out = "noise_regression_rest"
        nuisance_txt = 'nuisance'
        moco_params_name = "moco_params"   
        moco = "moco"
        add_outname = ""   
        add_outliers = ''
        add_moco     = ''
        OUTLYN = "0"
        
        if structure=="spine" and "csf" in self.denoising_regs and "pnm" in self.denoising_regs:
            run_string = f'cp {physiopath}/{subname}_spine_evlist.txt {spine_of}/{out_reg}_spine.txt;'
            print(run_string)
            os.system(run_string)

        if "pnm" not in self.denoising_regs and "csf" not in self.denoising_regs:
            pnmpaths = ''
        else:
            pnmpaths = os.path.join(spine_of,f"{out_reg}_{structure}.txt")
        ## Adding motion regressor based on FD metric
        if "moco" in self.motionregs:
            print("### Info: adding moco params regressors ...")
            OUTLYN = "1"
            if not os.path.exists(os.path.join(spine_of, f"/Nuisance/{moco}_nohdr.txt")):
                os.makedirs(os.path.join(spine_of, 'Nuisance'), exist_ok=True)
                run_string = f"cd {spine_of}; tail -n +2 ../{self.func_moco}/{moco_params_name}.tsv > Nuisance/{moco}_nohdr.txt"
                print(run_string)
                os.system(run_string)
            add_outname += "_moco"
            add_moco = f"Nuisance/{moco}_nohdr.txt"
            
            ## Combination of regressors    
            if len(add_moco)!= 0 and len(add_outliers) == 0:
                run_string = f"cd {spine_of}; paste -d '\t' {add_moco} > Nuisance/{nuisance_txt}.txt"
                print(run_string)
                os.system(run_string)

            if len(add_moco)==0 and len(add_outliers)==0:
                nuispath = ''
            else:
                nuispath = os.path.join(spine_of,f"Nuisance/{nuisance_txt}.txt")
        
        ## Modify the updated FSL paths
        if os.uname()[1] in ['srv5','srv6','srv7']:
            add_paths = 'FSLDIR=../fsl; . ${FSLDIR}/etc/fslconf/fsl.sh; PATH=${FSLDIR}/bin:${PATH}; export FSLDIR PATH;'
            fsl_feat = 'feat'
            tmpout = '> ../tmp/out_feat.txt'           
        
        ###################
        #[E] Noise regression 
        ##################
        if not os.path.exists(os.path.join(spine_of, f'{noise_reg_out}.feat')):
            os.system(f'export DIREC={spine_of}; \
                        export FSL_TEMP={self.fsl_template_rest}; \
                        export FEATOUTPUTNAME={noise_reg_out};\
                        export OUTDIR=$DIREC/{noise_reg_out};\
                        export PNMPATHS={pnmpaths};\
                        export OUTLYN={OUTLYN};\
                        export OUTLPATH={nuispath};\
                        export DATAPATH={spine_of}/../m{fmriname}.nii.gz;\
                        bash {self.working_dir}/../_templates/rest_template.sh')
            run_string_denoise = f'{add_paths} cd {spine_of}; {fsl_feat} {noise_reg_out}.fsf {tmpout}; cp {noise_reg_out}.feat/stats/res4d.nii.gz m{fmriname}_rest_denoised_womean.nii.gz;\
                                fslcpgeom {sps}/{self.spine_dir}/m{fmriname}.nii.gz m{fmriname}_rest_denoised_womean.nii.gz'
            print("### Info: running denoising...")          
            print(run_string_denoise)
            os.system(run_string_denoise)
        
        # [For adding the mean back to denoised image]
        if not os.path.exists(os.path.join(sps, self.spine_dir, f"m{fmriname}_rest_denoised.nii.gz")):
            run_string_addmean = f'cd {sps}/{self.spine_dir}; fslmaths {self.denoise_spine}/m{fmriname}_rest_denoised_womean.nii.gz -add {self.denoise_spine}/{noise_reg_out}.feat/mean_func m{fmriname}_rest_denoised.nii.gz'
            print(run_string_addmean)
            os.system(run_string_addmean)

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

    if '--pnm0' in sys.argv:
        PR.pnm0_preptxt = True
    if '--pnm1' in sys.argv:
        PR.pnm1_stage1 = True
    if '--pnm2' in sys.argv:
        # check peaks
        ind = sys.argv.index('--pnm2') + 1
        # it will specified the mode: auto or subject 'name'
        PR.mode = sys.argv[ind] 
        PR.pnm2_peaks = True
    if '--pnm3' in sys.argv:
        # generate evs
        # specify whether to read the automatically detetected peaks or not
        ind = sys.argv.index('--pnm3') + 1
        PR.mode = sys.argv[ind] 
        PR.pnm3_gen_evs = True

    if '--denoising_brain_rest' in sys.argv:
        PR.denoising_brain_rest = True
    if '--denoising_spine_rest' in sys.argv:
        PR.denoising_spine_rest = True
    PR.processes()





