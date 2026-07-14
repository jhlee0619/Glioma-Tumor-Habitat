import os
from glob import glob
import numpy as np
import json
import random
import ants
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.multiprocessing import Pool
from scipy.ndimage import morphology


dataroot_server = os.environ.get('DSC_DATAROOT', './data')

class DSC(Dataset):
    def __init__(self, dataroot, phase, transform):
        super().__init__()
        self.dataroot = dataroot
        self.phase = phase
        self.transform = transform

        self.dataset = []
        self.dataset_xyz_idx_dict = {}

        self.append_cohort()

        print(f'Total {phase} dataset size - volume: {len(self.dataset_xyz_idx_dict)}, signal: {len(self.dataset)}')

    def __getitem__(self, index):
        dataroot, patient_id, study_date, patient_dir, BAT, xyz, dsc_signal, dsc_signal_maxv = self.dataset[index]
        
        data = self.transform({'dsc_signal': dsc_signal, 'dsc_signal_maxv': dsc_signal_maxv})
        data.update({'dataroot': dataroot, 'patient_id': patient_id, 'study_date': study_date, 'patient_dir': patient_dir, 'BAT': BAT, 'xyz': xyz})
        
        return data

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.dataset)

    def _scan(self, dataset_):
        with Pool(16) as pool:
            dataset = list(tqdm(pool.imap(self._single_scan, dataset_), total=len(dataset_)))
            dataset = [item for sublist in dataset for item in sublist]
        self.dataset += dataset
    
    def _single_scan(self, dataset_):
        dataroot, patient_id, study_date, BAT, brain_index = dataset_
        
        preprocessed_dir = os.path.join(dataroot, 'preprocessed')
        os.makedirs(preprocessed_dir, exist_ok=True)
        preprocessed_file = os.path.join(preprocessed_dir, f"{patient_id}_{study_date}_sub_data.npy")

        if os.path.exists(preprocessed_file):
            sub_data = np.load(preprocessed_file, allow_pickle=True).tolist()
        else:
            patient_dir = os.path.join(dataroot, 'nifti', patient_id, study_date)
            dsc_path = os.path.join(patient_dir, 'dsc.nii.gz')
            brain_mask_path = os.path.join(patient_dir, 'brain_mask.nii.gz')
            tumor_mask_path = os.path.join(patient_dir, 'corrected_tumor_mask.nii.gz')
            
            dsc = ants.image_read(dsc_path).numpy()
            brain_mask = ants.image_read(brain_mask_path).numpy()
            tumor_mask = ants.image_read(tumor_mask_path).numpy() > 0
            
            dsc[dsc < 0] = 0
            dsc_signal = dsc[tumor_mask]
            dsc_signal_maxv = np.quantile(dsc_signal, 0.995, axis=(1)).reshape(-1, 1)
            dsc_signal = dsc_signal / (dsc_signal_maxv + 1e-8)

            brain_index = tumor_mask.nonzero()
            brain_index = list(zip(*brain_index))

            sub_data = []
            for i in range(len(brain_index)):
                sub_data.append([dataroot, patient_id, study_date, patient_dir, BAT, brain_index[i], dsc_signal[i], dsc_signal_maxv[i]])
            np.save(preprocessed_file, np.array(sub_data, dtype=object))

        return sub_data
    
    def append_cohort(self):
        dataroot_cohort = f'{self.dataroot}/cohort/project/quantized_DSC'

        json_path = os.path.join(dataroot_server, 'cohort/project/quantized_DSC/source_code', f'{self.phase}_list.json')
        self.patient_list = json.load(open(json_path, 'r'))

        patient_dir_list = glob(f'{dataroot_cohort}/nifti/*/*')
        patient_dir_list = [p for p in patient_dir_list if os.path.exists(f'{p}/corrected_tumor_mask.nii.gz')]
        patient_dir_list = [p for p in patient_dir_list if p.split('/')[-2] in self.patient_list]

        dataset_cohort = [[dataroot_cohort, *patient_dir.split('/')[-2:], 0, [0]] for patient_dir in patient_dir_list]
        self._scan(dataset_cohort)

        self.dataset_xyz_idx_dict.update({f"{patient_dir.split('/')[-2]}_{patient_dir.split('/')[-1]}": {'BAT': 0, 'brain_index': [0]} for patient_dir in patient_dir_list})


class DSCTrain(DSC):
    def __init__(self, dataroot, transform):
        super().__init__(dataroot, phase='train', transform=transform)


class DSCValidation(DSC):
    def __init__(self, dataroot, transform):
        super().__init__(dataroot, phase='valid', transform=transform)


class DSCInference(Dataset):
    def __init__(self, dataroot, transform):
        super().__init__()
        self.dataroot = dataroot
        self.transform = transform

        self.dataset = []
        
        self.append_dataset(dataroot)

        print(f'Total inference dataset size - volume: {len(self.dataset)}')

    def __getitem__(self, index):
        patient_dir, dsc_signal, dsc_signal_maxv, brain_mask, tumor_mask = self._single_scan(self.dataset[index])
        
        return {'patient_dir': patient_dir, 'dsc_signal': dsc_signal, 'dsc_signal_maxv': dsc_signal_maxv, 'brain_mask': brain_mask, 'tumor_mask': tumor_mask}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.dataset)
    
    def _single_scan(self, patient_dir):
        dsc_path = os.path.join(patient_dir, 'dsc', 'dsc.nii')
        brain_mask_path = os.path.join(patient_dir, 'brain_mask.nii.gz')
        tumor_mask_path = os.path.join(patient_dir, 'tumor_mask.nii.gz')
        
        dsc_signal = ants.image_read(dsc_path).numpy()
        brain_mask = ants.image_read(brain_mask_path).numpy()
        tumor_mask = ants.image_read(tumor_mask_path).numpy() > 0
        
        dsc_signal[dsc_signal < 0] = 0
        dsc_signal_maxv = np.expand_dims(np.quantile(dsc_signal, 0.995, axis=(3)), -1)
        dsc_signal = dsc_signal / (dsc_signal_maxv + 1e-8)
        
        dsc_signal = torch.tensor(dsc_signal, dtype=torch.float32)
        dsc_signal_maxv = torch.tensor(dsc_signal_maxv, dtype=torch.float32)
        brain_mask = torch.tensor(brain_mask, dtype=torch.bool)
        tumor_mask = torch.tensor(tumor_mask, dtype=torch.bool)
        
        return [patient_dir, dsc_signal, dsc_signal_maxv, brain_mask, tumor_mask]
        
    def append_dataset(self, dataroot):
        patient_dir_list = glob(f'{dataroot}/*')
        patient_dir_list = [p for p in patient_dir_list if os.path.exists(f'{p}/dsc/dsc.nii')]
        # patient_dir_list = [p for p in patient_dir_list if '59939751' in p]
        # patient_dir_list = patient_dir_list[:1]
        
        self.dataset += [patient_dir for patient_dir in patient_dir_list]
