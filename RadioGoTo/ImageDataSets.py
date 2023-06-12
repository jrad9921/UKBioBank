
import os
import numpy as np
from ImageImport import read_dicom_series, read_dicom_series_zip, read_nifti_zip
import nibabel as nib
import torch
from torch.utils.data import Dataset
from Metadata_Functions import DICOM_Metadata
import helpers
import h5py
from pathlib import Path
########################################################################
#Initially from brannislav1991 on GitHub:
class ImageDataSetH5(Dataset):

    """Represents an abstract HDF5 Image dataset.
    
    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        subdirectories: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(self, file_path, subdirectories, load_data, data_cache_size=3, transform=None):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform

        # Search for all h5 files
        p = Path(file_path)
        assert(p.is_dir())
        if subdirectories:
            files = sorted(p.glob('**/*.h5'))
        else:
            files = sorted(p.glob('*.h5'))
        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')

        for h5dataset_fp in files:
            self._add_data_infos(str(h5dataset_fp.resolve()), load_data)
        
    def __getitem__(self, index):
        # get data
        x = self.get_data("data", index)
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x)

        # get label
        y = self.get_data("label", index)
        y = torch.from_numpy(y)
        return (x, y)
    
    def __len__(self):
        return len(self.get_data_infos('data'))
    
    def _add_data_infos(self, file_path, load_data):
        with h5py.File(file_path) as h5_file:
            # Walk through all groups, extracting datasets
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    # if data is not loaded its cache index is -1
                    idx = -1
                    if load_data:
                        # add data to the data cache
                        idx = self._add_to_cache(ds.value, file_path)
                    
                    # type is derived from the name of the dataset; we expect the dataset
                    # name to have a name such as 'data' or 'label' to identify its type
                    # we also store the shape of the data in case we need it
                    self.data_info.append({'file_path': file_path, 'type': dname, 'shape': ds.value.shape, 'cache_idx': idx})

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        with h5py.File(file_path) as h5_file:
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    # add data to the data cache and retrieve
                    # the cache index
                    idx = self._add_to_cache(ds.value, file_path)

                    # find the beginning index of the hdf5 file we are looking for
                    file_idx = next(i for i,v in enumerate(self.data_info) if v['file_path'] == file_path)

                    # the data info should have the same index since we loaded it in the same way
                    self.data_info[file_idx + idx]['cache_idx'] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [{'file_path': di['file_path'], 'type': di['type'], 'shape': di['shape'], 'cache_idx': -1} if di['file_path'] == removal_keys[0] else di for di in self.data_info]

    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self, type):
        """Get data infos belonging to a certain type of data.
        """
        data_info_type = [di for di in self.data_info if di['type'] == type]
        return data_info_type

    def get_data(self, type, i):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """
        fp = self.get_data_infos(type)[i]['file_path']
        if fp not in self.data_cache:
            self._load_data(fp)
        
        # get new cache_idx assigned by _load_data_info
        cache_idx = self.get_data_infos(type)[i]['cache_idx']
        return self.data_cache[fp][cache_idx]

    



#%%
###########################
def Metadata_caller(file_paths, output_path, modality, extension):

    if extension in ["dcm","DICOM",".dcm"]:
        DICOM_Metadata(paths= file_paths, modality=modality, output_path=output_path)
    else: 
        print("aaaa")


#%%
def CT_dataset_prepare(paths, file_ext, output_path, modality="CT", preprocess_steps= None, zipped= False, pattern= None):
    if file_ext not in ["dcm", "DICOM","nii","nii.gz","nifti"]:
        raise ValueError("The extension of the files is not any of the correct ones, it should be: dcm, DICOM, nii, nii.gz, nifti ")
    
    if zipped == True and pattern is None and file_ext in ["nii", "nii.gz", "nifti"] :
        raise ValueError("If the files are niftis and they are zipped, a name of the file inside to extract has to be given")

    if preprocess_steps == None:
        if file_ext in ["dcm","DICOM"] and zipped ==False:
            img=[]
            output_path= output_path +"//" + "experiment_no_pre.npz"
            for im in paths:
                img.append(read_dicom_series(image_folder=im, modality= modality))

        if file_ext in ["dcm","DICOM"] and zipped ==True:
            img=[]
            output_path= output_path +"//" + "experiment_no_pre.npz"
            for im in paths:
                img.append(read_dicom_series_zip(image_folder=im, modality= modality))   

        if file_ext in ["nii", "nii.gz", "nifti"] and zipped == True:
            img= []
            output_path= os.path.join(output_path,"experiment_no_pre.npz")
            for im in paths:
                img.append(read_nifti_zip(in_path=im, out_path=output_path, file_string= pattern))
        
        return np.save(output_path, img) #To change to h5
    
###############################################

def MRI_dataset_prepare(paths, file_ext, output_path, modality="MRI", preprocess_steps= None, zipped=False, pattern=None, mask=False):
    if file_ext not in ["dcm", "DICOM",".nii","nii",".nii.gz","nii.gz","nifti"]:
        raise ValueError("The extension of the files is not any of the correct ones, it should be: dcm, DICOM, nii, nii.gz, nifti ")
    
    if zipped == True and pattern is None and file_ext in ["nii",".nii", "nii.gz",".nii.gz", "nifti"] :
            raise ValueError("If the files are niftis and they are zipped, a name of the file inside to extract has to be given")
 
    if mask:
        print("Howdyyyy still in construction to add the mask :)")
    ###############################################################
    #create tensors first
    if preprocess_steps == None:
        if file_ext in ["dcm","DICOM"] and zipped ==False:
            img=[]
            output_path_1= os.path.join(output_path, "experiment_no_pre.h5")
            for im in paths:
                img.append(read_dicom_series(image_folder=im, modality= modality))

        if file_ext in ["dcm","DICOM"] and zipped ==True:
            img=[]
            output_path_1= os.path.join(output_path, "experiment_no_pre.h5")
            for im in paths:
                img.append(read_dicom_series_zip(image_folder=im, modality= modality))   
                
        if file_ext in ["nii", "nii.gz", "nifti"] and zipped != True:
            img= []
            output_path_1= os.path.join(output_path, "experiment_no_pre.h5")
            
            for im in paths:
                img.append(nib.load(im))
                #Add more functionality

        if file_ext in ["nii", "nii.gz", "nifti"] and zipped == True:
            img= []
            
            output_path_1= os.path.join(output_path,"experiment_no_pre.h5")
            output_path_2 = os.path.join(output_path,"Temp_Images")
            os.mkdir(path=output_path_2)
            
            h5f= h5py.File(output_path_1,"w")

            tns = h5f.create_group("Tensors")
            pts=paths.copy() #If not the entire thing is deleted and does not work
            for im in paths:
                 a, b = read_nifti_zip(in_path=im, out_path=output_path_2, file_string= pattern)
                 if b=="1":
                    img.append(torch.from_numpy(a))
                 else: pts.remove(im) 
        #Equilibrate the dimensions across arrays 
        dim1 = max([a.shape[0] for a in img])        
        dim2 = max([a.shape[1] for a in img])
        dim3 = max([a.shape[2] for a in img])

        #resize based on the size of the biggest x and y and smallest deth
        #Important, this is organ-specific! Our brain images have lots of different and reducndant depth because of the mouth

        for i, im in enumerate(img):
            pad_2= int(dim3 - im.shape[2])
            pad_0= int(dim1 - im.shape[0])
            pad_1= int(dim2 - im.shape[1])
            #Add padding for both even and uneven slice differences
            pads= [pad_2//2,pad_2//2,pad_1//2,pad_1//2,pad_0//2,pad_0//2]
            if pad_0 % 2 !=0:
                pads[4], pads[5]=pad_0//2, (pad_0//2) + 1
            if pad_1%2 !=0:
                pads[2], pads[3]=pad_1//2, (pad_1//2) + 1
            if pad_2%2 !=0:
                pads[0], pads[1]=pad_2//2, (pad_2//2) + 1
            print (pads)
            img[i]=torch.nn.functional.pad(input=im, pad= pads, mode="constant", value=0.)
            tns.create_dataset(name=pts[i], data=img[i])
        h5f.close()
        return "Extraction completed! :)"
 ####################################################   

def PET_dataset_prepare(paths, file_ext, output_path, modality="PET", preprocess_steps= None):
    if file_ext not in ["dcm", "DICOM","nii","nii.gz","nifti"]:
        raise ValueError("The extension of the files is not any of the correct ones, it should be: dcm, DICOM, nii, nii.gz, nifti ")
    
    if preprocess_steps == None:
        if file_ext in ["dcm","DICOM"]:
            img=[]
            output_path= output_path +"//" + "experiment_no_pre.npy"
            for im in paths:
                img.append(read_dicom_series(image_folder=im, modality= modality))
                
        if file_ext in ["nii", "nii.gz", "nifti"]:
            img= []
            output_path= output_path +"//" + "experiment_no_pre.npy"
            for im in paths:
                img.append(nib.load(im))

        return np.save(output_path, img)
    ####################################################################################
