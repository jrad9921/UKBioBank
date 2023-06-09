#%%i
import os
import pandas as pd
import numpy as np
from ImageImport import read_dicom_series, read_dicom_series_zip, read_nifti_zip
import nibabel as nib
import torch
from torch.utils.data import Dataset
from Metadata_Functions import DICOM_Metadata
########################################################################
#%%

class ImageDataSet(Dataset):
    def __init__(self, experiment_numpy_path, labels):
        self.np_tensor= torch.to_tensor(np.load(experiment_numpy_path))
        self.labels= labels
    def __len__(self):
        return len(self.file_paths)
    def __getitem__(self, index):
        return self.np_array[index,:,:,:], labels[index]
    



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
        
        return np.save(output_path, img)
    
#%%
def MRI_dataset_prepare(paths, file_ext, output_path, modality="MRI", preprocess_steps= None, zipped=False, pattern=None, mask=False):
    if file_ext not in ["dcm", "DICOM","nii","nii.gz","nifti"]:
        raise ValueError("The extension of the files is not any of the correct ones, it should be: dcm, DICOM, nii, nii.gz, nifti ")
    
    if zipped == True and pattern is None and file_ext in ["nii", "nii.gz", "nifti"] :
            raise ValueError("If the files are niftis and they are zipped, a name of the file inside to extract has to be given")
 
    if mask:
        print("Howdyyyy still in cosntruction to add the mask :)")
    ###############################################################
    #create tensors first
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
                
        if file_ext in ["nii", "nii.gz", "nifti"] and zipped != True:
            img= []
            output_path= output_path +"//" + "experiment_no_pre.npz"
            for im in paths:
                img.append(nib.load(im))
                #Add more functionality

        if file_ext in ["nii", "nii.gz", "nifti"] and zipped == True:
            img= []
            outp_1 = os.path.join(output_path,"extracted.csv")
            output_path= os.path.join(output_path,"experiment_no_pre.npz")
            for im in paths:
                 a, b = read_nifti_zip(in_path=im, out_path=output_path, file_string= pattern)
                 if b=="1":
                    img.append(torch.from_numpy(a))
        #Equilibrate the dimensions across arrays 
        dim1 = max([a.shape[0] for a in img])        
        dim2 = max([a.shape[1] for a in img])
        dim3 = min([a.shape[2] for a in img])
        #resize based on the size of the biggest x and y and smallest deth
        #Important, this is organ-specific! Our brain images have lots of different and reducndant depth because of the mouth

        for im in img:   
            if im.shape[2]> dim3:
                
            if im.shape[0] < dim1:
                a= dim1-im.shape[0]

            if im.shape[1]< dim2:
                b= dim2 - im.shape[1]
            pads= [a/2,a/2,b/2,b/2,0,0]
            if a % 2 !=0:
                pads[0], pads[1]=int(a/2), int(a/2) + 1
            if b%2 !=0:
                pads[2], pads[3]=int(b/2), int(b/2) + 1

            im=torch.nn.functional.pad(input=img, pad= pads, mode="constant", value=0.)
            
        np.savez(output_path, *img)
        extraction = pd.DataFrame({"Extracted": extraction}) 
        extraction.to_csv(outp_1,index=False)
        return "Extraction completed! :)"
#%%    
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
