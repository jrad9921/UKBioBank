#%%
import os
import pandas as pd
from ImageImport import get_dicom_tag
import pydicom as pyd
import zipfile
import nibabel as nib
import shutil
import numpy as np

def DICOM_Metadata(paths, modality, output_path):
    if modality == "CT":
        df = pd.Series({})
        lst_downmost=[]

        for files in paths:
            file_list= os.listdir(files)
            lst_dicom_data =[]
            file_list = [file_name for file_name in file_list if file_name.lower().endswith(".dcm")]

            if len(file_list) ==0:
                raise FileNotFoundError("The selected folder has no DICOM image files")
    
            image_position_x = []
            image_position_y = []
            image_position_z = []
            sop_instance_uid = []
            #Slice positions per file in each image folder#
            for file_name in file_list:
                #Header load
                dcm = pyd.dcmread(os.path.join(files, file_name),
                                  stop_before_pixels=True, force=True)
        #Find slice of origin of the patients:
                slice_origin = get_dicom_tag(dcm_seq= dcm, tag= (0x0020, 0x0032), tag_type="multi_float", default=np.array([0.0,0.0,0.0]))[::-1] #This gets the first pixel in each slice
        # Update with slice positions
                image_position_x += [slice_origin[2]]
                image_position_y += [slice_origin[1]]
                image_position_z += [slice_origin[0]]

        # Find the sop instance UID of each slice.
                slice_sop_instance_uid = get_dicom_tag(dcm_seq=dcm, tag=(0x0008, 0x0018), tag_type="str")

        # Update with the slice SOP instance UID.
                sop_instance_uid += [slice_sop_instance_uid]
                lst_dicom_data += [dcm]
    # Order ascending position (DICOM: z increases from feet to head)
            file_table = pd.DataFrame({"file_name": file_list,
                                        "position_z": image_position_z,
                                        "position_y": image_position_y,
                                        "position_x": image_position_x,
                                        "sop_instance_uid": sop_instance_uid}).sort_values(by=["position_z",
                                                                                                "position_y",
                                                                                                "position_x"])
#%%
def NIFTI_Metadata(paths, zipped= False, output_path= None, pattern=None):
    '''
    Get Metadata from a Nifti files.
    paths: list of strings, list of paths to folders or zips with the niftis
    modality: string, Image modality used for the experiment
    output_path: string, path to where  we can operate with the csv files
    zipped: boolean, True if the files containing niftis are zipped
    pattern: string. Specific niftis to search for.
    '''
    if pattern == None:
         raise ValueError("Please input a pattern to search for for the nifti name")
    lst =[]

    if zipped == True:
        for file in paths:
                hd = load_zip_nifti_header(in_path=file, out_path=output_path, file_string=pattern)
                df= pd.Series({
                        "Files" : os.path.join(file, pattern),
                        "Number_x_slices" :hd["dim"][1], 
                        "Number_y_slices" :hd["dim"][2],
                        "Number_z_slices" :hd["dim"][3],
                        "timepoints":hd["dim"][4],
                        "x_dim" :hd["pixdim"][1],
                        "y_dim" :hd["pixdim"][2],
                        "z_dim" :hd["pixdim"][3],
                        "Bits_p_pixel": hd["bitpix"],
                        "Bits_type": hd["datatype"]

                })
                lst.append(df)  

    else:
        for file in paths:
                hd = nib.load(os.path.join(file, pattern)).header
                df= pd.Series({
                        "Files" : os.path.join(file, pattern),
                        "Number_x_slices" :hd["dim"][1], 
                        "Number_y_slices" :hd["dim"][2],
                        "Number_z_slices" :hd["dim"][3],
                        "timepoints":hd["dim"][4],
                        "x_dim" :hd["pixdim"][1],
                        "y_dim" :hd["pixdim"][2],
                        "z_dim" :hd["pixdim"][3],
                        "Bits_p_pixel": hd["bitpix"],
                        "Bits_type": hd["datatype"]

                })
                lst.append(df)            


    df= pd.DataFrame(lst)
    return df

#
def load_zip_nifti_header(in_path, out_path, file_string):
     with zipfile.ZipFile(in_path) as fil:
          fil.extract(file_string, path=out_path)
          hd=nib.load(file_string).header
          shutil.rmtree(out_path)
     return hd
