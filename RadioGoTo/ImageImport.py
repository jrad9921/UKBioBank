
import os
import pandas as pd
import pydicom as pyd
import numpy as np
import zipfile
import nibabel as nib
import shutil
###################################################################


def read_dicom_series(image_folder, modality=None, series_uid=None):

    file_list= os.listdir(image_folder)
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
        dcm = pyd.dcmread(os.path.join(image_folder, file_name),
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
    # Order ascending position (DICOM: z increases from feet to head)
    file_table = pd.DataFrame({"file_name": file_list,
                               "position_z": image_position_z,
                               "position_y": image_position_y,
                               "position_x": image_position_x,
                               "sop_instance_uid": sop_instance_uid}).sort_values(by=["position_z",
                                                                                      "position_y",
                                                                                      "position_x"])
    n_x = get_dicom_tag(dcm_seq=dcm, tag=(0x0028, 0x011), tag_type="int")
    n_y = get_dicom_tag(dcm_seq=dcm, tag=(0x0028, 0x010), tag_type="int")

    # Create an empty voxel grid. Use z, y, x ordering for consistency
    voxel_grid = np.zeros((len(file_table), n_y, n_x), dtype=np.float32)
    slice_dcm_list = [pyd.dcmread(os.path.join(image_folder, file_name), stop_before_pixels=False, force=True) for file_name in file_table.file_name.values]

    # Iterate over the different slices to fill out the voxel_grid.
    for ii, file_name in enumerate(file_table.file_name.values):

        # Read the dicom file and extract the slice grid
        slice_dcm = slice_dcm_list[ii]
        slice_grid = slice_dcm.pixel_array.astype(np.float32)

        # Update with scale and intercept. These may change per slice.
        rescale_intercept = get_dicom_tag(dcm_seq=slice_dcm, tag=(0x0028, 0x1052), tag_type="float", default=0.0)
        rescale_slope = get_dicom_tag(dcm_seq=slice_dcm, tag=(0x0028, 0x1053), tag_type="float", default=1.0)
        slice_grid = slice_grid * rescale_slope + rescale_intercept

        # Convert all images to SUV at admin
        #if get_dicom_tag(dcm_seq=dcm, tag=(0x0008, 0x0060), tag_type="str") == "PT":
            #suv_conversion_object = SUVscalingObj(dcm=slice_dcm)
            #scale_factor = suv_conversion_object.get_scale_factor(suv_normalisation="bw")

            # Convert to SUV
            #slice_grid *= scale_factor

            # Update the DICOM header
            #slice_dcm = suv_conversion_object.update_dicom_header(dcm=slice_dcm)

        # Store in voxel grid
        voxel_grid[ii, :, :] = slice_grid

    return voxel_grid

def read_nifti_zip(in_path, out_path, file_string):
    print(in_path)
    with zipfile.ZipFile(in_path) as fil:
        if file_string in fil.namelist():
          existence ="1"
          fil.extract(file_string, path=out_path)
          im=nib.load(os.path.join(out_path, file_string)).get_fdata()
          shutil.rmtree(out_path)
        else:
            existence = "0"
            im = None
    return im, existence

##############################################################################################################################
def read_dicom_series_zip(zip_folder, modality= None, series_uid=None):
    image_position_x = []
    image_position_y = []
    image_position_z = []

    with zipfile.ZipFile(zip_folder) as zf:
        file_list= zf.namelist()
        file_list=[file_name for file_name in file_list if file_name.lower().endswith(".dcm")]
        if len(file_list) ==0:
            raise FileNotFoundError("The selected folder has no DICOM image files")
        for  file_name in file_list:
            with zf.open(file_name) as fil:
                dcm= pyd.dcmread(fil, stop_before_pixels=True, force= True)
            slice_origin = get_dicom_tag(dcm_seq= dcm, tag= (0x0020, 0x0032), tag_type="multi_float", default=np.array([0.0,0.0,0.0]))[::-1] #This gets the first pixel in each slice
        # Update with slice positions
            image_position_x += [slice_origin[2]]
            image_position_y += [slice_origin[1]]
            image_position_z += [slice_origin[0]]

    #Slice positions per file in each image folder#
    # Order ascending position (DICOM: z increases from feet to head)
    file_table = pd.DataFrame({"file_name": file_list,
                               "position_z": image_position_z,
                               "position_y": image_position_y,
                               "position_x": image_position_x}).sort_values(by=["position_z",
                                                                                      "position_y",
                                                                                      "position_x"])
    n_x = get_dicom_tag(dcm_seq=dcm, tag=(0x0028, 0x011), tag_type="int")
    n_y = get_dicom_tag(dcm_seq=dcm, tag=(0x0028, 0x010), tag_type="int")

    # Create an empty voxel grid. Use z, y, x ordering for consistency
    voxel_grid = np.zeros((len(file_table), n_y, n_x), dtype=np.float32)
    slice_dcm_list =[]
    with zipfile.ZipFile(zip_folder) as zf:
        for file_name in file_table.file_name.values:
            with zf.open(file_name) as f:
                slice_dcm_list.append(pyd.dcmread(f, stop_before_pixels=False, force=True))

    # Iterate over the different slices to fill out the voxel_grid.
    for ii, file_name in enumerate(file_table.file_name.values):

        # Read the dicom file and extract the slice grid
        slice_dcm = slice_dcm_list[ii]
        slice_grid = slice_dcm.pixel_array.astype(np.float32)

        # Update with scale and intercept. These may change per slice.
        rescale_intercept = get_dicom_tag(dcm_seq=slice_dcm, tag=(0x0028, 0x1052), tag_type="float", default=0.0)
        rescale_slope = get_dicom_tag(dcm_seq=slice_dcm, tag=(0x0028, 0x1053), tag_type="float", default=1.0)
        slice_grid = slice_grid * rescale_slope + rescale_intercept

        # Convert all images to SUV at admin
        #if get_dicom_tag(dcm_seq=dcm, tag=(0x0008, 0x0060), tag_type="str") == "PT":
            #suv_conversion_object = SUVscalingObj(dcm=slice_dcm)
            #scale_factor = suv_conversion_object.get_scale_factor(suv_normalisation="bw")

            # Convert to SUV
            #slice_grid *= scale_factor

            # Update the DICOM header
            #slice_dcm = suv_conversion_object.update_dicom_header(dcm=slice_dcm)

        # Store in voxel grid
        voxel_grid[ii, :, :] = slice_grid

    return voxel_grid
################################################################################################################################

def get_dicom_tag(dcm_seq, tag, tag_type=None, default=None, test_tag=False):
    """
    Function used to retrieve single metadata tags from a dicom sequence
    """
    tag_value = default

    try:
        tag_value= dcm_seq[tag].value
    except KeyError:
        if test_tag:
            return False
        else:
            pass
    if test_tag:
        return True
    
    if isinstance(tag_value, bytes):
        tag_value= tag_value.decode("ASCII")
    #Empty entries:
    if tag_value is not None:
        if tag_value == "":
            tag_value=default
    #Cast to correct types:

    if tag_value is not None:
        if tag_type == "str":
            tag_value = str(tag_value)
    
    if tag_value is not None:

        # String
        if tag_type == "str":
            tag_value = str(tag_value)

        # Float
        elif tag_type == "float":
            tag_value = float(tag_value)

        # Multiple floats
        elif tag_type == "mult_float":
            tag_value = [float(str_num) for str_num in tag_value]

        # Integer
        elif tag_type == "int":
            tag_value = int(tag_value)

        # Multiple floats
        elif tag_type == "mult_int":
            tag_value = [int(str_num) for str_num in tag_value]

        # Boolean
        elif tag_type == "bool":
            tag_value = bool(tag_value)

        elif tag_type == "mult_str":
            tag_value = list(tag_value)

    return tag_value

    

