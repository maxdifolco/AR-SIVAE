import os 
import sys
import pandas as pd
import configparser
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from math import degrees

def load_array(path):
    file  = nib.load(path)
    return file.get_fdata()

def _get_first_slice(data):
    counts = []
    slices = []
        
    for k in range(data.shape[2]):
        _, cnts = np.unique(data[:,:,k], return_counts=True)
        counts.append(sum(cnts[1:]))
        ind = np.where(cnts > 10)[0] # at least 10 pixels of each entit
        slices.append(True) if len(ind) == 4 else slices.append(False)
    ind = np.where(slices)[0]
    return ind[0] if counts[ind[0]] > counts[ind[-1]] else ind[-1]

def _find_ed(data):
    [_,j] = np.unique(data[:,:,5,:], return_index=True, axis=2)
    # j return like [0 t1 t2]; the bigger sum of the t1 and t2 moment correspond to the ED, the other to ES
    ed = j[1] if data[:,:,:,j[1]].sum() > data[:,:,:,j[2]].sum() else j[2]
    return int(ed)

def resample_images(itk_image, spacing = (1.3,1.3,10.0), is_label = False):

    #itk_image = sitk.DICOMOrient(itk_image,'LPS')

    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    #print(original_size,original_spacing)

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / spacing[2])))]
    #print(out_size, spacing)

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())

    if is_label:
        resample.SetDefaultPixelValue(0)
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
        resample.SetInterpolator(sitk.sitkBSpline)

    itk_image = resample.Execute(itk_image)

    return itk_image

def get_slice(image, n_slice = 0):
    
    size = list(image.GetSize())
    if len(size) == 4:
        size[3] = 0
        index = [0, 0, 0, n_slice]
        extractor = sitk.ExtractImageFilter()
        extractor.SetSize(size)
        extractor.SetIndex(index)
        image = extractor.Execute(image)
    return image


def _lv_myo_center(segmentation, labels):
    [i, j] = np.where(segmentation == (labels['LV'] or labels['Myo']))
    return np.array([np.mean(i), np.mean(j)])


def _rv_center(segmentation, labels):
    [i, j] = np.where(segmentation == labels['RV'])
    return np.array([np.mean(i), np.mean(j)])


def compute_shift(segmentation, labels):
    segmentation_center = np.array(segmentation.shape[:2]) // 2
    return (_lv_myo_center(segmentation, labels) - segmentation_center).astype(float)


def compute_rotation(segmentation, labels):
    lv_myo_center = _lv_myo_center(segmentation, labels)
    rv_center = _rv_center(segmentation, labels)
    #print(lv_myo_center,rv_center)

    centers_diff = rv_center - lv_myo_center
    #print(centers_diff)

    rotation_angle = degrees(np.arctan2(centers_diff[1], centers_diff[0]))
    # print(rotation_angle)
    #rotation_angle = 90 - ((rotation_angle + 360) % 360)
    #rotation_angle = ((rotation_angle + 360) % 360) - 90
    rotation_angle = -90-(rotation_angle+360)%360

    return rotation_angle


def rotate_image(image, angle):
    # Input angle in degree

    # Define the rotation center as the center of the image
    center = np.array(image.GetSize()) / 2

    # Create a rotation transform
    rotation_transform = sitk.Euler2DTransform()
    rotation_transform.SetCenter(center)
    rotation_transform.SetAngle(np.deg2rad(angle))

    # Apply the rotation transform to the image
    rotated_image = sitk.Resample(image, image, rotation_transform)

    # Convert the rotated image back to a numpy array
    rotated_array = sitk.GetArrayFromImage(rotated_image)

    return rotated_image, rotated_array


def shift_image(image, shift):
    # Create a SimpleITK image from the matrix

    shift = list(shift[::-1])
    # Create a translation transform

    if isinstance(image, np.ndarray):
        image = sitk.GetImageFromArray(image)
    translation_transform = sitk.TranslationTransform(image.GetDimension())
    translation_transform.SetParameters(shift)

    # Apply the translation transform to the image
    shifted_image = sitk.Resample(image, image, translation_transform)

    # Convert the shifted image back to a numpy array
    shifted_array = sitk.GetArrayFromImage(shifted_image)

    return shifted_image, shifted_array


def align_case(image, segmentation, labels = {'LV':1, 'Myo':2, 'RV':3}):
    ### Input as sitk image

    shift = compute_shift(segmentation, labels)
    #print(shift)

    # Apply shift
    shifted_seg, seg_array = shift_image(segmentation, shift)
    shifted_image, _ = shift_image(image, shift)

    angle = compute_rotation(seg_array, labels)
    # Apply rotation
    t_segmentation, t_seg_array = rotate_image(shifted_seg, angle)
    t_image, t_img_array = rotate_image(shifted_image, angle)

    return t_img_array, t_seg_array


        