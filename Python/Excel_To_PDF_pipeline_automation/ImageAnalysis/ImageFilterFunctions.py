import scipy.ndimage as ndimage
import numpy as np
import skimage.filters.rank as rank_mod
import skimage.morphology as disk_mod 
from past.utils import old_div
import itk
import SimpleITK as sitk


def moving_variance_python(input_image, radius=5):
    """ Applies variance filter with scipy.

    :param input_image: image
    :type input_image: Image
    :param radius: radius, defaults to 5
    :type radius: int, optional
    :return: variance
    :rtype: ndarray
    """
    input_image = input_image.astype(np.double)
    local_mean_x = ndimage.uniform_filter(input_image, size=radius * 2 + 1)
    local_mean_x2 = ndimage.uniform_filter(
        input_image ** 2, size=radius * 2 + 1)
    return np.maximum(local_mean_x2 - local_mean_x ** 2, 0)


def moving_variance_skimage(input_image, radius=5):
    """ Applies variance filter with skimage.

    :param input_image: image
    :type input_image: Image
    :param radius: radius, defaults to 5
    :type radius: int, optional
    :return: variance
    :rtype: ndarray
    """
    local_mean_x = skimage_rank(
        rank_mod.mean, input_image, disk_mod.disk(radius))
    local_mean_x2 = skimage_rank(
        rank_mod.mean, input_image ** 2, disk_mod.disk(radius))
    return np.maximum(local_mean_x2 - local_mean_x ** 2, 0)


def moving_sigma_sitk(input_image, radius=5):
    """Implements a fast rectangular sigma filter, with SimpleITK, using the accumulator approach. 

    :param input_image: image
    :type input_image: ndarray
    :param radius: radius, defaults to 5
    :type radius: int, optional
    :return: Image with sigma filter
    :rtype: ndarray
    """
    sitk_im = sitk.GetImageFromArray(input_image)
    output_image = sitk.BoxSigma(sitk_im, radius=[radius, radius])
    return np.copy(sitk.GetArrayFromImage(output_image))


def moving_mean_sitk(input_image, radius=5):
    """Implements a fast rectangular mean filter,with SimpleITK, using the accumulator approach. 

    :param input_image: image as array
    :type input_image: ndarray
    :param radius: radius, defaults to 5 
    :type radius: int, optional
    :return: image with mean filter
    :rtype: ndarray
    """
    sitk_im = sitk.GetImageFromArray(input_image)
    output_image = sitk.BoxMean(sitk_im, radius=[radius, radius])
    return np.copy(sitk.GetArrayFromImage(output_image))


def moving_noise_sitk(input_image, radius=5):
    """Applies noise image filter with SimpleITK.

    :param input_image: image
    :type input_image: ndarray
    :param radius: radius, defaults to 5
    :type radius: int, optional
    :return: image with noise filter
    :rtype: ndarray
    """
    sitk_im = sitk.GetImageFromArray(input_image)
    output_image = sitk.Noise(sitk_im, radius=[radius, radius])
    return np.copy(sitk.GetArrayFromImage(output_image))


def moving_median_sitk(input_image, radius=5):
    """Applies a median filter with SimpleITK to an image. 

    :param input_image: image
    :type input_image: ndarray
    :param radius: [description], defaults to 5
    :type radius: int, optional
    :return: image with applied median filter
    :rtype: ndarray
    """
    sitk_im = sitk.GetImageFromArray(input_image)
    output_image = sitk.Median(sitk_im, radius=[radius, radius])
    return np.copy(sitk.GetArrayFromImage(output_image))


def moving_filters_sitk(input_image, radius=5, features=None):
    """Applies itk features to image

    :param input_image: image
    :type input_image: ndarray
    :param radius: radius, defaults to 5 
    :type radius: int, optional
    :param features: features to apply ('sigma', 'mean', 'std', 'median'), defaults to None
    :type features: Collection of features, optional
    :return: None when no features given, all features when no image/features = None , else the image with apllied features
    :rtype: None, [features], ndarray
    """
    feat_all = ['sigma', 'mean', 'std', 'median']
    if input_image is None:
        return feat_all
    if features is None:
        features = feat_all
    if not features:
        return None
    sitk_im = sitk.GetImageFromArray(input_image)
    output_image = {}
    if 'sigma' in features:
        output_image['sigma'] = np.copy(sitk.GetArrayFromImage(
            sitk.BoxSigma(sitk_im, radius=[radius, radius])))
    if 'mean' in features:
        output_image['mean'] = np.copy(sitk.GetArrayFromImage(
            sitk.BoxMean(sitk_im, radius=[radius, radius])))
    if 'std' in features:
        output_image['std'] = np.copy(sitk.GetArrayFromImage(
            sitk.Noise(sitk_im, radius=[radius, radius])))
    if 'median' in features:
        output_image['median'] = np.copy(sitk.GetArrayFromImage(
            sitk.Median(sitk_im, radius=[radius, radius])))
    return output_image

def skimage_rank(filter, image, *args, **kwargs):
    """wrapper for skimage filter

    :param filter: skimages function to use
    :type filter: object
    :param image: image
    :type image: ndarray
    :return: image
    :rtype: ndarray
    """
    mi = image.min()
    ma = image.max()
    if 'mask' in kwargs:
        mi = image[kwargs['mask']].min()
        ma = image[kwargs['mask']].max()
    im_adj = ((image.astype(np.double) - mi) / (ma - mi)
              * ((2 ** 12) - 1)).astype(np.uint16)
    out_im = filter(im_adj, *args, **kwargs).astype(np.double) / ((2 ** 12) - 1) * (ma - mi) + mi
    if 'mask' in kwargs:
        out_im[np.logical_not(kwargs['mask'])] = 0
    return out_im


def skimage_rank_gradient(filter, image, *args, **kwargs):
    mi = image.min()
    ma = image.max()
    if 'mask' in kwargs:
        mi = image[kwargs['mask']].min()
        ma = image[kwargs['mask']].max()
    im_adj = ((image.astype(np.double) - mi) / (ma - mi)
              * ((2 ** 12) - 1)).astype(np.uint16)
    out_im = filter(im_adj, *args, **kwargs).astype(np.double) / ((2 ** 12) - 1) * (ma - mi)
    if 'mask' in kwargs:
        out_im[np.logical_not(kwargs['mask'])] = 0
    return out_im


def skimage_rank_man(filter, image, mi, ma, *args, **kwargs):
    im_adj = ((image.astype(np.double) - mi) / (ma - mi)
              * ((2 ** 12) - 1)).astype(np.uint16)
    out_im = filter(im_adj, *args, **kwargs).astype(np.double) / ((2 ** 12) - 1) * (ma - mi) + mi
    if 'mask' in kwargs:
        out_im[np.logical_not(kwargs['mask'])] = 0
    return out_im


def skimage_rank_gradient_man(filter, image, mi, ma, *args, **kwargs):
    im_adj = ((image.astype(np.double) - mi) / (ma - mi)
              * ((2 ** 12) - 1)).astype(np.uint16)
    out_im = filter(im_adj, *args, **kwargs).astype(np.double) / ((2 ** 12) - 1) * (ma - mi)
    if 'mask' in kwargs:
        out_im[np.logical_not(kwargs['mask'])] = 0
    return out_im


def skimage_rank_entropy_man(filter, image, mi, ma, *args, **kwargs):
    im_adj = ((image.astype(np.double) - mi) / (ma - mi)
              * ((2 ** 12) - 1)).astype(np.uint16)
    out_im = filter(im_adj, *args, **kwargs).astype(np.double) / 1000.0
    if 'mask' in kwargs:
        out_im[np.logical_not(kwargs['mask'])] = 0
    return out_im