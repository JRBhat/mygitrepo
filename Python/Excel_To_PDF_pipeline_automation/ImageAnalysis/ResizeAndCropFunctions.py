import cv2
import numpy as np
from scipy.misc import imresize
from past.utils import old_div

def resize_to_fit_bewertunsmonitor(im_out, x=2560.0, y=1600.0, mode=None, enlarge=False):
    """resizes tge image to fit the bewertungsmonnitor

    :param im_out: image to resize
    :type im_out: ndarray
    :param x: x width to fit, defaults to 2560.0
    :type x: float, optional
    :param y: y width to fit, defaults to 1600.0
    :type y: float, optional
    :param mode: resize mode, defaults to None
    :type mode: string, optional
    :param enlarge: enlarge image, defaults to False
    :type enlarge: bool, optional
    :return: the resized image
    :rtype: ndarray
    """

    resize_factor = min(
        float(x) / float(im_out.shape[1]), float(y) / float(im_out.shape[0]))
    # print resize_factor
    if resize_factor < 1 or enlarge:
        im_out = imresize(im_out, resize_factor, mode=mode)
    return im_out

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """resizes an image

    :param image: image to resize
    :type image: ndarray
    :param width: resize width, defaults to None
    :type width: integer, optional
    :param height: resize height, defaults to None
    :type height: integer, optional
    :param inter: how to calculate pixel values (INTER_NEAREST,INTER_LINEAR,INTER_AREA,INTER_CUBIC,INTER_LANCZOS4), defaults to cv2.INTER_AREA
    :type inter: INTER_NEAREST,INTER_LINEAR,INTER_AREA,INTER_CUBIC,INTER_LANCZOS4, optional
    :return: resized image
    :rtype: ndarray
    """

    dim = None
    # initialize the dimensions of the image to be resized and
    # grab the image size
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

def cropBBox(image, bbox, border=0, additional_crops=None):
    """crop image to bounding box (PIL/NUMPY)

    :param image: image
    :type image: numpy array
    :param bbox: bounding box
    :type bbox: list
    :param border: add pxel border
    :type border: int(0)
    :return: cropped image
    :rtype: numpy array
    """
    bbox = np.array(bbox) + [-border, -border, border, border]
    bbox[:2] = np.maximum(bbox[:2], [0, 0])
    bbox[2:] = np.minimum(bbox[2:], image.shape[:2][::-1])
    if additional_crops is not None:
        out_dat = []
        for dat in additional_crops:
            if len(dat.shape) == 2:
                out_dat.append(dat[bbox[1]: bbox[3], bbox[0]:bbox[2]])
            elif len(dat.shape) == 3:
                out_dat.append(dat[bbox[1]: bbox[3], bbox[0]:bbox[2], :])
        if len(image.shape) == 2:
            return image[bbox[1]: bbox[3], bbox[0]:bbox[2]], bbox, out_dat
        elif len(image.shape) == 3:
            return image[bbox[1]: bbox[3], bbox[0]:bbox[2], :], bbox, out_dat
    else:
        if len(image.shape) == 2:
            return image[bbox[1]: bbox[3], bbox[0]:bbox[2]]
        elif len(image.shape) == 3:
            return image[bbox[1]: bbox[3], bbox[0]:bbox[2], :]

def cropBBoxITK(image, bbox, border=0, square=False):
    """crop image to bounding box (ITK)

    :param image: image
    :type image: numpy array
    :param bbox: bounding box [idx, idx, size, size]
    :type bbox: list
    :param border: add pxel border
    :type border: int(0)
    :param square: enlarge to have square region
    :type square: type(False)
    :return: cropped image,bounding box
    :rtype: numpy array, ndarray
    """
    idx = np.array(bbox)[:2]
    size = np.array(bbox)[2:]
    if square:
        newSize = np.array([size.max(), size.max()])
        diffSize = newSize - size
        idx -= old_div(diffSize, 2)
        size = newSize
    idx -= [border, border]
    size += 2 * np.array([border, border])
    bbox = np.array([idx, idx + size]).flatten()
    bbox[:2] = np.maximum(bbox[:2], [0, 0])
    bbox[2:] = np.minimum(bbox[2:], image.shape[:2][::-1])
    if len(image.shape) == 2:
        return image[bbox[1]: bbox[3], bbox[0]:bbox[2]], bbox
    elif len(image.shape) == 3:
        return image[bbox[1]: bbox[3], bbox[0]:bbox[2], :], bbox
