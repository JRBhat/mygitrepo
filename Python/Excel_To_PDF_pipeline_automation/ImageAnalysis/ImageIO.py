import io
from PIL import Image
import numpy as np
import os
import logging
import imageio
from  . import ColorConversion as Cc
import cv2

def __svndata__():
    """
    | $Author: ndrews $
    | $Date: 2021-06-09 12:26:18 +0200 (Mi., 09 Jun 2021) $
    | $Rev: 12309 $
    | $URL: http://sw-server:8090/svn/ImageProcessingLibrary/Python/proDERM_ImageAnalysisLibrary/ImageAnalysis/ImageIO.py $
    | $Id: ImageIO.py 12309 2021-06-09 10:26:18Z ndrews $
    """
    # only for documentation purpose
    return {
        'author': "$Author: ndrews $".replace('$', '').replace('Author:', '').strip(),
        'date': "$Date: 2021-06-09 12:26:18 +0200 (Mi., 09 Jun 2021) $".replace('$', '').replace('Date:', '').strip(),
        'rev': "$Rev: 12309 $".replace('$', '').replace('Rev:', '').strip(),
        'id': "$Id: ImageIO.py 12309 2021-06-09 10:26:18Z ndrews $".replace('$', '').replace('Id:', '').strip(),
        'url': "$URL: http://sw-server:8090/svn/ImageProcessingLibrary/Python/proDERM_ImageAnalysisLibrary/ImageAnalysis/ImageIO.py $".replace('$', '').replace('URL:', '').strip()
    } 

READ_IMAGES = {}
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.info("Importing %s, Vs: %s" % (__name__, __svndata__()['id']))

def readRGBImage(filename, keep=False):
    """Reads image and converts to RGB

    :param filename: file to convert
    :type filename: str
    :param keep: keep file, defaults to False
    :type keep: bool, optional
    :return: image
    :rtype: ndarray
    """
    img = readImage(filename, keep)
    if len(img.shape) == 2:
        return np.dstack([img, img, img])
    else:
        return img

def readImage(filename, keep=False):
    """reades an image as numpy array

    :param filename: image location
    :type filename: string
    :param keep: keep image, defaults to False
    :type keep: bool, optional
    :raises IOError: File not found/error while reading
    :return: image as array
    :rtype: Numpy Array
    """
    if keep:
        global READ_IMAGES
        if filename in READ_IMAGES:
            return READ_IMAGES[filename]
    if not os.path.exists(filename):
        LOGGER.error("Could not read File '%s', File does not exist" %
                     filename)  # log error
        raise IOError
    try:
        im = imageio.imread(filename)
        LOGGER.info("Read Image %s using 'freeimage'" % filename)
    except:
        try:
            im = Image.open(filename)
            LOGGER.info("Read Image %s using 'pil'" % filename)
        except:
            try:
                frames = imageio.mimread(filename)
                im = frames[len(frames)//2]
                LOGGER.info(
                    "Read a video and took image at position " + str(len(frames)//2) + "   %s using 'imageio.mimread'" % filename)
            except Exception as inst:
                LOGGER.error("Could not read File '%s', Exception: %s" %
                             (filename, inst))  # log error
                raise
    return np.array(im)

def readGrayscaleImage(filename, keep=False):
    """ Reads image and converts to grayscale (weighted colors)

    :param filename: file to convert
    :type filename: str
    :param keep: keep image, defaults to False
    :type keep: bool, optional
    :return: image in grayscale
    :rtype: ndarray
    """
    return rgb_to_gray_image(readImage(filename, keep).astype(np.float32))


def rgb_to_gray_image(img):
    """transforms rgb to gray image

    :param img: image
    :type img: ndarray
    :return: image in rgb
    :rtype: ndarray
    """
    intype = img.dtype
    if len(img.shape) == 3:
        img = img.astype(np.float32)
        return (0.21 * img[:, :, 0] + 0.72 * img[:, :, 1] + 0.07 * img[:, :, 2]).astype(intype)
    else:
        return img

def writeImage(data, filename):
    """write an image

    :param data: data to write
    :type data: ndarray
    :param filename: location to save image
    :type filename: string
    :return: filename or false if fails
    :rtype: string or false
    """
    if data.max() > 255:
        try:
            imageio.imsave(filename, data, 'freeimage')
            LOGGER.info("Write Image %s using 'freeimage'" % filename)
        except Exception as inst:
            LOGGER.error("Could not write File '%s', Exception: %s" %
                         (filename, inst))  # log error
            raise
        return filename
    else:
        try:
            try:
                Image.fromarray(data).save(filename)
                LOGGER.info("Write Image %s using 'pil'" % filename)
            except:
                try:
                    imageio.imsave(filename, data)
                    LOGGER.info(
                        "write Image %s using 'imageio.imread'" % filename)
                except:
                    cv2.imwrite(filename, data)
                    LOGGER.info("Write Image %s using 'openCV'" % filename)
        except Exception as inst:
            LOGGER.error("Could not write File '%s', Exception: %s" %
                         (filename, inst))  # log error
            raise
        return filename