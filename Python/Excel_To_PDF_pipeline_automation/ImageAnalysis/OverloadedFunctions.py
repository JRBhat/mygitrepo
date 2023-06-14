from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from future import standard_library
from PIL import Image, ImageDraw, ImageFont, ImageCms
import imageio
from skimage.measure import ransac
from skimage.transform import AffineTransform, warp_coords, ProjectiveTransform, SimilarityTransform
from past.utils import old_div
from . import Util
from .ImageIO import readRGBImage, rgb_to_gray_image
from .FindTemplateFunctions import find_template_contour
from .ResizeAndCropFunctions import cropBBoxITK
from  . import ColorConversion

import cv2
import os.path
import skimage
import numpy as np
import logging
import tempfile
import sys
from PySide2.QtGui import QImage
import pickle
import io


MODULES = {}
#MODULES['scipy.ndimage'] = None
#MODULES['itk'] = None
READ_IMAGES = {}

def __svndata__():
    """
    | $Author: ndrews $
    | $Date: 2021-06-30 13:34:46 +0200 (Mi., 30 Jun 2021) $
    | $Rev: 12394 $
    | $URL: http://sw-server:8090/svn/ImageProcessingLibrary/Python/proDERM_ImageAnalysisLibrary/ImageAnalysis/OverloadedFunctions.py $
    | $Id: OverloadedFunctions.py 12394 2021-06-30 11:34:46Z ndrews $
    """
    # only for documentation purpose
    return {
        'author': "$Author: ndrews $".replace('$', '').replace('Author:', '').strip(),
        'date': "$Date: 2021-06-30 13:34:46 +0200 (Mi., 30 Jun 2021) $".replace('$', '').replace('Date:', '').strip(),
        'rev': "$Rev: 12394 $".replace('$', '').replace('Rev:', '').strip(),
        'id': "$Id: OverloadedFunctions.py 12394 2021-06-30 11:34:46Z ndrews $".replace('$', '').replace('Id:', '').strip(),
        'url': "$URL: http://sw-server:8090/svn/ImageProcessingLibrary/Python/proDERM_ImageAnalysisLibrary/ImageAnalysis/OverloadedFunctions.py $".replace('$', '').replace('URL:', '').strip()
    }


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.info("Importing %s, Vs: %s" % (__name__, __svndata__()['id']))

def do_import(module, as_name=None):
    global MODULES
    if as_name is None:
        as_name = module
    if as_name not in MODULES or MODULES[as_name] is None:
        mod_list = module.split('.')
        loaded_mod = __import__(module)
        for k in range(1, len(mod_list)):
            loaded_mod = getattr(loaded_mod, mod_list[k])
        MODULES[as_name] = loaded_mod
    return MODULES[as_name]


class ImageClass(object):
    """ Basic Image class 
    :param filename: image filename (existing or will be created as 8bit-srgb)
    :type filename: str
    :param from_lab: CIELAB input
    :type from_lab: numpy array(None)
    :param from_srgb_norm: normed srgb input (no gamma, float)
    :type from_srgb_norm: numpy array(None)
    """
    keep_data = True
    """ :param keep_data: keep data in memory
        :type keep_data: bool"""
    calc_mode = 'best'
    """ :param calc_mode: 'best' = CUDA (fallback to numpy)
        :type calc_mode: str"""
    save_data = False
    """ :param save_data: save data to files
        :type save_data: bool"""
    temp_directory = tempfile.gettempdir()
    """ :param temp_directory: if :para:save_data True, save to that directory
        :type temp_directory: str(directory)"""

    def __init__(self, filename, from_lab=None, from_srgb_norm=None, from_hsv=None):
        self._filename = filename
        self._save_base_name = Util.create_filename_from_basefile(
            self._filename, file_ext="", directory=self.temp_directory)
        Util.createDirectory(self.temp_directory)
        if from_lab is not None:
            writeData(from_lab, self._save_base_name + "_lab.dat")
            if self.calc_mode == 'best' or self.calc_mode == 'cuda':
                try:
                    imd = ((ColorConversion.convertToLab().LabTosRGB(
                        from_lab).clip(0, 1)) * (2 ** 16 - 1)).astype(np.uint16)
                    if os.path.exists(self._filename):
                        Util.backupFile(self._filename)
                        LOGGER.warning(
                            "%s exists, do backup and save new" % self._filename)
                    writeImage(imd, self._filename)
                except Exception as inst:
                    LOGGER.warning("Cuda not working")
                    LOGGER.warning(inst)
                    imd = ColorConversion.colormath_conversion_image(
                        from_lab, 'lab', 'srgb', mask=None, illuminant='D65')
                    writeImage(imd, self._filename)
        if from_hsv is not None:
            writeData(from_hsv, self._save_base_name + "_hsv.dat")
            if self.calc_mode == 'best' or self.calc_mode == 'cuda':
                try:
                    imd = ((ColorConversion.convertToLab().HSVTosRGB(
                        from_hsv).clip(0, 1)) * (2 ** 16 - 1)).astype(np.uint16)
                    if os.path.exists(self._filename):
                        Util.backupFile(self._filename)
                        LOGGER.warning(
                            "%s exists, do backup and save new" % self._filename)
                    writeImage(imd, self._filename)
                except Exception as inst:
                    LOGGER.warning("Cuda not working")
                    LOGGER.warning(inst)
                    imd = ColorConversion.colormath_conversion_image(
                        from_hsv, 'hsv', 'srgb', mask=None, illuminant='D65')
                    writeImage(imd, self._filename)
        if from_srgb_norm is not None:
            if self.calc_mode == 'best' or self.calc_mode == 'cuda':
                imd = ((ColorConversion.srgbGammaNP(from_srgb_norm).clip(
                    0, 1)) * (2 ** 16 - 1)).astype(np.uint16)
                writeImage(imd[::-1, ::-1, :], self._filename)
        self._np_data = None
        self._qimage = None
        self._image_type = None
        self._free_image_data = None
        self._hsv = None
        self._rgb = None
        self._lab = None
        self._xyz = None
        self._shape = None
        self.has_data = False
        self._calc_mask = None
        self._calc_mask_bounding_box = None

    @property
    def free_image(self):
        """ smc.freeimage Image class """
        if not self.keep_data:
            self._image_type = ColorConversion.get_colorprofile(self.filename) #TODO nachfragen eventuell nicht nötig doppelte öffnung
            im = Image.open(self.filename)
            self._shape = [im.size[1], im.size[0], {
                3 * 16: 3, 3 * 8: 3, 4 * 8: 4, 4 * 16: 4, 16: 1, 8: 1, 2: 1}[self._image_type[2]]]
            return im
        if self._free_image_data is None:
            print("Freeimage: Really Read %s" % self.filename)
            self._free_image_data = Image.open(self.filename)  # aa
            self._image_type = ColorConversion.get_colorprofile(self.filename) #TODO nachfragen eventuell nicht nötig doppelte öffnung
            self._shape = [self._free_image_data.size[1], self._free_image_data.size[0], {
                3 * 16: 3, 3 * 8: 3, 4 * 8: 4, 4 * 16: 4, 16: 1, 8: 1, 2: 1}[self._image_type[2]]]
            self.has_data = True
        return self._free_image_data

    @property
    def filename(self):
        """ filename """
        return self._filename

    @filename.setter
    def filename(self, filename):
        if self._filename != filename:
            self.clear_memory()
            self._filename = filename

    @free_image.setter
    def free_image(self, *args, **kwargs):
        raise AttributeError

    @property
  #  @numpy_testing_almost_equal()
    def np_data(self):
        if not self.keep_data:
            im = self.convert_rgb(readRGBImage(self.filename))
            self._shape = im.shape
            return im
        else:
            if self._np_data is None:
                print("ImageLibrary: Really Read %s" % self.filename)
                self._np_data = self.convert_rgb(readRGBImage(self.filename))
                self._shape = self._np_data.shape
                self.has_data = True
            return self._np_data        

    @np_data.setter
    def np_data(self, *args, **kwargs):
        raise AttributeError

    def convert_rgb(self, data):
        ttype = data.dtype
        if np.issubdtype(ttype, np.int) or (ttype.type is np.uint8) or (ttype.type is np.uint16):
            data = old_div(data.astype(np.float), np.iinfo(ttype).max) 
        return data


    @property
    #@numpy_testing_almost_equal()
    def srgb_norm(self):
        if self.colordepth == 48:
            np_image = readRGBImage(self._filename)
        else:
            np_image = self.convert_rgb(self.np_data)
        LOGGER.debug("Colorspace name: %s" % self.colorspace_name)
        if self.colorspace_name == 'adobe':
            if self.calc_mode == 'best' or self.calc_mode == 'cuda':
                try:
                    return ColorConversion.convertToLab().adobeRGBToNormSRGB(np_image).clip(0, 1)
                except Exception as inst:
                    LOGGER.warning("Cuda not working")
                    LOGGER.warning(inst)
            return ColorConversion.adobeToNormSRGBNP(np_image).clip(0, 1)
        elif self.colorspace_name == 'srgb':
            return ColorConversion.srgbGammaRemoveNP(self.convert_rgb(self.np_data).astype(np.double) / 255.0)
        else:
            print("not supported")
            sys.exit(-1)

    @srgb_norm.setter
    def srgb_norm(self, *args, **kwargs):
        raise AttributeError

    @property
    def greyscale(self):
        return rgb_to_gray_image(self.np_data)

    @greyscale.setter
    def greyscale(self, *args, **kwargs):
        raise AttributeError

    @property
    def qimage(self):
        if not self.keep_data:
            return QImage(self.filename)
        if self._qimage is None:
            print("qimage: Really Read %s" % self.filename)
            self._qimage = QImage(self.filename)
            self.has_data = True
        return self._qimage

    @qimage.setter
    def qimage(self, *args, **kwargs):
        raise AttributeError

    def clear_memory(self):
        self._qimage = None
        self._image_type = None
        self._free_image_data = None
        self._hsv = None
        self._rgb = None
        self._lab = None
        self._xyz = None
        self._shape = None
        self.has_data = False

    @property
    #@numpy_testing_almost_equal()
    def rgb48bpp(self):
        if ColorConversion.mode_to_bpp(self.free_image.mode) == 48:
            return self.np_data
        if ColorConversion.mode_to_bpp(self.free_image.mode) == (8 * 3):
            return self.np_data.astype(np.uint16) * 255
        return (self.convert_rgb(self.np_data).astype(np.uint16) * 255)

    @rgb48bpp.setter
    def rgb48bpp(self, *args, **kwargs):
        raise AttributeError

    @property
   # @numpy_testing_almost_equal()
    def rgb24bpp(self):
        if self.colordepth == 48:
            return _convert_uint16_to_uint8(self.np_data)
        if self.colordepth == (8 * 3):
            return self.np_data
        return self.convert_rgb

    @rgb24bpp.setter
    def rgb24bpp(self, *args, **kwargs):
        raise AttributeError

    @property
    #@numpy_testing_almost_equal()
    def srgb24(self):
        if self.save_data:
            try:
                return readRGBImage(self._save_base_name + "_srgb.png")
            except:
                pass
        np_image = self.np_data
        if self.colorspace_name == 'adobe':
            if self.calc_mode == 'best' or self.calc_mode == 'cuda':
                try:
                    imd = ((ColorConversion.convertToLab().adobeRGBTosRGB(
                        np_image).clip(0, 1)) * 255).astype(np.uint8)
                    if self.save_data:
                        writeImage(imd, self._save_base_name + "_srgb.png")
                    return imd
                except Exception as inst:
                    LOGGER.warning("Cuda not working")
                    LOGGER.warning(inst)
            imd = ((ColorConversion.adobeTosRGBNP(np_image).clip(0, 1))
                   * 255).astype(np.uint8)
            if self.save_data:
                writeImage(imd, self._save_base_name + "_srgb.png")
            return imd
        elif self.colorspace_name == 'srgb':
            imd = ((np_image.clip(0, 1))
                   * 255).astype(np.uint8)
            if self.save_data:
                writeImage(imd, self._save_base_name + "_srgb.png")
            return imd
        else:
            LOGGER.error("not supported")
            sys.exit(-1)
        return np_image

    @srgb24.setter
    def srgb24(self, *args, **kwargs):
        raise AttributeError

    @property
    def lab(self):
        if self.save_data:
            try:
                return readData(self._save_base_name + "_lab.dat")
            except Exception as inst:
                pass
        if self.colordepth == 48:
            np_image = self.np_data
        else:
            np_image = self.convert_rgb(self.np_data)
        if self._calc_mask is not None:
            np_image = np_image[self._calc_mask_bounding_box]
        LOGGER.debug("Colorspace name: %s" % self.colorspace_name)
        if self.colorspace_name == 'adobe':
            if self.calc_mode == 'best' or self.calc_mode == 'cuda':
                try:
                    LOGGER.debug("start adobe-lab in ImageClass")
                    imd = ColorConversion.convertToLab().adobeRGBToLab(np_image)
                    if self._calc_mask is not None:
                        imdd = np.zeros(self.shape, imd.dtype)
                        imdd[self._calc_mask_bounding_box] = imd
                        imd = imdd
                    if self.save_data:
                        writeData(imd, self._save_base_name + "_lab.dat")
                    return imd
                except Exception as inst:
                    LOGGER.warning("Cuda not working")
                    LOGGER.warning(inst)
            imd = ColorConversion.adobe_to_lab(np_image)
            if self._calc_mask is not None:
                imdd = np.zeros(self.shape, imd.dtype)
                imdd[self._calc_mask_bounding_box] = imd
                imd = imdd
            if self.save_data:
                writeData(imd, self._save_base_name + "_lab.dat")
            return imd
        elif self.colorspace_name == 'srgb':
            if self.calc_mode == 'best' or self.calc_mode == 'cuda':
                try:
                    imd = ColorConversion.convertToLab().sRGBToLab(np_image)
                    if self._calc_mask is not None:
                        imdd = np.zeros(self.shape, imd.dtype)
                        imdd[self._calc_mask_bounding_box] = imd
                        imd = imdd
                    if self.save_data:
                        writeData(imd, self._save_base_name + "_lab.dat")
                    return imd
                except Exception as inst:
                    LOGGER.warning("Cuda not working")
                    LOGGER.warning(inst)
            imd = ColorConversion.sRGBToLabNP(np_image)
            if self._calc_mask is not None:
                imdd = np.zeros(self.shape, imd.dtype)
                imdd[self._calc_mask_bounding_box] = imd
                imd = imdd
            if self.save_data:
                writeData(imd, self._save_base_name + "_lab.dat")
            return imd
        else:
            LOGGER.error("not supported")
            sys.exit(-1)

    @lab.setter
    def lab(self, *args, **kwargs):
        raise AttributeError

    @property
    def hsv(self):
        if self.save_data:
            try:
                return readData(self._save_base_name + "_hsv.dat")
            except Exception as inst:
                pass
        if self.colordepth == 48:
            np_image = self.np_data
        else:
            np_image = self.convert_rgb(self.np_data)
        if self._calc_mask is not None:
            np_image = np_image[self._calc_mask_bounding_box]
        LOGGER.debug("Colorspace name: %s" % self.colorspace_name)
        if self.colorspace_name == 'adobe':
            if self.calc_mode == 'best' or self.calc_mode == 'cuda':
                try:
                    imd = ColorConversion.convertToLab().adobeRGBToHSV(np_image)
                    if self.save_data:
                        writeData(imd, self._save_base_name + "_hsv.dat")
                    return imd
                except Exception as inst:
                    LOGGER.warning("Cuda not working")
                    LOGGER.warning(inst)
            imd = ColorConversion.adobeRGBToHSVNP(np_image)
            if self.save_data:
                writeData(imd, self._save_base_name + "_hsv.dat")
            return imd
        elif self.colorspace_name == 'srgb':
            if self.calc_mode == 'best' or self.calc_mode == 'cuda':
                try:
                    imd = ColorConversion.convertToLab().sRGBToHSV(np_image)
                    if self.save_data:
                        writeData(imd, self._save_base_name + "_hsv.dat")
                    return imd
                except Exception as inst:
                    LOGGER.warning("Cuda not working")
                    LOGGER.warning(inst)
            imd = ColorConversion.sRGBToHSVNP(np_image)
            if self.save_data:
                writeData(imd, self._save_base_name + "_hsv.dat")
            return imd
        else:
            LOGGER.error("not supported")
            sys.exit(-1)

    @hsv.setter
    def hsv(self, *args, **kwargs):
        raise AttributeError

    @property
    def shape(self):
        if self._shape:
            return self._shape
        try:
            im = imageio.imread(self.filename)
            self._shape = [im.size[1], im.size[0], len(im.mode)]
            return self._shape
        except:
            return self.np_data.shape

    @shape.setter
    def shape(self, *args, **kwargs):
        raise AttributeError

    @property
    def image_type(self):
        if not self._image_type:
            self.free_image
        return self._image_type

    @image_type.setter
    def image_type(self, *args, **kwargs):
        raise AttributeError

    @property
    def colorspace_name(self):
        return self.image_type[0]

    @colorspace_name.setter
    def colorspace_name(self, *args, **kwargs):
        raise AttributeError

    @property
    def colorspace_type(self):
        return self.image_type[1]

    @colorspace_type.setter
    def colorspace_type(self, *args, **kwargs):
        raise AttributeError

    @property
    def colordepth(self):
        return self.image_type[2]

    @colordepth.setter
    def colordepth(self, *args, **kwargs):
        raise AttributeError

    @property
    def np_dtype(self):
        return {8: np.uint8, 16: np.uint16, 3 * 8: np.uint8, 4 * 8: np.uint8, 3 * 16: np.uint16, 4 * 16: np.uint16}[self.image_type[2]]

    @np_dtype.setter
    def np_dtype(self, *args, **kwargs):
        raise AttributeError

    @property
    def calc_mask(self):
        return self._calc_mask

    @calc_mask.setter
    def calc_mask(self, mask):
        from scipy.ndimage import find_objects
        self._calc_mask = mask
        self._calc_mask_bounding_box = find_objects(mask.astype(np.uint8))[0]
        if self.get_cropped_mask().sum() != self._calc_mask.sum():
            LOGGER.critical("Error in calc_mask")
            sys.exit(-1)

    def get_cropped_mask(self):
        if self._calc_mask is not None:
            return self.calc_mask[self._calc_mask_bounding_box]

    def _general_color(self, name, adobe_col_function_cuda, adobe_col_function_numpy, srgb_col_function_cuda, srgb_col_function_numpy):
        meorize_filename = self._save_base_name + "_%s.dat" % name
        if self.save_data:
            try:
                return readData(meorize_filename)
            except Exception as inst:
                pass
        if self.colordepth == 48:
            np_image = self.np_data
        else:
            np_image = self.convert_rgb(self.np_data)
        if self._calc_mask is not None:
            np_image = np_image[self._calc_mask_bounding_box]
        LOGGER.debug("Colorspace name: %s" % self.colorspace_name)
        if self.colorspace_name == 'adobe':
            if self.calc_mode == 'best' or self.calc_mode == 'cuda':
                try:
                    LOGGER.debug("start adobe-lab in ImageClass")
                    imd = adobe_col_function_cuda(np_image)
                    if self._calc_mask is not None:
                        imdd = np.zeros(self.shape, imd.dtype)
                        imdd[self._calc_mask_bounding_box] = imd
                        imd = imdd
                    if self.save_data:
                        writeData(imd, meorize_filename)
                    return imd
                except Exception as inst:
                    LOGGER.warning("Cuda not working")
                    LOGGER.warning(inst)
            imd = adobe_col_function_numpy(np_image)
            if self._calc_mask is not None:
                imdd = np.zeros(self.shape, imd.dtype)
                imdd[self._calc_mask_bounding_box] = imd
                imd = imdd
            if self.save_data:
                writeData(imd, meorize_filename)
            return imd
        elif self.colorspace_name == 'srgb':
            if self.calc_mode == 'best' or self.calc_mode == 'cuda':
                try:
                    imd = srgb_col_function_cuda(np_image)
                    if self._calc_mask is not None:
                        imdd = np.zeros(self.shape, imd.dtype)
                        imdd[self._calc_mask_bounding_box] = imd
                        imd = imdd
                    if self.save_data:
                        writeData(imd, meorize_filename)
                    return imd
                except Exception as inst:
                    LOGGER.warning("Cuda not working")
                    LOGGER.warning(inst)
            imd = srgb_col_function_numpy(np_image)
            if self._calc_mask is not None:
                imdd = np.zeros(self.shape, imd.dtype)
                imdd[self._calc_mask_bounding_box] = imd
                imd = imdd
            if self.save_data:
                writeData(imd, meorize_filename)
            return imd
        else:
            LOGGER.error("not supported")
            sys.exit(-1)

    @property
    def xyz(self):
        try:
            cvz = ColorConversion.convertToLab()
        except:
            self.calc_mode = "numpy"
        return self._general_color('xyz', cvz.adobeRGBToXYZ, ColorConversion.adobeToXYZNP, cvz.sRGBToXYZ, ColorConversion.sRGBToXYZNP)

    @xyz.setter
    def xyz(self, *args, **kwargs):
        raise AttributeError


class ImagePyramideClass(list):
    keep_data = True
    calc_mode = 'best'
    save_data = False
    temp_directory = "C:\\Experimente\\Image_Class_Pyramide_TEMP"
    do_upscale = False

    def __init__(self, filename, depth=-1):
        Util.createDirectory(self.temp_directory)
        self._filename = filename
        ImageClass.keep_data = self.keep_data
        ImageClass.save_data = self.save_data
        ImageClass.temp_directory = self.temp_directory
        image_class_pyramide = [ImageClass(filename)]
        self._save_base_name = Util.create_filename_from_basefile(
            self._filename, file_ext="", directory=self.temp_directory)
        self._max_depth = np.floor(np.log2(image_class_pyramide[0].shape[:2]))
        if depth > 0:
            self._pyramide_depth = int(min(depth, self._max_depth.min()))
        else:
            self._pyramide_depth = int(self._max_depth.min())
        if self.do_upscale:
            from skimage.transform import resize
            image_class_pyramide.extend([ImageClass(
                self._save_base_name + "_pyramide_upscale_%d.png" % rg) for rg in range(1, self._pyramide_depth)])
        else:
            image_class_pyramide.extend([ImageClass(
                self._save_base_name + "_pyramide_%d.png" % rg) for rg in range(1, self._pyramide_depth)])
        missing_files_idx = -1
        try:
            missing_files_idx = [os.path.exists(
                x.filename) for x in image_class_pyramide].index(False)
        except ValueError:
            pass
        if missing_files_idx > -1:
            im = image_class_pyramide[missing_files_idx - 1].np_data
            for x in image_class_pyramide[missing_files_idx:]:
                # need to convert so that there is no difference between saved<>created data
                im = skimage.transform.pyramid_reduce(im)
                if self.do_upscale:
                    im = resize(im, image_class_pyramide[0].shape[:2], order=1)
                im = np.round(im * ((2 ** 8) - 1)).astype(np.uint8)
                writeImage(im, x.filename)
        # print 1
        super(ImagePyramideClass, self).__init__(image_class_pyramide)

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, *args, **kwargs):
        raise AttributeError

    @property
    def shape(self):
        return self[0].shape

    @shape.setter
    def shape(self, *args, **kwargs):
        raise AttributeError

    @property
    def image_type(self):
        return self[0].image_type

    @image_type.setter
    def image_type(self, *args, **kwargs):
        raise AttributeError

    @property
    def colorspace_name(self):
        return self[0].colorspace_name

    @colorspace_name.setter
    def colorspace_name(self, *args, **kwargs):
        raise AttributeError

    @property
    def colorspace_type(self):
        return self[0].colorspace_type

    @colorspace_type.setter
    def colorspace_type(self, *args, **kwargs):
        raise AttributeError

    @property
    def colordepth(self):
        return self[0].colordepth

    @colordepth.setter
    def colordepth(self, *args, **kwargs):
        raise AttributeError


def find_template(image, template, cropimage=None, showdata=False):
    """find a template in a image

    :param image: image to search in
    :type image: Image
    :param template: template to search for
    :type template: ndarray
    :param cropimage: cropped image, defaults to None
    :type cropimage: Image, optional
    :param showdata: show data (pylab.show()), defaults to False
    :type showdata: bool, optional
    :return: template , bounding box
    :rtype: ndarray, tupel
    """
    assert image.dtype == template.dtype
    (h, w) = template.shape[:2]
    method = cv2.TM_CCORR_NORMED
    value_choice = [1, 3]
    ttype = image.dtype
    if np.issubdtype(ttype, np.int) and not ((ttype.type is np.uint8) or (ttype.type is np.uint16)):
        LOGGER.warning("Imagetype is integer but not uint8 or uint16!")
    if ttype.type is np.uint16:
        image = _convert_uint16_to_uint8(image)
        template = _convert_uint16_to_uint8(template)
    out = cv2.normalize(cv2.matchTemplate(image, template, method))
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(out)
    minMaxValues = cv2.minMaxLoc(out)
    if cropimage is None:
        cropimage = image
    bd = 0  # 100
    testImAuss, bbox_crop = cropBBoxITK(
        image, [minMaxValues[value_choice[-1]][0], minMaxValues[value_choice[-1]][1], w, h], border=bd)
    if showdata:
        import pylab
        pylab.subplot(222)
        imgplot = pylab.imshow(out)
        imgplot.set_interpolation('nearest')
        pylab.scatter(maxLoc[0], maxLoc[1], c='r', s=40)
        pylab.scatter(minLoc[0], minLoc[1], c='g', s=40)
        pylab.subplot(221)
        imgplot = pylab.imshow(image)
        imgplot.set_interpolation('nearest')
        pylab.scatter(maxLoc[0], maxLoc[1], c='r', s=40)
        pylab.scatter(minLoc[0], minLoc[1], c='g', s=40)
        pylab.subplot(223)
        imgplot = pylab.imshow(template)
        imgplot.set_interpolation('nearest')
        pylab.subplot(224)
        imgplot = pylab.imshow(testImAuss)
        imgplot.set_interpolation('nearest')
        pylab.show()
    return testImAuss, (minMaxValues[value_choice[-1]][1], (minMaxValues[value_choice[-1]][1] + h), minMaxValues[value_choice[-1]][0], minMaxValues[value_choice[-1]][0] + w)


def find_template_for_ransac(image, template, mode=cv2.TM_SQDIFF_NORMED, best_value_choice=[0, 2]):
    """do opencv template matching
    :param image: image to search
    :type image: numpy.array(uint8 or uint16)
    :param template: template image
    :type template: numpy.array(uint8 or uint16)
    :param mode: opencv template mode (see opencv docu)
    :type mode: opencv constant(cv2.TM_SQDIFF_NORMED)
    :param best_value_choice: [0,2] for minimisation metrc, [1,3] otherwise (see cv2.minmaxLoc)
    :type best_value_choice: list([0,2]
    :return: metric image, coordinates
    :rtype: numpy.array(float), [x0,x1,y0,y1]
    """
    assert(image.dtype == template.dtype)
    (h, w) = template.shape[:2]
    method = mode
    value_choice = best_value_choice
    ttype = image.dtype
    if np.issubdtype(ttype, np.int) and not ((ttype.type is np.uint8) or (ttype.type is np.uint16)):
        LOGGER.warning("Imagetype is integer but not uint8 or uint16!")
    if ttype.type is np.uint16:
        image = _convert_uint16_to_uint8(image)
        template = _convert_uint16_to_uint8(template) 
    # cv2.TM_SQDIFF)) # or CV_TM_CCORR_NORMED))
    out = cv2.matchTemplate(image, template, method)
    range_min_mx = [out.min(), out.max()]
    range_best = cv2.matchTemplate(template, template, method)
    minMaxValues = cv2.minMaxLoc(out)
    if abs(range_best - range_min_mx[0]) < abs(range_best - range_min_mx[1]):
        # , alpha=range_best, beta=range_min_mx[1])
        out = 1.0 - (old_div((out - range_best),
                     (range_min_mx[1] - range_best))).clip(0, 1)
    else:
        # cv2.normalize(out, alpha=range_min_mx[0], beta=range_best)
        out = (
            old_div((out - range_min_mx[0]), (range_best - range_min_mx[0]))).clip(0, 1)
    return out, minMaxValues[value_choice[-1]], (minMaxValues[value_choice[-1]][1], (minMaxValues[value_choice[-1]][1] + h), minMaxValues[value_choice[-1]][0], minMaxValues[value_choice[-1]][0] + w)

def _convert_uint16_to_uint8(data):
    """Convert images to uint8

    :param data: imgae to convert
    :type data: ndarray
    :return: image in uint8
    :rtype: ndarray
    """
    return (old_div(data, 255)).astype(np.uint8)

def find_template_transform(input_image, contour_list_fg, contour_list_bg, find_image, find_bbox, min_size=200, contours_to_move=None):
    """finds templates

    :param input_image: input image
    :type input_image: nd array
    :param contour_list_fg: contour list foreground
    :type contour_list_fg: list of tupel
    :param contour_list_bg: contour list background
    :type contour_list_bg: list of tupel
    :param find_image: image to search for
    :type find_image: ndarray
    :param find_bbox: bbox to search for
    :type find_bbox: ndarray
    :param min_size: min size, defaults to 100
    :type min_size: int, optional
    :param contours_to_move: list of coordinates to transform according to 'best-fit-transform', defaults to None
    :type contours_to_move: list of list of [x,y](None), optional
    :return: distance array, found contours, found template 
    :rtype: array,list, ndarray
    """
    out_region, cc = find_template_contour(input_image, contour_list_bg, [find_image], [
                                           find_bbox], min_size=min_size, template_bbox=None)
    dst = []
    corr_images = {}
    for (y, x) in contour_list_fg:
        xmin = int(np.max([0, x - old_div(min_size, 2)]))
        xmax = int(np.min([input_image.shape[0], x + old_div(min_size, 2)]))
        ymin = int(np.max([0, y - old_div(min_size, 2)]))
        ymax = int(np.min([input_image.shape[1], y + old_div(min_size, 2)]))
        template = input_image[xmin:xmax, ymin:ymax, :]
        if find_bbox is None:
            find_bbox_loc = [0, 0, find_image.shape[1], find_image.shape[0]]
        elif find_bbox == 'auto':
            x_perc = 0.1 * find_image.shape[0]
            y_perc = 0.1 * find_image.shape[1]
            find_bbox_loc = [max(ymin - y_perc, 0), max(xmin - x_perc, 0), min(
                ymax + y_perc, find_image.shape[1]), min(xmax + x_perc, find_image.shape[0])]
        bbox = [int(xa) for xa in [max(0, find_bbox_loc[1] - (xmax - xmin)), min(find_image.shape[0], find_bbox_loc[3] + (xmax - xmin)),
                                   max(0, find_bbox_loc[0] - (ymax - ymin)), min(find_image.shape[1], find_bbox_loc[2] + (xmax - xmin))]]
        corr_image, best_local, bbox_data = find_template_for_ransac(
            find_image[bbox[0]:bbox[1], bbox[2]:bbox[3], :], template)
        diff_x = - xmin + bbox_data[0] + bbox[0]
        diff_y = - ymin + bbox_data[2] + bbox[2]
        contour_x = x + diff_x
        contour_y = y + diff_y
        dst.append([contour_y, contour_x])
        corr_images[(y, x)] = [corr_image, best_local,
                               diff_x, diff_y, template]
    # ransac
    src = np.array(contour_list_fg)
    dst = np.array(dst)
    transform = SimilarityTransform  # ProjectiveTransform
    model = transform()
    model.estimate(src, dst)
    robust_transformed_dst = model(src)
    distances = np.sqrt(((robust_transformed_dst - dst) ** 2).sum(axis=1))
    real_outliers = np.where(distances > 10)[0]
    value_choice = [0, 2]  # see find_template_for_ransac
    if len(real_outliers) > 0:
        for k in real_outliers:
            new_coord = robust_transformed_dst[k, :]
            diff_best_coord = dst[k, :] - new_coord
            x_local = corr_images[tuple(
                contour_list_fg[k])][1][0] + (new_coord[0] - dst[k, 0])
            y_local = corr_images[tuple(
                contour_list_fg[k])][1][1] + (new_coord[1] - dst[k, 1])
            im = corr_images[tuple(contour_list_fg[k])][0]
            y_local_range = [max(y_local - 150, 0),
                             min(y_local + 150, im.shape[0])]
            x_local_range = [max(x_local - 150, 0),
                             min(x_local + 150, im.shape[1])]
            minMaxValues = cv2.minMaxLoc(
                im[y_local_range[0]:y_local_range[1], x_local_range[0]:x_local_range[1]])
            dst[k, :] = [new_coord[1] + (x_local_range[0] - x_local) + minMaxValues[value_choice[-1]]
                         [1], new_coord[0] + (y_local_range[0] - y_local) + minMaxValues[value_choice[-1]][0]]
    model_robust_nn, inliers = ransac(
        (src, dst), transform, min_samples=4, residual_threshold=3, max_trials=200)
    out_contours = None
    model_nn = transform()
    model_nn.estimate(src, dst)
    if contours_to_move:
        out_contours = [(model_robust_nn(src_c)).tolist()
                        for src_c in contours_to_move]
    return dst.tolist(), out_region[0], out_contours


def find_template_ransac(input_image, contour_list_fg, contour_list_bg, find_image, find_bbox, min_size=200, contours_to_move=None, transform=SimilarityTransform):
    """finds templates with ransac

    :param input_image: input image
    :type input_image: nd array
    :param contour_list_fg: contour list foreground
    :type contour_list_fg: list of tupel
    :param contour_list_bg: contour list background
    :type contour_list_bg: list of tupel
    :param find_image: image to search for
    :type find_image: ndarray
    :param find_bbox: bbox to search for
    :type find_bbox: ndarray
    :param min_size: min size, defaults to 100
    :type min_size: int, optional
    :param contours_to_move: list of coordinates to transform according to 'best-fit-transform', defaults to None
    :type contours_to_move: list of list of [x,y](None), optional
    :param transform: transformation type, defaults to SimilarityTransform
    :type transform: skimage.Transform, optional
    :return: distance array, found template 
    :rtype: array, ndarray
    """
    out_region, _ = find_template_contour(input_image, contour_list_bg, [find_image], [
                                          find_bbox], min_size=min_size, template_bbox=None)
    dst = []
    corr_images = {}
    for (y, x) in contour_list_fg:
        xmin = int(np.max([0, x - old_div(min_size, 2)]))
        xmax = int(np.min([input_image.shape[0], x + old_div(min_size, 2)]))
        ymin = int(np.max([0, y - old_div(min_size, 2)]))
        ymax = int(np.min([input_image.shape[1], y + old_div(min_size, 2)]))
        template = input_image[xmin:xmax, ymin:ymax, :]
        if find_bbox is None:
            find_bbox = [0, 0, find_image.shape[1], find_image.shape[0]]
        elif find_bbox == 'auto':
            x_perc = 0.1 * find_image.shape[0]
            y_perc = 0.1 * find_image.shape[1]
            find_bbox = [max(ymin - y_perc, 0), max(xmin - x_perc, 0), min(
                ymax + y_perc, find_image.shape[1]), min(xmax + x_perc, find_image.shape[0])]
        bbox = [int(xa) for xa in [max(0, find_bbox[1] - (xmax - xmin)), min(find_image.shape[0], find_bbox[3] + (xmax - xmin)),
                                   max(0, find_bbox[0] - (ymax - ymin)), min(find_image.shape[1], find_bbox[2] + (xmax - xmin))]]
        corr_image, best_local, bbox_data = find_template_for_ransac(
            find_image[bbox[0]:bbox[1], bbox[2]:bbox[3], :], template)
        diff_x = - xmin + bbox_data[0] + bbox[0]
        diff_y = - ymin + bbox_data[2] + bbox[2]
        contour_x = x + diff_x
        contour_y = y + diff_y
        dst.append([contour_y, contour_x])
        corr_images[(y, x)] = [corr_image, best_local,
                               diff_x, diff_y, template]
    # ransac
    src = np.array(contour_list_fg)
    dst = np.array(dst)
    model = transform()
    model.estimate(src, dst)
    model_robust, inliers = ransac_transform(
        (src, dst), transform, min_samples=4, residual_threshold=3, max_trials=2000)  # max_trials=2000)
    robust_transformed_dst = model_robust(src)
    distances = np.sqrt(((robust_transformed_dst - dst) ** 2).sum(axis=1))
    real_outliers = np.where(distances > 10)[0]
    value_choice = [1, 3]  # see find_template_for_ransac
    if len(real_outliers) > 0:
        for k in real_outliers:
            new_coord = robust_transformed_dst[k, :]
            diff_best_coord = dst[k, :] - new_coord
            x_local = corr_images[tuple(
                contour_list_fg[k])][1][0] + (new_coord[0] - dst[k, 0])
            y_local = corr_images[tuple(
                contour_list_fg[k])][1][1] + (new_coord[1] - dst[k, 1])
            im = corr_images[tuple(contour_list_fg[k])][0]
            y_local_range = [max(y_local - 150, 0),
                             min(y_local + 150, im.shape[0])]
            x_local_range = [max(x_local - 150, 0),
                             min(x_local + 150, im.shape[1])]
            minMaxValues = cv2.minMaxLoc(
                im[y_local_range[0]:y_local_range[1], x_local_range[0]:x_local_range[1]])
            dst[k, :] = [new_coord[0] + (y_local_range[0] - y_local) + minMaxValues[value_choice[-1]]
                         [0], new_coord[1] + (x_local_range[0] - x_local) + minMaxValues[value_choice[-1]][1]]
    return dst.tolist(), out_region[0]


def ransac_transform(xxx_todo_changeme, transform, weight=None, min_samples=3, max_samples=0, distances_eps=1e-5, residual_threshold=3, max_trials=200):
    """find best fit transform between numbered corrdinate points
    :param src_full: source points coordinates
    :type (src_full: list of [x,y]
    :param dst_full: destination points coordinates
    :type dst_full: list of [x,y]
    :param transform: transform to use see skimage.Transform
    :type transform: skimage.Transform
    :param weight: list of weights how to "pick" subset /(randomised)
    :type weight: list(None)
    :param min_samples: minimum number of points to randomly pick
    :type min_samples: int(3)
    :param max_samples: maximum number of points to randomly pick
    :type max_samples: type(0=all)
    :param distances_eps: stop if movement < distances_eps
    :type distances_eps: float(1e-5)
    :param residual_threshold: count ouliers if distance larger than residual_threshold
    :type residual_threshold: float(3)
    :param max_trials: maximum number of runs
    :type max_trials: int(200)
    :return: best tramnsformation model, outliers
    :rtype: skimage.Transform, list
    """
    (src_full, dst_full) = xxx_todo_changeme
    data_len = max(src_full.shape)
    if weight is None:
        weight = np.ones((data_len,))
    else:
        weight = np.copy(weight)
    weight = old_div(weight, weight.sum())
    if max_samples <= 0:
        max_samples = data_len
    max_samples = min(max_samples, (weight > 0).sum())
    min_dist = np.inf
    best_choice = None
    for run in range(max_trials):
        random_choice_r = np.random.randint(min_samples, max_samples + 1)
        random_choice = np.random.choice(
            data_len, size=random_choice_r, replace=False, p=weight)
        model = transform()
        model.estimate(src_full[random_choice], dst_full[random_choice])
        transformed_dst = model(src_full)
        distances = np.sqrt(((transformed_dst - dst_full) ** 2).sum(axis=1))
        real_outliers = np.where(distances > residual_threshold)[0]
        dist = np.mean(distances)
        if dist < min_dist:
            best_choice = np.copy(random_choice)
            min_dist = dist
        if min(real_outliers.shape) == 0 and dist <= distances_eps:
            break
    best_model = transform()
    best_model.estimate(src_full[best_choice], dst_full[best_choice])
    transformed_dst = best_model(src_full)
    distances = np.sqrt(((transformed_dst - dst_full) ** 2).sum(axis=1))
    real_inliers = np.where(distances <= residual_threshold)[0]
    return best_model, real_inliers


def find_template_ransac_weighted(input_image_list, contour_list_fg_list, find_image, find_bbox, min_size=200, contours_to_move=None, transform_model=SimilarityTransform):
    """do weighted template matching of in all templates in input_image_list:
        "Track all points in contour_list_fg_list, apply "best" transformation and refit points for outliers
    :param input_image_list: templates
    :type input_image_list: list of numpy.array
    :param contour_list_fg_list: list of centre coordinates of templates to track
    :type contour_list_fg_list: list of [x,y]
    :param find_image: image to search
    :type find_image: numpy.array
    :param find_bbox: set ROI where to track
    :type find_bbox: value(None: full image, 'auto' same as input coordinate bounding box, [x0,y0, width, height])
    :param min_size: minimal size of templates
    :type min_size: int(200)
    :param contours_to_move: list of coordinates to transform according to 'best-fit-transform'
    :type contours_to_move: list of list of [x,y](None)
    :return: transformed point set, [], transformed contours_to_move
    :rtype: list of [x,y], ,list of list of [x,y]
    """
    cv2_template_modes = [['sqdiff', cv2.TM_SQDIFF_NORMED, [0, 2]], [
        'ccorr', cv2.TM_CCORR_NORMED, [1, 3]], ['CCOEFF', cv2.TM_CCOEFF_NORMED, [1, 3]]]
    # currently used metrics with min/max value information
    dst_full = []
    src_full = []
    dst = []
    src = []
    weight = []
    data_src = []
    weight_full = []
    corr_images = {}
    for idx, (input_image, contour_list_fg) in enumerate(zip(input_image_list, contour_list_fg_list)):
        # track single point
        for p_idx, (y, x) in enumerate(contour_list_fg):
            # define region
            xmin = int(np.max([0, x - old_div(min_size, 2)]))
            xmax = int(
                np.min([input_image.shape[0], x + old_div(min_size, 2)]))
            ymin = int(np.max([0, y - old_div(min_size, 2)]))
            ymax = int(
                np.min([input_image.shape[1], y + old_div(min_size, 2)]))
            template = input_image[xmin:xmax, ymin:ymax, :]
            if find_bbox is None:
                find_bbox = [0, 0, find_image.shape[1], find_image.shape[0]]
            elif find_bbox == 'auto':
                x_perc = 0.1 * find_image.shape[0]
                y_perc = 0.1 * find_image.shape[1]
                find_bbox = [max(ymin - y_perc, 0), max(xmin - x_perc, 0), min(
                    ymax + y_perc, find_image.shape[1]), min(xmax + x_perc, find_image.shape[0])]
            bbox = [int(xa) for xa in [max(0, find_bbox[1] - (xmax - xmin)), min(find_image.shape[0], find_bbox[3] + (xmax - xmin)),
                                       max(0, find_bbox[0] - (ymax - ymin)), min(find_image.shape[1], find_bbox[2] + (xmax - xmin))]]
            src_local = dict()
            dst_local = dict()
            corr_im = []
            best_local_set = set([])
            # track using all metrics
            for t_mode in cv2_template_modes:
                corr_image, best_local, bbox_data = find_template_for_ransac(
                    find_image[bbox[0]:bbox[1], bbox[2]:bbox[3], :], template, mode=t_mode[1], best_value_choice=t_mode[2])
                diff_x = - xmin + bbox_data[0] + bbox[0]
                diff_y = - ymin + bbox_data[2] + bbox[2]
                contour_x = x + diff_x
                contour_y = y + diff_y
                dst_full.append([contour_y, contour_x])
                src_full.append(contour_list_fg_list[0][p_idx])
                weight_full.append(corr_image[best_local[::-1]])
                data_src.append((idx, t_mode[0], p_idx))
                corr_images[(idx, t_mode[0], p_idx)] = [
                    corr_image, best_local, diff_x, diff_y, template]
                corr_im.append(corr_image)
                src_local[best_local] = contour_list_fg_list[0][p_idx]
                dst_local[best_local] = [contour_y, contour_x]
            # average all "correlation" images per metric
            corr_im = np.mean(corr_im, axis=0)
            for best_local in src_local:
                src.append(src_local[best_local])
                dst.append(dst_local[best_local])
                weight.append(corr_im[best_local[::-1]])
    src = np.array(src)
    dst = np.array(dst)
    src_full = np.array(src_full)
    dst_full = np.array(dst_full)
    weight = np.array(weight).astype(np.float)
    weight_full = np.array(weight_full).astype(np.float)
    # define Transform mpodel to use
    transform = transform_model
    model = transform()
    model.estimate(src, dst)
    model_robust, inliers = ransac_transform((src, dst), transform, weight=old_div(weight, weight.sum(
    )), min_samples=max(old_div(max(dst.shape), 6), 4), residual_threshold=3, max_trials=200)
    robust_transformed_dst = model_robust(src_full)
    distances = np.sqrt(((robust_transformed_dst - dst_full) ** 2).sum(axis=1))
    real_outliers = np.where(distances > 10)[0]
    value_choice = [1, 3]  # see find_template_for_ransac
    if len(real_outliers) > 0:
        # for all outliers find good point close to "best-fit" transformed
        # point
        for k in real_outliers:
            new_coord = robust_transformed_dst[k, :]
            diff_best_coord = dst_full[k, :] - new_coord
            x_local = corr_images[data_src[k]][1][0] + \
                (new_coord[0] - dst_full[k, 0])
            y_local = corr_images[data_src[k]][1][1] + \
                (new_coord[1] - dst_full[k, 1])
            im = corr_images[data_src[k]][0]
            y_local_range = [max(y_local - 150, 0),
                             min(y_local + 150, im.shape[0])]
            x_local_range = [max(x_local - 150, 0),
                             min(x_local + 150, im.shape[1])]
            minMaxValues = cv2.minMaxLoc(
                im[y_local_range[0]:y_local_range[1], x_local_range[0]:x_local_range[1]])
            dst_full[k, :] = [new_coord[0] + (y_local_range[0] - y_local) + minMaxValues[value_choice[-1]]
                              [0], new_coord[1] + (x_local_range[0] - x_local) + minMaxValues[value_choice[-1]][1]]
        model_robust, inliers = ransac_transform((src_full, dst_full), transform, weight=old_div(
            weight_full, weight_full.sum()), min_samples=max(old_div(max(dst.shape), 6), 4), residual_threshold=3, max_trials=200)
    out_contours = None
    if contours_to_move:
        out_contours = [(model_robust(src_c)) for src_c in contours_to_move]
    robust_transformed_dst = model_robust(contour_list_fg_list[0])
    quality = 0
    return robust_transformed_dst, [], out_contours

def writeImage(data, filename):
    """write an image

    :param data: data to write
    :type data: ndarray
    :param filename: location to save image
    :type filename: string
    :return: true or fals
    :rtype: true or false
    """
    if data.max() > 255: #TODO prüfen ob noch nötig + unteren abschnitt anpassen
        try:
            skimage.io.imsave(filename, data)
            LOGGER.info("Write Image %s using 'freeimage'" % filename)
        except Exception as inst:
            LOGGER.error("Could not write File '%s', Exception: %s" %
                         (filename, inst))  # log error
            raise
        return False
    else:
        try:
            try:
                # ImageImage.write(fileneme, skimage.io.imsave(filename, data, 'pil')
                Image.fromarray(data).save(filename)
                LOGGER.info("Write Image %s using 'pil'" % filename)
            except Exception as inst:
                try:
                    skimage.io.imsave(filename, data)
                    LOGGER.info(
                        "write Image %s using 'skimage.io.imread'" % filename)
                except Exception as inst:
                    import cv2
                    cv2.imwrite(filename, data)
                    LOGGER.info("Write Image %s using 'openCV'" % filename)
        except Exception as inst:
            LOGGER.error("Could not write File '%s', Exception: %s" %
                         (filename, inst))  # log error
            raise
        return False
    return True

def relabelMap(labelledImage, posLabel):
    """Function name   : relabelMap
    relabel labelled image , only keep posLabels and relabel them into 1...n
    :param labelledImage: labelled image
    :type labelledImage: numpy array (uint)
    :param posLabel: list of labels to keep
    :type posLabel: list (uint)
    :rtype: numpy array (uint)
    """
    labeledSITKNew = np.zeros_like(labelledImage)
    for k, v in zip(posLabel, list(range(1, len(posLabel) + 1))):
        labeledSITKNew[labelledImage == k] = v
    return labeledSITKNew

    
def writeData(data, filename):
    """writes data to file

    :param data: data to write
    :type data: dtyape
    :param filename: location of the file
    :type filename: string
    """
    try:
        Util.writeData(data, filename)
    except Exception as inst:
        LOGGER.error("Could not marshale File '%s', Exception: %s" %
                     (filename, inst))  # log error
        raise


def readData(filename):
    """reads data from file

    :param filename: file path
    :type filename: string
    :return: read data
    :rtype: dtype
    """
    try:
        return Util.readData(filename)
    except Exception as inst:
        LOGGER.error("Could not read marshaled File '%s', Exception: %s" %
                     (filename, inst))  # log error
        raise
