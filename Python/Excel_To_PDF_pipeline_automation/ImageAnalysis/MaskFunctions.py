import numpy as np
import cv2
from past.utils import old_div
from PIL import Image, ImageDraw
from . import ImageIO as Io
    
def maskImage(filename, mask, outfile=None):
    """combines two images

    :param filename: path of first image
    :type filename: pathlib.Path
    :param mask: path of second image
    :type mask: pathlib.Path
    :param outfile: path of output image, defaults to None
    :type outfile: pathlib.Path, optional
    :return: new image
    :rtype: Image
    """
    orgIm = Image.open(filename)
    maskImage = Image.open(mask)
    if ((orgIm.size[0] > orgIm.size[1]) and (maskImage.size[0] < maskImage.size[1])) or ((orgIm.size[0] < orgIm.size[1]) and (maskImage.size[0] > maskImage.size[1])):
        maskImage = maskImage.transpose(Image.ROTATE_270)
    if orgIm.size == maskImage.size:
        testim = Image.composite(orgIm, maskImage.convert("RGB"), maskImage)
    else:
        maskImage = maskImage.crop((0, 0, orgIm.size[0], orgIm.size[1]))
        testim = Image.composite(orgIm, maskImage.convert("RGB"), maskImage)
    if outfile is not None:
        testim.save(outfile)
    else:
        return testim


def getMaskCircle(filename, canny_threshold=200, circle_quality=200, mean_radius=old_div(3950, 2), diff_radius=10):
    """ The function finds circles in an image using a modification of the Hough transform.

    :param filename: Image
    :type filename: pathlib.Path
    :param canny_threshold: canny threshold, defaults to 200
    :type canny_threshold: int, optional
    :param circle_quality: accumulator threshold for the circle centers at the detection stage, defaults to 200
    :type circle_quality: int, optional
    :param mean_radius: mean radius of circles, defaults to old_div(3950, 2)
    :type mean_radius: int, optional
    :param diff_radius: allowed difference from mean radius, defaults to 10
    :type diff_radius: int, optional
    :return: found circles
    :rtype: Array of cv.CV_32FC3
    """
    img = cv2.imread(filename)
    # Convert colorspace to gray
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find circles
    circles = cv2.HoughCircles(grey, cv2.cv.CV_HOUGH_GRADIENT, 10, minDist=100, param1=canny_threshold, param2=circle_quality,
                               minRadius=mean_radius - diff_radius, maxRadius=mean_radius + diff_radius)  
    return circles

def getMaskCircleData(grey, canny_threshold=200, circle_quality=200, mean_radius=old_div(3950, 2), diff_radius=10):
    """The function finds circles in an image using a modification of the Hough transform.

    :param grey: grey Image to analyze
    :type grey: Image
    :param canny_threshold: canny threshold, defaults to 200
    :type canny_threshold: int, optional
    :param circle_quality: accumulator  threshold for the circle centers at the detection stage, defaults to 200
    :type circle_quality: int, optional
    :param mean_radius: mean radius of circles, defaults to old_div(3950, 2)
    :type mean_radius: int, optional
    :param diff_radius: allowed difference from mean radius, defaults to 10
    :type diff_radius: int, optional
    :return: found circles
    :rtype: Array of cv.CV_32FC3
    """
    circles = cv2.HoughCircles(grey, cv2.cv.CV_HOUGH_GRADIENT, 10, minDist=100, param1=canny_threshold, param2=circle_quality,
                               minRadius=mean_radius - diff_radius, maxRadius=mean_radius + diff_radius)  
    return circles


def create_mask_from_polygone_n(im_shape, points, bg_value=0, fg_value=1, dtype=np.uint8):
    """Creates mask from polygone

    :param im_shape: image shape
    :type im_shape: array dimensions
    :param points: points
    :type points: Array
    :param bg_value: background indicator, defaults to 0
    :type bg_value: int, optional
    :param fg_value: foregorund indicator, defaults to 1
    :type fg_value: int, optional
    :param dtype: datatype of the mask, defaults to np.uint8
    :type dtype: dtype, optional
    :return: only background if invalid else mask and  [xmin, xmax, ymin, ymax]
    :rtype: NumpyArray, Array
    """
    y = np.array([z[0] for z in points])
    x = np.array([z[1] for z in points])
    xmin = int(np.round(np.max([0, x.min() - 10])))
    xmax = int(np.round(np.min([im_shape[0], x.max() + 10])))
    ymin = int(np.round(np.max([0, y.min() - 10])))
    ymax = int(np.round(np.min([im_shape[1], y.max() + 10])))
    if xmin >= xmax or ymin >= ymax:
        return np.ones(im_shape, np.uint8) * bg_value
    im = Image.new("L", (ymax - ymin, xmax - xmin))
    draw = ImageDraw.Draw(im)
    ttp = [(x[0] - ymin, x[1] - xmin) for x in points]
    draw.polygon(ttp, fill=255, outline=0)
    del draw
    out_im = np.array(im)
    out_im_cp = np.copy(out_im)
    out_im[out_im_cp == 0] = bg_value
    out_im[out_im_cp == 255] = fg_value
    out_im_n = bg_value * np.ones(im_shape, dtype=dtype)
    out_im_n[xmin:xmax, ymin:ymax] = out_im
    return out_im_n.astype(dtype), [xmin, xmax, ymin, ymax]


def create_mask_from_circles(im_shape, points, diameter, bg_value=0, fg_value=1, dtype=np.uint8):
    """ Create mask from circle

    :param im_shape: Image shape
    :type im_shape: array dimensions
    :param points: points
    :type points: Array
    :param diameter: diameter
    :type diameter: Array
    :param bg_value: background indicator, defaults to 0
    :type bg_value: int, optional
    :param fg_value: foregorund indicator, defaults to 1
    :type fg_value: int, optional
    :param dtype: datatype of the mask, defaults to np.uint8
    :type dtype: dtype, optional
    :return: only background if invalid else mask
    :rtype: NumpyArray, Array
    """
    y = np.array([z[0] for z in points])
    x = np.array([z[1] for z in points])
    xmin = int(np.round(np.max([0, x.min() - max(diameter)])))
    xmax = int(np.round(np.min([im_shape[0], x.max() + max(diameter)])))
    ymin = int(np.round(np.max([0, y.min() - max(diameter)])))
    ymax = int(np.round(np.min([im_shape[1], y.max() + max(diameter)])))
    if xmin >= xmax or ymin >= ymax:
        return np.ones(im_shape, np.uint8) * bg_value
    im = Image.new("L", (ymax - ymin, xmax - xmin))
    draw = ImageDraw.Draw(im)
    ttp = [(x[0] - ymin, x[1] - xmin) for x in points]  # ;
    for c_ppx, rad in zip(ttp, diameter):
        draw.ellipse([c_ppx[0] - old_div(rad, 2), c_ppx[1] - old_div(rad, 2), c_ppx[0] + old_div(
            rad, 2), c_ppx[1] + old_div(rad, 2)],  fill=1)
    del draw
    out_im = np.array(im)
    out_im_cp = np.copy(out_im)
    if bg_value != 0:
        out_im[out_im_cp == 0] = bg_value
    if fg_value != 1:
        out_im[out_im_cp == 1] = fg_value
    out_im_n = bg_value * np.ones(im_shape, dtype=dtype)
    out_im_n[xmin:xmax, ymin:ymax] = out_im
    return out_im_n.astype(dtype)


def create_labelled_mask_from_circles(im_shape, points, radius, bg_value=0, fg_values=1, dtype=np.uint8):
    """ Creates labelled mask from circles

    :param im_shape: Image shape
    :type im_shape: array dimensions
    :param points: points
    :type points: Array
    :param radius: radius of circles
    :type radius: Array of int
    :param bg_value: background indicator, defaults to 0
    :type bg_value: int, optional
    :param fg_values: foreground indicator, defaults to 1
    :type fg_values: int, optional
    :param dtype: datatype of mask, defaults to np.uint8
    :type dtype: dtype, optional
    :return: mask as dtype
    :rtype: dtype
    """
    y = np.array([z[0] for z in points])
    x = np.array([z[1] for z in points])
    xmin = int(np.round(np.max([0, x.min() - max(radius)])))
    xmax = int(np.round(np.min([im_shape[0], x.max() + max(radius)])))
    ymin = int(np.round(np.max([0, y.min() - max(radius)])))
    ymax = int(np.round(np.min([im_shape[1], y.max() + max(radius)])))
    if xmin >= xmax or ymin >= ymax:
        return np.ones(im_shape, np.uint8) * bg_value
    im = Image.new("L", (ymax - ymin, xmax - xmin))
    draw = ImageDraw.Draw(im)
    ttp = [(x[0] - ymin, x[1] - xmin) for x in points]  # ;
    for idx, (c_ppx, rad) in enumerate(zip(ttp, radius)):
        try:
            label = fg_values[idx]
        except:
            try:
                label += 1
            except:
                try:
                    label = fg_values[-1]
                except:
                    try:
                        label = fg_values
                    except:
                        label = 1
        draw.ellipse([c_ppx[0] - old_div(rad, 2), c_ppx[1] - old_div(rad, 2), c_ppx[0] + old_div(rad, 2),
                     c_ppx[1] + old_div(rad, 2)],  fill=label)
    del draw
    out_im = np.array(im)
    out_im_cp = np.copy(out_im)
    out_im[out_im_cp == 0] = bg_value
    out_im_n = bg_value * np.ones(im_shape, dtype=dtype)
    out_im_n[xmin:xmax, ymin:ymax] = out_im
    return out_im_n.astype(dtype)


def create_mask_from_polygone(im_shape, points, bg_value=0, fg_value=1, dtype=np.uint8):
    """Creates mask from polygone

    :param im_shape: image shape
    :type im_shape: array dimensions
    :param points: points
    :type points: Array
    :param bg_value: background indicator, defaults to 0
    :type bg_value: int, optional
    :param fg_value: foregorund indicator, defaults to 1
    :type fg_value: int, optional
    :param dtype: datatype of the mask, defaults to np.uint8
    :type dtype: dtype, optional
    :return: mask
    :rtype: dtype
    """
    im = Image.new("L", (im_shape[1], im_shape[0]))
    draw = ImageDraw.Draw(im)
    ttp = [(np.round(x[0]), np.round(x[1])) for x in points]
    draw.polygon(ttp, fill=255)
    del draw
    out_im = np.array(im)
    out_im_cp = np.copy(out_im)
    out_im[out_im_cp == 0] = bg_value
    out_im[out_im_cp == 255] = fg_value
    return out_im.astype(dtype)


def mask_from_hough_circle(orgIm_size, circle):
    """creates mask from hough circle

    :param orgIm_size: Image size
    :type orgIm_size: array dimensions
    :param circle: the circle
    :type circle: cv.CV_32FC3
    :return: mask image
    :rtype: Image
    """
    maskImage = Image.new("1", (orgIm_size[0], orgIm_size[1]))
    draw = ImageDraw.Draw(maskImage)
    draw.ellipse((circle[0] - circle[2], circle[1] - circle[2], circle[0] + circle[2],
                 circle[1] + circle[2]), fill=255, outline=255)
    del draw
    return maskImage


def maskImageCircle(filename, circle, bds=0, outfile=None):
    """blends the image with a circle mask

    :param filename: Image 
    :type filename: String
    :param circle: Circle
    :type circle: cv.CV_32FC3
    :param bds: reduce the radius of the circle, defaults to 0
    :type bds: int, optional
    :param outfile: outfile location, defaults to None
    :type outfile: String, optional
    :return: Image
    :rtype: Image
    """

    orgIm = Image.fromarray(Io.readRGBImage(filename))
    maskImage = Image.new("1", (orgIm.size[0], orgIm.size[1]))
    draw = ImageDraw.Draw(maskImage)
    circle[2] -= bds  # reduce mask
    draw.ellipse((circle[0] - circle[2], circle[1] - circle[2], circle[0] + circle[2],
                 circle[1] + circle[2]), fill=255, outline=255)
    del draw
    testim = Image.composite(orgIm, maskImage.convert("RGB"), maskImage)
    if outfile is not None:
        testim.save(outfile)
    else:
        return testim


def maskImageBar(filename, bbox, outfile=None, blackBG=True):
    """creates a bar on image

    :param filename: Image
    :type filename: string
    :param bbox: Bbbox
    :type bbox: Numpy Array
    :param outfile: outfile location, defaults to None
    :type outfile: string, optional
    :param blackBG: black background, defaults to True
    :type blackBG: bool, optional
    :return: Image
    :rtype: Image
    """
    orgIm = Image.open(filename)
    if blackBG:
        maskImage = Image.new("1", (orgIm.size[0], orgIm.size[1]))
    else:
        maskImage = Image.new("1", (orgIm.size[0], orgIm.size[1]), 255)
    draw = ImageDraw.Draw(maskImage)
    if blackBG:
        draw.rectangle([bbox[0], bbox[1], bbox[0] + bbox[2],
                       bbox[1] + bbox[3]], fill=255, outline=255)
    else:
        draw.rectangle([bbox[0], bbox[1], bbox[0] + bbox[2],
                       bbox[1] + bbox[3]], fill=0, outline=0)
    del draw
    testim = Image.composite(orgIm, maskImage.convert("RGB"), maskImage)
    if outfile is not None:
        testim.save(outfile)
    else:
        return testim