import cv2
import pylab
import numpy as np
from scipy import optimize as scipy_optimize
from scipy import interpolate as scipy_interpolate
import collections.abc
from past.utils import old_div
import scipy.ndimage as mod

def width_profile_optimizer_function(x, z):
    """optimizes width

    :param x: measured profile
    :type x: ndarray
    :param z: profile parameters
    :type z: ndarray
    :return: optimized width
    :rtype: ndarray
    """
    [x0, x1, x2, a, b] = z
    out = np.zeros_like(x).astype(np.double)
    out[np.logical_or(x <= x0, x >= x0 + x1 + x2 + x1)] = a
    out[np.logical_and(x >= x0 + x1, x <= x0 + x1 + x2)] = b
    out[np.logical_and(x > x0, x < x0 + x1)] = ((b - a) *
                                                (x[np.logical_and(x > x0, x < x0 + x1)] - x0) / x1) + a
    out[np.logical_and(x > x0 + x1 + x2, x < x0 + x1 + x2 + x1)] = (a - b) * (
        x[np.logical_and(x > x0 + x1 + x2, x < x0 + x1 + x2 + x1)] - x0 - 2 * x1 - x2) / x1 + a
    return out


def width_profile_optimizer_func(x, x0, x1, x2, a, b):
    """[summary]

    :param x: independent variable where the data is measure
    :type x: [type]
    :param x0: independent variable where the data is measure
    :type x0: [type]
    :param x1: independent variable where the data is measure
    :type x1: [type]
    :param x2: independent variable where the data is measure
    :type x2: [type]
    :param a: independent variable where the data is measure
    :type a: [type]
    :param b: independent variable where the data is measure
    :type b: [type]
    :return: optimized width
    :rtype: ndarray
    """
    return width_profile_optimizer_function(x, [x0, x1, x2, a, b])

def get_optimized_profile_lstsqr(test_profile, p_start):
    """least square optimizer for profiles

    :param test_profile: test profile
    :type test_profile: ndarray
    :param p_start: start parameter
    :type p_start: Array
    :return: Optimal values for the parameters, calculation proof, value
    :rtype: Array,ndarray, int
    """
 
    try:
        popt, pcov = scipy_optimize.curve_fit(width_profile_optimizer_func, np.arange(
            len(test_profile)), test_profile, p_start)
    except Exception as inst:
       # LOGGER.error("%s" % inst)
        return None, None, np.inf
    calc_prof = width_profile_optimizer_function(
        np.arange(len(test_profile)), popt)
    value = ((calc_prof - test_profile) ** 2).sum()
    return popt, calc_prof, value

def get_optimized_profile_with_initializer(test_profile, mode=None):
    xx = np.arange(len(test_profile))
    test_parameters = [[((width_profile_optimizer_function(xx, [x0, x1, x2, a, b]) - test_profile) ** 2).sum(), x0, x1, x2, a, b]
                       for a in np.linspace(np.median(test_profile), test_profile.max(), 2)
                       for b in np.linspace(test_profile.min(), np.median(test_profile), 2)
                       for x0 in np.linspace(5, 30, 5)
                       for x1 in np.linspace(5, 30, 5)
                       for x2 in np.linspace(np.maximum(5, len(test_profile) - 80), np.maximum(10, len(test_profile) - 20), 5)]
    zz = np.array(test_parameters)
    popt, calc_prof, value = get_optimized_profile_lstsqr(
        test_profile, zz[zz[:, 0].argmin(), 1:])
    if popt is None:
        return None, None
    if mode is not None and (mode == 'test' or mode == "draw"):
        pylab.plot(test_profile, 'b', width_profile_optimizer_function(
            xx, zz[zz[:, 0].argmin(), 1:]), 'g', calc_prof, 'r')
    if mode is not None and mode == 'test':
        test_parameters_tt = [[((width_profile_optimizer_function(xx, [x0, x1, x2, a, b]) - test_profile) ** 2).sum(), x0, x1, x2, a, b]
                              for a in np.linspace(np.median(test_profile), test_profile.max(), 10)
                              for b in np.linspace(test_profile.min(), np.median(test_profile), 10)
                              for x0 in np.linspace(5, 30, 10)
                              for x1 in np.linspace(5, 30, 10)
                              for x2 in np.linspace(np.maximum(5, len(test_profile) - 80), np.maximum(10, len(test_profile) - 20), 10)]
        zz_tt = np.array(test_parameters_tt)
        [x0, x1, x2, a, b] = zz_tt[zz_tt[:, 0].argmin(), 1:]
        popt_tt, pcov = scipy_optimize.curve_fit(width_profile_optimizer_func, np.arange(
            len(test_profile)), test_profile, [x0, x1, x2, a, b])
        calc_prof_tt = width_profile_optimizer_function(
            np.arange(len(test_profile)), popt_tt)
       # pylab.hold(True)
        pylab.plot(width_profile_optimizer_function(
            xx, zz_tt[zz_tt[:, 0].argmin(), 1:]), 'm--', calc_prof_tt, 'y--')
        pylab.cla()

    return popt, calc_prof


def get_width_profile(bin_image, emptytest, line_colour, output_validation_file, pts_idx, steps, val_image, xnew, ynew, min_width=1, add_profile_length=20):
    """[summary]

    :param bin_image: [description]
    :type bin_image: [type]
    :param emptytest: [description]
    :type emptytest: [type]
    :param line_colour: [description]
    :type line_colour: [type]
    :param output_validation_file: [description]
    :type output_validation_file: [type]
    :param pts_idx: [description]
    :type pts_idx: [type]
    :param steps: [description]
    :type steps: [type]
    :param val_image: [description]
    :type val_image: [type]
    :param xnew: [description]
    :type xnew: [type]
    :param ynew: [description]
    :type ynew: [type]
    :param min_width: [description], defaults to 1
    :type min_width: int, optional
    :param add_profile_length: [description], defaults to 20
    :type add_profile_length: int, optional
    :return: [description]
    :rtype: [type]
    """
    p1 = np.array(np.matrix((ynew[pts_idx + 1], xnew[pts_idx + 1])) -
                  np.matrix((ynew[pts_idx + 0], xnew[pts_idx + 0])))[0, :]
    p2 = np.array(np.matrix((ynew[pts_idx + 2], xnew[pts_idx + 2])) -
                  np.matrix((ynew[pts_idx + 1], xnew[pts_idx + 1])))[0, :]
    z = old_div(p1, np.sqrt(p1.dot(p1))) + old_div(p2, np.sqrt(p2.dot(p2)))
    z = old_div(z, np.sqrt(z.dot(z)))
    zT = np.matrix([z[1], -z[0]])
    t_start = np.max(bin_image.shape)
    while True:
        tangent = np.matrix((ynew[pts_idx + 1], xnew[pts_idx + 1])) + np.matrix(
            [-t_start, t_start]).T * zT 
        tangent = tangent.astype(np.int32)
        retval, pt1, pt2 = cv2.clipLine((0, 0, bin_image.shape[1], bin_image.shape[0]), (
            tangent[0, 0], tangent[0, 1]), (tangent[1, 0], tangent[1, 1]))
        if not((tangent[0] < 0).any() and (tangent[1] < 0).any()) and ((pt1 == tangent[0]).all() or (pt1 == tangent[1]).all() or (pt2 == tangent[0]).all() or (pt2 == tangent[1]).all()):
            t_start = t_start * 2
        else:
            break
    if output_validation_file is not None and (pts_idx == 0 or pts_idx == steps - 3 or pts_idx % (old_div(steps, 10)) == 0):
        cv2.line(emptytest, pt1, pt2, line_colour)
        cv2.line(emptytest, (ynew[pts_idx], xnew[pts_idx]),
                 (ynew[pts_idx + 1], xnew[pts_idx + 1]), 2)
        cv2.line(emptytest, (ynew[pts_idx + 1], xnew[pts_idx + 1]),
                 (ynew[pts_idx + 2], xnew[pts_idx + 2]), 2)
        line_colour += 1
    test_image = np.zeros_like(val_image)
    test_image[bin_image] = 1
    li = cv2.cv.InitLineIterator(cv2.cv.fromarray(test_image), pt1, pt2)
    whole_mask = np.array([c for c in li])
    if whole_mask.sum() < min_width:
        return None, emptytest, None
    test_image = np.zeros_like(val_image)
    test_image[:] = val_image[:]
    li = cv2.cv.InitLineIterator(cv2.cv.fromarray(test_image), pt1, pt2)
    whole_profile = np.array([c for c in li])
    possible_edges_complete = np.where(whole_mask > 0)[0]
    test_profile = whole_profile[possible_edges_complete[0] -
                                 add_profile_length: possible_edges_complete[-1] + add_profile_length]
    return test_profile, emptytest, [pt1, pt2, len(whole_profile), possible_edges_complete[0] - add_profile_length, possible_edges_complete[-1] + add_profile_length - 1]

    
def get_spline_width(bin_image, val_image, steps=100, initialize_step=5, output_validation_file=None, validation_rgbfile=None,DEBUG= False):
    """calculate the length and width of binary objects by finding the spline-centre-line,
    calculate width along orthogonal lines. Output median width. Output length where width is
    within 2 MAD(Median deviations)

    :param bin_image: binary image
    :type bin_image: bin_image
    :param val_image: [description]
    :type val_image: [type]
    :param steps: [description], defaults to 100
    :type steps: int, optional
    :param initialize_step: [description], defaults to 5
    :type initialize_step: int, optional
    :param output_validation_file: [description], defaults to None
    :type output_validation_file: [type], optional
    :param validation_rgbfile: [description], defaults to None
    :type validation_rgbfile: [type], optional
    :return: outer length, outer width, width
    :rtype: sum_along_axis : ndarray
    """
    im_nr = 1
    s = 1
    k = 2
    nest = 20
    med_coord = getMedianLineFull(bin_image)
    tckp, u = scipy_interpolate.splprep(
        [[x[0] for x in med_coord], [x[1] for x in med_coord]], s=s, k=k, nest=nest)
    xnew, ynew = scipy_interpolate.splev(np.linspace(0, 1, steps), tckp)
    if len(med_coord) <= k + 1:
        return None, None, None
    tckp, u = scipy_interpolate.splprep(
        [[x[0] for x in med_coord], [x[1] for x in med_coord]], s=s, k=k, nest=nest)
    xnew, ynew = scipy_interpolate.splev(np.linspace(0, 1, steps), tckp)
    emptytest = None
    if output_validation_file is not None:
        emptytest = np.copy(bin_image).astype(np.uint8)
        if validation_rgbfile is None:
            validation_rgbfile = np.copy(emptytest)
    xnew = xnew.astype(np.float32)
    ynew = ynew.astype(np.float32)
    width = []
    width_measured = []
    length_profile = []
    tangent_points = []
    if output_validation_file is not None:
        test = []
    line_colour = 3
    profile_parameters_init = [[0, None]]
    while len(profile_parameters_init) == 1 and initialize_step > 0:
        for pts_idx in range(0, steps - 2):
            if (pts_idx % initialize_step) == 0 or (len(profile_parameters_init) < old_div(pts_idx, initialize_step) + 2 and (pts_idx % initialize_step) == old_div(initialize_step, 2)):
                width_profile, emptytest, plot_line_data = get_width_profile(
                    bin_image, emptytest, line_colour, output_validation_file, pts_idx, steps, val_image, xnew, ynew, min_width=5)
                if width_profile is None or len(width_profile) == 0:
                    continue
                optimal_parameters, optimal_profile = get_optimized_profile_with_initializer(
                    width_profile)
                if optimal_parameters is None:
                    continue
                profile_parameters_init.append([pts_idx, optimal_parameters])
                current_width = optimal_parameters[1:3].sum()
                width.append(current_width)
                width_measured.append(pts_idx)
                length_profile.append(
                    optimal_parameters[-2] - optimal_parameters[-1])
                plot_line_data.extend(optimal_parameters[:3])
                tangent_points.append(plot_line_data)
        if initialize_step == 1:
            initialize_step = -1
        else:
            initialize_step = 1
    if len(profile_parameters_init) == 1 and initialize_step == -1:
        return None, None, None
    profile_parameters_init.append([steps - 3, None])
    profile_parameters_start = collections.deque(maxlen=5)
    for ([pts_idx_before, optimal_parameters_before], [pts_idx_after, optimal_parameters_after]) in zip(profile_parameters_init[:-1], profile_parameters_init[1:]):
        try:
            profile_parameters_start.remove(optimal_parameters_before)
        except ValueError:
            pass
        try:
            profile_parameters_start.remove(optimal_parameters_after)
        except ValueError:
            pass
        if optimal_parameters_before is not None:
            profile_parameters_start.append(optimal_parameters_before)
        if optimal_parameters_after is not None:
            profile_parameters_start.append(optimal_parameters_after)
        for pts_idx in range(pts_idx_before + 1, pts_idx_after):
            width_profile, emptytest, plot_line_data = get_width_profile(
                bin_image, emptytest, line_colour, output_validation_file, pts_idx, steps, val_image, xnew, ynew, min_width=1)
            if width_profile is None or len(width_profile) == 0:
                continue
            optimal_parameters_data = [[get_optimized_profile_lstsqr(
                width_profile, x), x] for x in profile_parameters_start]
            best_value_idx = np.argmin([x[0][2]
                                       for x in optimal_parameters_data])
            if np.isinf(optimal_parameters_data[best_value_idx][0][2]):
                continue
            optimal_parameters, optimal_profile, best_value = optimal_parameters_data[
                best_value_idx][0][:3]
            diff_optimal_parameters = np.min([np.maximum(4 * ((optimal_parameters - x) ** 2).max(
            ), ((optimal_parameters - x) ** 2).sum()) for x in profile_parameters_start])
            if diff_optimal_parameters > 20:
                profile_parameters_start.append(optimal_parameters)
            current_width = optimal_parameters[1:3].sum()
            width.append(current_width)
            width_measured.append(pts_idx)
            length_profile.append(
                optimal_parameters[-2] - optimal_parameters[-1])
            plot_line_data.extend(optimal_parameters[:3])
            tangent_points.append(plot_line_data)
    out_width = np.median(width)
    width = np.array(width)[np.argsort(width_measured)]
    length_profile = np.array(length_profile)[np.argsort(width_measured)]
    st = np.maximum(np.median(np.abs(width - np.median(width))) / 0.6745, 1)
    st_col = np.maximum(
        np.median(np.abs(length_profile - np.median(length_profile))) / 0.6745, 1)
    good_data_width = np.where(np.abs(width - np.median(width)) <= 3 * st)[0]
    good_data_col = np.where(
        np.abs(length_profile - np.median(length_profile)) <= 3 * st_col)[0]
    good_data = np.where(np.logical_and(np.abs(width - np.median(width)) <= 3 *
                         st, np.abs(length_profile - np.median(length_profile)) <= 3 * st_col))[0]
    # sum way along spline
    whole_length = np.sum(np.sqrt(np.diff(xnew) ** 2 + np.diff(ynew) ** 2))
    good_data_parts = getPartsFromIdx(good_data, closeGap=np.maximum(old_div(int(
        whole_length), 50), 1), minLength=np.maximum(old_div(int(whole_length), 100), 1))
    if len(good_data_parts) == 1:
        max_idx = 0
    else:
        partlen = [x[-1] - x[0] for x in good_data_parts]
        max_idx = np.argmax(partlen)
    p_xx = xnew[good_data_parts[max_idx][0]:good_data_parts[max_idx][-1]]
    p_yy = ynew[good_data_parts[max_idx][0]:good_data_parts[max_idx][-1]]
    out_width = np.median(
        width[good_data_parts[max_idx][0]:good_data_parts[max_idx][-1]])
    out_length = np.sum(np.sqrt(np.diff(p_xx) ** 2 + np.diff(p_yy) ** 2))
    if output_validation_file is not None:
        #global pylab
        if not pylab:
            import pylab
        ax1 = pylab.subplot(221)
        pylab.imshow(validation_rgbfile)
        ax2 = pylab.subplot(222)
        pylab.imshow(bin_image)
        ax3 = pylab.subplot(223)
        pylab.imshow(emptytest * bin_image)
       # pylab.hold(True)  Matplotlib now always behaves as if hold = True. To clear an axes you can manually use cla (), or to clear an entire figure use clf ().
        pylab.plot(p_yy, p_xx, 'w', linewidth=0.1)
        pylab.cla() # cla used instead of hold(false)
        ax4 = pylab.subplot(224)
        pylab.imshow(validation_rgbfile)
        #pylab.hold(True)
        pylab.plot(p_yy, p_xx, 'w', linewidth=0.1)
        pylab.cla() # cla used instead of hold(false)
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax4.axis('off')
        if DEBUG:
            pass
        else:
            pylab.savefig(output_validation_file, dpi=600)
        pylab.close()
    return out_length, out_width, width

def getPartsFromIdx(data, closeGap=1, minLength=0):
    """gets parts (index) of function which have same slope direction, closes gaps and set minimum part length

    :param data input array
    :type data: numpy array
    :param closeGap: size of gaps to be closed
    :type closeGap: int(1)
    :param minLength: minimum length of parts to be considered
    :type minLength: int(0)
    :return: list of start, end indexes
    :rtype: list(list(startIdx, endIdx))
    """
    gaps = np.where(np.diff(data) > closeGap)[0].tolist()
    if gaps:
        print((1))
    gaps.insert(0, -1)
    gaps.append(-1)
    partsLeft = [[data[gaps[idx] + 1], data[gaps[idx + 1]] + 1]
                 for idx in range(len(gaps) - 1)]
    if minLength > 0:
        partsLeft = [x for x in partsLeft if x[1] - x[0] >= minLength]
    return partsLeft

def getMedianLineFull(bin_image):
    """detects median line of binary image

    :param bin_image: binary image
    :type bin_image: numpy labeled image
    :return: coords
    :rtype: list
    """
    bbox = getBoundingBoxes(bin_image)[0]
    yy = bin_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    x_good = np.where(np.abs(np.diff(yy, axis=1)).sum(axis=1) == 2)[0]
    y_good = yy[x_good, :]
    coords = [[r + bbox[1], np.where(yy[r, :].cumsum() >= 0.5 * yy[r, :].sum())[
        0][0] + 1 + bbox[0]] for r in x_good]
    return coords

def getBoundingBoxes(labelledImage, posSet=None):
    """get bounding box for each label

    :param labelledImage:  numpy label image
    :type labelledImage: ndarray
    :param posSet: bounding boxes to find, defaults to None
    :type posSet: list, optional
    :return: list of bounding boxes
    :rtype: list
    """
    def dummyMap(sl):
        if sl is None:
            return (0, 0, 0, 0)
        dy, dx = sl[:2]
        return (dx.start, dy.start, dx.stop + 1, dy.stop + 1)
    if posSet is not None:
        data = mod.find_objects(labelledImage, int(
            np.array(list(posSet)).max()) + 1)
        bbox = [dummyMap(data[int(x) - 1]) for x in posSet]
    else:
        data = mod.find_objects(labelledImage)
        bbox = list(map(dummyMap, data))
    return bbox