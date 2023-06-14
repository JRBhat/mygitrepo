from __future__ import division
from PIL import Image, ImageDraw, ImageFont
from numpy.lib.shape_base import expand_dims
from past.utils import old_div
from . import ImageIO as Io
import numpy as np

def drawImageCircle(filename, circles, outFile):
    """Draws a circles on to image

    :param filename: Image 
    :type filename: pathlib.Path
    :param circles: circles
    :type circles: Iterable of circles 
    :param outFile: Image with circles
    :type outFile: pathlib.Path
    """
    orgIm = Image.open(filename)
    draw = ImageDraw.Draw(orgIm)
    for idx, circle in enumerate(circles):
        draw.ellipse((circle[0] - circle[2], circle[1] - circle[2], circle[0] + circle[2],
                     circle[1] + circle[2]), fill=None, outline=old_div(255, (idx + 1)))
    del draw
    orgIm.save(outFile)

def drawImageBBox(filename, bboxes, outFile=None, colors=None, legend=None, add_text=None, keep_image=False, width=1, text_size=50, extend=False):
    """Draws bounding boxes on to image

    :param filename: Image
    :type filename: String
    :param bboxes: Bounding boxes
    :type bboxes: Iterable of tupel
    :param outFile: Image, defaults to None
    :type outFile: String, optional
    :param colors: text and outline colors, defaults to None
    :type colors: Tupel for one color; Array of Tupel for more colors, optional
    :param legend: legend for colors, defaults to None
    :type legend: (String,Color), optional
    :param add_text: text for bboxes, defaults to None
    :type add_text: Array of String, optional
    :param keep_image: keep image, defaults to False
    :type keep_image: bool, optional
    :param width: width of the box, defaults to 1
    :type width: int, optional
    :param text_size: text size, defaults to 50
    :type text_size: int, optional
    :param extend: extend image, defaults to False
    :type extend: bool, optional
    """
    def extend_image(foreground_old, col=0):
        """extends the given image

        :param foreground_old: Image in RGB
        :type foreground_old: NumpyArray
        :param col: color, defaults to 0
        :type col: int, optional
        :return: extended image
        :rtype: Image
        """
        size = list(foreground_old.shape)
        size[0] += 400
        size[1] += 400
        inpaint_mask = np.ones(size, foreground_old.dtype)
        if len(size) == 3:
            for k in range(len(size)):
                inpaint_mask[..., k] *= col[k]
        else:
            inpaint_mask *= col
        inpaint_mask[200:-200, 200:-200] = foreground_old
        return inpaint_mask
    if extend:
        orgIm = extend_image(Io.readRGBImage(
            filename, keep=keep_image), col=[0, 0, 0])
    else:
        orgIm = Io.readRGBImage(filename, keep=keep_image)
    if orgIm.shape[2] == 3:
        mode = 'RGB'
    else:
        raise Exception("wrong image format, must be RGB")
    orgIm = Image.fromarray(orgIm, mode)
    draw = ImageDraw.Draw(orgIm)
    if colors is not None:
        if not isinstance(colors[0], (list, tuple)):
            colors = [colors]
    text_size = 20
    f = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", text_size)
    for idx, bbox in enumerate(bboxes):
        if colors is None:
            drawColor = (255, 0, 0)
        else:
            if idx >= len(colors):
                drawColor = colors[-1]
            else:
                drawColor = colors[idx]
        for x in range(max(1, width - 1)):
            draw.rectangle(
                tuple((np.array(bbox) + [-x, -x, x, x]).tolist()), fill=None, outline=drawColor)
        if add_text is not None:
            draw.text((bbox[2], bbox[3]), add_text[idx],
                      fill=drawColor, font=f)
    f = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", text_size)
    if legend:
        sz = 0
        for leg in legend:
            draw.text((0, sz), leg[1], fill=leg[0], font=f)
            sz += text_size
    del draw
    if outFile is not None:
        orgIm.save(outFile)
    else:
        return orgIm


def draw_polygone_on_image(filename, points, manual_points=None, outFile=None, color=(255, 255, 255), radius=10, width=1):
    """Draws a polygone on image

    :param filename: Image
    :type filename: String
    :param points: points
    :type points: Array
    :param manual_points: manual points selection, defaults to None
    :type manual_points: Array of point indexes, optional
    :param outFile: outfile, defaults to None
    :type outFile: String, optional
    :param color: Color, defaults to (255, 255, 255)
    :type color: tuple, optional
    :param radius: radius, defaults to 10
    :type radius: int, optional
    :param width: width, defaults to 1
    :type width: int, optional
    :return: Images
    :rtype: NumpyArray
    """
    orgIm = Image.open(filename)
    draw = ImageDraw.Draw(orgIm)
    ttp = [(x[0], x[1]) for x in points]
    ttp.append((points[0][0], points[0][1]))
    for x, y in zip(ttp[:-1], ttp[1:]):
        draw.line([x, y], fill=color, width=width)
    if manual_points is None:
        for point in points:
            draw.ellipse((point[0] - radius, point[1] - radius,
                         point[0] + radius, point[1] + radius), fill=color)
    else:
        for idx in manual_points:
            point = points[idx]
            draw.ellipse((point[0] - radius, point[1] - radius,
                         point[0] + radius, point[1] + radius), fill=color)
    del draw
    if outFile is not None:
        orgIm.save(outFile)
    return np.array(orgIm)


def draw_polygones_on_image(input_image, points, manual_points=None, outFile=None, colors=((255, 255, 255)), radius=10, width=1):
    """Draws a polygones on image

    :param filename: Image
    :type filename: Array
    :param points: points
    :type points: Array
    :param manual_points: manual points selection, defaults to None
    :type manual_points: Array of point indexes, optional
    :param outFile: outfile, defaults to None
    :type outFile: String, optional
    :param color: Color, defaults to (255, 255, 255)
    :type color: tuple, optional
    :param radius: radius, defaults to 10
    :type radius: int, optional
    :param width: width, defaults to 1
    :type width: int, optional
    :return: Images
    :rtype: NumpyArray
    """
    orgIm = Image.fromarray(input_image)
    draw = ImageDraw.Draw(orgIm)
    for idx in range(len(points)):
        if idx-1 > len(colors):
            color = colors[-1]
        else:
            color = colors[idx]
        ttp = [(x[0], x[1]) for x in points[idx]]
        if not ttp:
            continue
        ttp.append((points[idx][0][0], points[idx][0][1]))
        for x, y in zip(ttp[:-1], ttp[1:]):
            draw.line([x, y], fill=color, width=width)
        if manual_points is None:
            for point in points[idx]:
                draw.ellipse((point[0] - radius, point[1] - radius,
                             point[0] + radius, point[1] + radius), fill=color)
        else:
            for p_idx in manual_points[idx]:
                point = points[idx][p_idx]
                draw.ellipse((point[0] - radius, point[1] - radius,
                             point[0] + radius, point[1] + radius), fill=color)
    del draw
    if outFile is not None:
        orgIm.save(outFile)
    return np.array(orgIm)

def addTextToImage(text, filename, outfile=None):
    """Draws white text in upper left corner of image

    :param text: text to draw
    :type text: string
    :param filename: input file
    :type filename: string
    :param outfile: if not None, write new File
    :type outfile: string (None)
    """
    orgIm = Image.open(filename)
    image = Image.new(orgIm.mode, (orgIm.size[0], 60))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 50)
    draw.text((0, 0), text, font=font)
    mask = Image.new("RGBA", (orgIm.size[0], 60))
    imageDraw = ImageDraw.Draw(mask)
    imageFont = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 50)
    imageDraw.text((0, 0), text, font=imageFont)
    oIm = orgIm.crop((0, 0, orgIm.size[0], 60))
    testim = Image.composite(image, oIm, mask)
    orgIm.paste(testim, (0, 0))
    if outfile is not None:
        orgIm.save(outfile)



def getNPTextImage(text, textSize=50, mode="RGB"):
    """Draws white text in upper left corner of an image

    :param text: text to draw
    :type text: str
    :param textSize: text size, defaults to 50
    :type textSize: int, optional
    :param mode: mode for the image to use, defaults to "RGB"
    :type mode: str, optional
    :return: image as ndarray
    :rtype: ndarray
    """
    imFont = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", textSize)
    imSize = imFont.getsize(text)
    image = Image.new(mode, imSize)
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text, font=imFont)
    return np.array(image)

def add_grid_to_image(filename, grid=(6, 6), outfile=None, linewidth=9):
    """add grid to image mainly for repositioning

    :param filename: input file
    :type filename: Str
    :param grid: number of grid elements in x and y direction, defaults to (6, 6)
    :type grid: tuple, optional
    :param outfile: output file, defaults to None
    :type outfile: str, optional
    :param linewidth: line width in pixel, defaults to 9
    :type linewidth: int, optional
    :return: image with grid as numpy array if outfile is None, or None
    :rtype: ndarray
    """
    if isinstance(grid, (tuple, list)):
        if len(grid) == 1:
            gridx = grid[0]
            gridy = grid[0]
        else:
            gridx = grid[0]
            gridy = grid[1]
    else:
        gridx = grid
        gridy = grid
    im = Io.readImage(filename)
    gridxs = np.round(np.linspace(0, 1.0, gridx + 1)
                      * im.shape[0]).astype(np.int)
    gridys = np.round(np.linspace(0, 1.0, gridy + 1)
                      * im.shape[1]).astype(np.int)
    lines = [[0, int(x), im.shape[1], int(x)] for x in gridxs[1:]]
    lines.extend([[int(x), 0, int(x), im.shape[0]] for x in gridys[1:]])
    if im.shape[2] == 3:
        mode = 'RGB'
    elif im.shape[2] == 1:
        mode = 'L'
    else:
        mode = 'RGBA'
    orgIm = Image.fromarray(im, mode)
    i = Image.new(orgIm.mode, orgIm.size)
    d = ImageDraw.Draw(i)
    for line in lines:
        d.line(line, width=linewidth)
    mask = Image.new("RGBA", orgIm.size)
    dm = ImageDraw.Draw(mask)
    for line in lines:
        dm.line(line, width=linewidth)
    testim = Image.composite(i, orgIm, mask)
    orgIm.paste(testim, (0, 0))
    if outfile is not None:
        orgIm.save(outfile)
    else:
        return orgIm

def addScaleBarToImage(scale, filename, outfile=None, linewidth=9, scaleBarLength=10, barWidth=0.5, scaleText=["0", "10mm"], fontSize=80):
    """draws scale bar on upper left corner in image (ToDo: other areas, color?)

    :param scale: pixel per unit ("defined" by scaleText)
    :type scale: float
    :param filename: input file
    :type filename: str
    :param outfile: if not None, write new File, defaults to None
    :type outfile: str, optional
    :param linewidth: in pixel, defaults to 9
    :type linewidth: int, optional
    :param scaleBarLength: in units, defaults to 10
    :type scaleBarLength: int, optional
    :param barWidth: length of vertical bar dividers, defaults to 0.5
    :type barWidth: float, optional
    :param scaleText: text written at scale bar - defines units, defaults to ["0", "10mm"]
    :type scaleText: list, optional
    :param fontSize: in pixel, defaults to 80
    :type fontSize: int, optional
    :return: image with scale bar
    :rtype: ndarray
    """
    if isinstance(scaleBarLength, (tuple, list)):
        scaleBarFullLength = scaleBarLength[-1]
        betweenScales = scaleBarLength[:-1]
    else:
        scaleBarFullLength = scaleBarLength
        betweenScales = []
    start = np.array([20, 20]) + scale * barWidth
    ende = start + [scale * scaleBarFullLength, 0]
    lines = [np.array([start, ende]).astype(np.int).flatten(),
             np.array([start - [0, scale * barWidth], start +
                      [0, scale * barWidth]]).astype(np.int).flatten(),
             np.array([ende - [0, scale * barWidth], ende + [0, scale * barWidth]]).astype(np.int).flatten()]
    for sca in betweenScales:
        lines.append(np.array([start + [scale * sca, -scale * barWidth // 2], start + [
                     scale * sca, scale * barWidth // 2]]).astype(np.int).flatten())
    startText = (start + [0, scale * barWidth]).astype(np.int)
    posText = [startText, (ende + [0, scale * barWidth]).astype(np.int)]
    imTest = Io.readRGBImage(filename)
    if (len(imTest.shape) == 3) and imTest.shape[2] == 3:
        mode = 'RGB'
    elif ((len(imTest.shape) == 3) and imTest.shape[2] == 1) or (len(imTest.shape) == 2):
        mode = 'L'
    else:
        mode = 'RGBA'
    orgIm = Image.fromarray(imTest, mode)
    i = Image.new(orgIm.mode, (orgIm.size[0], startText[1] + 20 + fontSize))
    d = ImageDraw.Draw(i)
    f = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", fontSize)
    for idx in range(len(posText)):
        textSize = d.textsize(scaleText[idx], font=f)
        print(textSize)
        d.text((posText[idx] - [old_div(textSize[0], 2), 0]
                ).astype(np.int).tolist(), scaleText[idx], font=f)
    for line in lines:
        d.line(line.tolist(), width=linewidth)
    mask = Image.new("RGBA", (orgIm.size[0], startText[1] + 20 + fontSize))
    dm = ImageDraw.Draw(mask)
    for idx in range(len(posText)):
        textSize = dm.textsize(scaleText[idx], font=f)
        print(textSize)
        dm.text((posText[idx] - [old_div(textSize[0], 2), 0]
                 ).astype(np.int).tolist(), scaleText[idx], font=f)
    for line in lines:
        dm.line(line.tolist(), width=linewidth)
    oIm = orgIm.crop((0, 0, orgIm.size[0], startText[1] + 20 + fontSize))
    testim = Image.composite(i, oIm, mask)
    orgIm.paste(testim, (0, 0))
    if outfile is not None:
        orgIm.save(outfile)
    else:
        return orgIm

def addTextToImagePos(text, filename, outfile=None, pos=(0, 0), textSize=50):
    """Draws white text in upper left corner of an image

    :param text: text to draw
    :type text: string
    :param filename: input file
    :type filename: string
    :param outfile: if not None, write new File
    :type outfile: string (None)
    """
    orgIm = Image.open(filename)
    textNP = getNPTextImage(text, textSize)
    imNP = np.array(orgIm)
    if (len(textNP.shape) == 3) and (len(imNP.shape) == 2):
        imNP[pos[0]: pos[0] + textNP.shape[0], pos[1]: pos[1] + textNP.shape[1]] = textNP.max(axis=2)
    else:
        imNP[pos[0]: pos[0] + textNP.shape[0], pos[1]: pos[1] + textNP.shape[1], :3] = textNP
    orgIm = Image.fromarray(imNP)
    if outfile is not None:
        orgIm.save(outfile)
    else:
        return imNP

def blendImageCircle(filename, circle, outfile=None, bds=0):
    """Blends a circle into image

    :param filename: Image location
    :type filename: string
    :param circle: Circle
    :type circle: cv.CV_32FC3
    :param outfile: output location, defaults to None
    :type outfile: string, optional
    :param bds: reduce radius of circle, defaults to 0
    :type bds: int, optional
    :return: Image
    :rtype: Image
    """
    orgIm = Image.fromarray(
        (old_div(Io.readRGBImage(filename), 255)).astype(np.uint8))
    maskImage = Image.new("1", (orgIm.size[0], orgIm.size[1]))
    draw = ImageDraw.Draw(maskImage)
    circle[2] -= bds  # reduce mask
    draw.ellipse((circle[0] - circle[2], circle[1] - circle[2], circle[0] + circle[2],
                 circle[1] + circle[2]), fill=255, outline=255)
    del draw
    testim = Image.blend(orgIm, maskImage.convert("RGB"), 0.6)
    if outfile is not None:
        testim.save(outfile)
    else:
        return testim