"""
Script that contains Region of Interest classes for the GUI

- Rectangular ROI
- Circular ROI
- Trapezoid ROI
"""
import numpy as np
from math import pi
from params import *


def check_point(x, y, p1, p2):
    """
    Function that checks where a point lies in relation to a line
    Args:
        x: x coordinate of point under test
        y: y coordinate of point under test
        p1/2: points of the line

    Returns:
    """
    return np.sign((p2[0] - p1[0]) * (y - p1[1]) - (p2[1] - p1[1]) * (x - p1[0]))

class Bbox:
    def __init__(self, _bbox_a = None):
        self.set(_bbox_a)

    def set(self, _bbox_a):
        if _bbox_a is not None:
            self.a = np.array(_bbox_a).astype(int)
        else:
            self.a = np.zeros((4,2))

        self._recalc()

    def _recalc(self):
        self.width = self.a[1][0] - self.a[0][0] + 1
        self.height = self.a[1][1] - self.a[0][1] + 1

    def fromBboxes(self, bboxes):
        bboxes_a = []
        for bbox in bboxes:
            bboxes_a.append(bbox.a[0])
            bboxes_a.append(bbox.a[1])

        bboxes_a = np.array(bboxes_a)
        self.a = np.array([np.min(bboxes_a, axis=0), np.max(bboxes_a, axis=0)])
        self._recalc()

    def getArray(self):
        return self.a

    def getFirstPoint(self):
        return self.a[0]

    def getSize(self):
        return (self.width, self.height)

    def getSlice(self, ref_box = None):
        offset = self.getFirstPoint()
        #print("Get-Slice: %d:%d, %d:%d" % (offset[1], offset[1] + self.height, offset[0], offset[0] + self.width))
        if ref_box != None:
            offset = offset - ref_box.getFirstPoint()
            #print("Slice-Ref: %d:%d, %d:%d" % (offset[1], offset[1] + self.height, offset[0], offset[0] + self.width))

        return np.s_[offset[1] : offset[1]+self.height, offset[0] : offset[0]+self.width]

    def getPlotPoints(self):
        return ([self.a[0,0], self.a[1,0], self.a[1,0], self.a[0,0], self.a[0,0]],
                [self.a[0,1], self.a[0,1], self.a[1,1], self.a[1,1], self.a[0,1]])

    def getCenter(self):
        offset = self.getFirstPoint()
        return (offset[0] + self.width/2, offset[1] + self.height/2)

    def enumerateX(self):
        return enumerate(np.arange(self.a[0][0], self.a[1][0], 1))

    def enumerateY(self):
        return enumerate(np.arange(self.a[0][1], self.a[1][1], 1))

    def getExtended(self, border):
        new_a = np.array(self.a)
        new_a[0] -= border
        new_a[1] += border

        return Bbox(new_a)

    def addToParams(self, params):
        params.addParam(key='top_left_x', name="", default=int(self.a[0][0]))
        params.addParam(key='top_left_y', name="", default=int(self.a[0][1]))
        params.addParam(key='bottom_right_x', name="", default=int(self.a[1][0]))
        params.addParam(key='bottom_right_y', name="", default=int(self.a[1][1]))

    def readFromParams(self, params):
        coords = []
        coords.append([params.getValue('top_left_x'), params.getValue('top_left_y')])
        coords.append([params.getValue('bottom_right_x'), params.getValue('bottom_right_y')])
        self.set(coords)

class RectangularRoi:
    def __init__(self, img, bbox_a):

        self.bbox = Bbox(bbox_a)

        self.img = img.getSubImage(self.bbox)
        self.width = self.img.columns
        self.height = self.img.lines

        self.mask = np.ones(self.img.data.shape)

    def plotBorder(self, axis):
        x,y = self.bbox.getPlotPoints()
        axis.plot(x, y, color='red', linewidth=3)

    def getMaskData(self):
        return self.mask

    def saveToTar(self, tar, fname_prefix):
        # create a temporary param set for easy storage to tar
        params = ParamGroup('roi', "")
        params.addParam(key='id', name="", default="RectangularRoi")
        # add bbox data to params
        self.bbox.addToParams(params)

        # save image parameter to tar
        params.writeToTar(tar, fname_prefix)

class CircularRoi:
    def __init__(self, img, bbox_a):
        self.bbox = Bbox(bbox_a)

        self.img = img.getSubImage(self.bbox)

        ## Setting outer pixels to zero by masking. Calculate mask image ...
        self.width = self.img.columns
        self.height = self.img.lines

        x0 =  self.width / 2 - 0.5
        y0 =  self.height / 2 - 0.5
        w2 = (self.width / 2 - 0.5) ** 2
        h2 = (self.height / 2 - 0.5) ** 2

        self.mask = np.zeros(self.img.data.shape)
        for y in range(self.height):
            for x in range(self.width):
                if ((x - x0) ** 2) / w2 + ((y - y0) ** 2) / h2 <= 1:
                    self.mask[y][x] = 1
        # multiply image with mask image
        masked_data = self.img.data * self.mask
        self.img.data = masked_data

    def plotBorder(self, axis):
        t = np.linspace(0, 2 * pi, 100)
        x0, y0 = self.bbox.getCenter()
        x0 -= 0.5
        y0 -= 0.5
        axis.plot(x0 + (self.width/2-1) * np.cos(t), y0 + (self.height/2-1) * np.sin(t), color='red', linewidth=3)

    def getMaskData(self):
        return self.mask

    def saveToTar(self, tar, fname_prefix):
        # create a temporary param set for easy storage to tar
        params = ParamGroup('roi', "")
        params.addParam(key='id', name="", default="CircularRoi")
        # add bbox data to params
        self.bbox.addToParams(params)

        # save image parameter to tar
        params.writeToTar(tar, fname_prefix)

class TrapezoidRoi:
    def __init__(self, img, vertices):
        self.vertices = np.array(vertices).astype(int)       # ensure that coords are int

        # Get bounding box by simply using min/max operators on the vertices
        self.bbox = Bbox(np.array([np.min(vertices, axis=0), np.max(vertices, axis=0)]))

        # Ordering the coordinates of the trapezoid. Details on this algorithm e.g. here:
        # https://pavcreations.com/clockwise-and-counterclockwise-sorting-of-coordinates/
        # https://pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
        sum_vertices = self.vertices.sum(axis=1)
        self.top_left = self.vertices[np.argmin(sum_vertices)]  # top left = smallest sum
        self.bottom_right = self.vertices[np.argmax(sum_vertices)]  # bottom right = largest sum
        diff = np.diff(self.vertices, axis=1)
        self.top_right = self.vertices[np.argmin(diff)]  # top right = smallest difference
        self.bottom_left = self.vertices[np.argmax(diff)]  # bottom left = largest difference

        # Get sub image (bounding box of trapezoid
        self.img = img.getSubImage(self.bbox)
        self.removeOutsideArea()

        self.width_bottom = self.bottom_right[0] - self.bottom_left[0]
        self.width_top = self.top_right[0] - self.top_left[0]
        self.width = max(self.bottom_right[0], self.top_right[0]) - min(self.bottom_left[0], self.top_left[0])

        self.height_left = self.bottom_left[1] - self.top_left[1]
        self.height_right = self.bottom_right[1] - self.top_right[1]
        self.height = max(self.bottom_left[1], self.bottom_right[1]) - min(self.top_left[1], self.top_right[1])

        mean_height = (self.height_right + self.height_left)/2

    def removeOutsideArea(self):
        ## Setting outer pixels to zero by masking. Calculate mask image ...
        self.mask = np.zeros(self.img.data.shape)

        # Check whether a point is inside the trapezoid:
        # https://math.stackexchange.com/questions/757591/how-to-determine-the-side-on-which-a-point-lies
        for yi, y in self.bbox.enumerateY():
            for xi, x in self.bbox.enumerateX():
                if ((check_point(x, y, self.top_left, self.bottom_left) <= 0) and
                        (check_point(x, y, self.top_right, self.bottom_right) >= 0) and
                        (check_point(x, y, self.top_left, self.top_right) >= 0) and
                        (check_point(x, y, self.bottom_left, self.bottom_right) < 0)):
                    self.mask[yi][xi] = 1

        # multiply image with mask image
        masked_data = self.img.data * self.mask
        self.img.data = masked_data

    def plotBorder(self, axis):
        x, y = self.vertices[:,0], self.vertices[:, 1]
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        axis.plot(x, y, color='red', linewidth=3)

    def getMaskData(self):
        return self.mask

    def saveToTar(self, tar, fname_prefix):
        # create a temporary param set for easy storage to tar
        params = ParamGroup('roi', "")
        params.addParam(key='id', name="", default="TrapezoidRoi")
        # add vertices to params
        params.addParam(key='num_vertices', name="", default=len(self.vertices))
        for idx, v in enumerate(self.vertices):
            params.addParam(key="v%d_x" % (idx), name="", default=int(v[0]))
            params.addParam(key="v%d_y" % (idx), name="", default=int(v[1]))

        # save image parameter to tar
        params.writeToTar(tar, fname_prefix)

def saveROIListToTar(rois, tar, fname_prefix):
    for idx, roi in enumerate(rois):
        roi.saveToTar(tar, "%s_roi%d" % (fname_prefix, idx))

def loadROIListFromTar(tar, tar_content, fname_prefix, src_image):
    rois = []
    idx = 0
    while True:
        params = ParamGroup('roi', "")
        if not params.readFromTar(tar, tar_content,  "%s_roi%d" % (fname_prefix, idx), add_new=True):
            break

        id = params.getValue('id')
        if id == 'TrapezoidRoi':
            vertices = []
            for vidx in range(params.getValue('num_vertices')):
                vertices.append([params.getValue("v%d_x" % (vidx)), params.getValue("v%d_y" % (vidx))])
            rois.append(TrapezoidRoi(src_image, vertices))

        else:
            bbox = Bbox()
            bbox.readFromParams(params)
            if id == 'CircularRoi':
                rois.append(CircularRoi(src_image, bbox.getArray()))
            else:
                rois.append(RectangularRoi(src_image, bbox.getArray()))
        idx += 1

    return rois
