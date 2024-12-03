"""

"""

import numpy as np

from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QSizePolicy,
    QLabel,
    QComboBox,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter
from matplotlib.widgets import PolygonSelector, RectangleSelector, EllipseSelector
from mpl_toolkits import axes_grid1

import custom_colormap
from roi_definitions import RectangularRoi, CircularRoi, TrapezoidRoi
from dugr_image_io import *


class ImageView(QWidget):
    def __init__(self, **kwargs):
        super(ImageView, self).__init__()

        self.use_roi = kwargs.get('use_roi', False)

        self.status_bar = None

        self.logarithmic_scaling_flag = 'x4'
        self.image = None
        self.plot = None

        self.shape_selector = None
        self.click = [None, None]
        self.release = [None, None]
        self.vertices = [[None, None]] * 4
        self.roi_shape_flag = "Trapezoid"
        self.roi_save_callback = None
        self.roi_delete_last_callback = None

        layout = QVBoxLayout()
        self.setLayout(layout)

        toolbar = QWidget()
        layout_toolbar = QHBoxLayout()
        toolbar.setLayout(layout_toolbar)

        ## Logarithmic Scaling
        label = QLabel("Logarithmic Scaling")
        label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        layout_toolbar.addWidget(label)
        cb = QComboBox()
        cb.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        cb.addItems(["x2", "x3", "x4", "x5", "x6", "x7"])
        cb.setCurrentText("x4")
        cb.currentTextChanged.connect(self.on_logarithmic_scaling_change)
        layout_toolbar.addWidget(cb)

        ## ROI
        if self.use_roi:
            # ROI Shape
            label = QLabel("ROI Shape")
            layout_toolbar.addWidget(label)
            label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            cb = QComboBox()
            cb.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            cb.addItems(["Trapezoid", "Rectangular", "Circular"])
            cb.currentTextChanged.connect(self.on_roi_shape_change)
            layout_toolbar.addWidget(cb)
            # Save ROI
            bt = QPushButton("Save ROI")
            bt.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            bt.clicked.connect(self.on_safe_roi_click)
            layout_toolbar.addWidget(bt)
            # Delete last ROI
            bt = QPushButton("Delete last ROI")
            bt.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            bt.clicked.connect(self.on_delete_last_roi)
            layout_toolbar.addWidget(bt)


        ##
        spacer = QWidget()
        layout_toolbar.addWidget(spacer)
        #spacer = QSpacerItem(10, 10, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        #layout_toolbar.addSpacerItem(spacer)

        # self.figure = FigureCanvas(Figure(figsize=(12, 8), layout='tight'))
        self.figure = FigureCanvas(Figure(figsize=(12, 8), layout = 'constrained'))
        figure_toolbar = NavigationToolbar2QT(self.figure, self)

        layout.addWidget(toolbar)
        layout.addWidget(figure_toolbar)
        layout.addWidget(self.figure)

        shortcut = QShortcut(QKeySequence("Escape"), self)
        shortcut.activated.connect(self.clear_mpl_selection)


    def setStatusBar(self, status_bar):
        self.status_bar = status_bar

    def statusBarMessage(self, msg):
        if self.status_bar:
            self.status_bar.showMessage(msg)

    def setROISaveCallback(self, roi_save_callback):
        self.roi_save_callback = roi_save_callback

    def _getROI(self):
        roi = None
        if self.roi_shape_flag == "Rectangular":
            if self.click != [None, None]:
                roi = RectangularRoi(self.image, np.array([self.click, self.release]))
        elif self.roi_shape_flag == "Circular":
            if self.click != [None, None]:
                roi = CircularRoi(self.image, np.array([self.click, self.release]))
        elif self.roi_shape_flag == "Trapezoid":
            if np.any(self.vertices):
                roi = TrapezoidRoi(self.image, self.vertices)
        return roi

    def on_safe_roi_click(self):
        if not hasattr(self, 'image'):
            return

        roi = self._getROI()
        if not roi:
            self.statusBarMessage("No ROI!")
            return

        if self.roi_save_callback:
            self.roi_save_callback(roi)

    def saveROIWhenExist(self):
        roi = self._getROI()
        if roi and self.roi_save_callback:
            self.roi_save_callback(roi)

    def setROIDeleteLastCallback(self, roi_delete_last_callback):
        self.roi_delete_last_callback = roi_delete_last_callback

    def on_delete_last_roi(self):
        if self.roi_delete_last_callback:
            self.roi_delete_last_callback()

    def on_logarithmic_scaling_change(self, scaling):
        self.logarithmic_scaling_flag = scaling
        if self.image is not None:
            vmin = np.max(self.image.data) / 10 ** int(self.logarithmic_scaling_flag[-1])
            self.plot.set_norm(LogNorm(vmin=vmin, vmax=np.max(self.image.data)))

            # set formatter again because the former instance seems to get removed when doing set_norm()
            formatter = myformatter(labelOnlyBase=False)
            self.colorbar.formatter = formatter

            self.figure.draw()

    def clear_mpl_selection(self):
        self.click = [None, None]
        self.release = [None, None]
        self.vertices = [[None, None]] * 4

        self.shape_selector.clear()

    def on_roi_shape_change(self, shape):
        if not self.image:
            return

        if self.shape_selector:
            self.clear_mpl_selection()
            self.shape_selector.set_active(False)
            self.shape_selector.set_visible(False)
            self.shape_selector = None

        self.roi_shape_flag = shape
        if self.roi_shape_flag == "Rectangular":
            self.shape_selector = RectangleSelector(ax=self._ax, onselect=self.on_roi_select, useblit=True,
                                                    button=[1, 3], interactive=True, spancoords='pixels',
                                                    ignore_event_outside=True, drag_from_anywhere=True)
        elif self.roi_shape_flag == "Circular":
            self.shape_selector = EllipseSelector(ax=self._ax, onselect=self.on_roi_select, useblit=True,
                                                  button=[1, 3], interactive=True, spancoords='pixels',
                                                  ignore_event_outside=True, drag_from_anywhere=True)
        elif self.roi_shape_flag == "Trapezoid":
            self.shape_selector = PolygonSelector(ax=self._ax, onselect=self.on_poly_select, useblit=True,
                                                  props=dict(color='white', linestyle='-', linewidth=2, alpha=0.5))

    def on_roi_select(self, eclick, erelease):
        self.click[:] = round(eclick.xdata), round(eclick.ydata)
        self.release[:] = round(erelease.xdata), round(erelease.ydata)

    def on_poly_select(self, vertices):
        self.vertices = np.array(vertices)

    def setImage(self, image, rois=None):
        self.image = image

        vmax = self.image.getMax()
        self.vmin = vmax / 10 ** int(self.logarithmic_scaling_flag[-1])

        self.figure.figure.clf()
        self._ax = self.figure.figure.gca()

        self.plot = self._ax.imshow(self.image.data, norm=LogNorm(vmin=self.vmin, vmax=vmax),
                                               cmap=custom_colormap.ls_cmap, extent=self.image.getExtent())

        # The divider class gives the best results of placing the color bar next to the image and place both
        # with top left alignment.
        divider = axes_grid1.make_axes_locatable(self._ax)
        divider.set_anchor("NW")
        cax = divider.append_axes("right", size=0.3, pad=0.05)

        # create colorbar with own tick lable formatter
        formatter = myformatter(labelOnlyBase=False)
        self.colorbar = self.figure.figure.colorbar(self.plot, cax=cax, label="cd/m²", format=formatter)


        if rois != None:
            for roi in rois:
                roi.plotBorder(self._ax)

        self.figure.draw()

        if self.use_roi:
            self.on_roi_shape_change(self.roi_shape_flag)

    def getImage(self):
        return self.image

    def clear(self):
        if self.shape_selector:
            self.clear_mpl_selection()
            self.shape_selector.set_active(False)
            self.shape_selector.set_visible(False)
            self.shape_selector = None

        self.figure.figure.clf()
        self.figure.draw()
        self.image = None

    def saveToTar(self, tar, fname_prefix):
        if self.image is None:
            return

        self.image.saveToTar(tar, fname_prefix)

    def loadFromTar(self, tar, tar_content, fname_prefix):
        if self.image is not None:
            self.clear()

        image = DUGRImage()
        if not image.loadFromTar(tar, tar_content, fname_prefix):
            return False

        self.setImage(image)

        return True


class myformatter(LogFormatter):

    def _num_to_string(self, x, vmin, vmax):
        if x > 1000000:
            s = '%g' % x
        elif x > 10000:
            s = '%.0f' % x
        else:
            s = '%g' % x
        return s
