"""
Script that contains the functionality for the DUGR calculation approach with projective correction
"""
import numpy as np
import pandas as pd
import dugr_image_io
import dugr_image_processing
import custom_colormap

from roi_definitions import RectangularRoi, CircularRoi
from os.path import exists
from math import pi, ceil, floor, log, sqrt
from matplotlib.colors import LogNorm
from json import load
from csv import reader

# Changed "Import Figure Canvas" to import "FigureCanvasQTAgg as Figure Canvas" -> Undo if this raises errors
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.widgets import PolygonSelector, RectangleSelector, EllipseSelector
from matplotlib.patches import Rectangle, Ellipse

from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QLineEdit,
    QLabel,
    QTabWidget,
    QComboBox,
    QCheckBox,
    QStatusBar,
)


class ProjectiveCorrUi(QWidget):
    def __init__(self, parent=None):
        super(ProjectiveCorrUi, self).__init__(parent)
        layout = QHBoxLayout()
        layout2 = QVBoxLayout()
        layout3 = QVBoxLayout()
        layout.addLayout(layout2)
        layout.addLayout(layout3)

        # Parameters
        self.lum_th = 500
        self.FWHM = 12
        self.rectification_width = 0
        self.rectification_height = 0
        self.luminous_area_width = 0
        self.luminous_area_height = 0
        self.measurement_distance = 0
        self.A = 0
        self.Ls = 0
        self.Aeff = 0
        self.Leff = 0
        self.k_square = 0
        self.A_new = 0
        self.DUGR = 0
        self.source_image = None
        self.rectified_image = None
        self.filtered_image = None
        self.binarize_image = None
        self.src_plot = None
        self.rect_plot = None
        self.click = [None, None]
        self.release = [None, None]
        self.vertices = [
            [None, None],
            [None, None],
            [None, None],
            [None, None]
        ]

        self.logarithmic_scaling_flag = 'x4'
        self.vmin = 0

        self.df = None

        self.ellipse_roi_data = []
        self.roi_shape_flag = "Rectangular"
        self.rois = []
        self.binarized_rois = []
        self.roi_coords = []
        self.projective_rect = np.zeros((4, 2), dtype="float32")
        self.R_mm = 1.0
        self.sigma = round((self.FWHM / 2.3548), 4)
        self.filter_width = 2 * ceil(3 * self.sigma) + 1
        self.border_size = floor(self.filter_width / 2)

        # Button to load an image file
        self.load_file_button = QPushButton("Open File", self)
        layout2.addWidget(self.load_file_button)
        self.load_file_button.clicked.connect(self.on_file_open_click)
        self.load_file_shortcut = QShortcut(QKeySequence("Ctrl+O"), self)
        self.load_file_shortcut.activated.connect(self.on_file_open_click)

        # Label and Combobox for logarithmic image scaling
        logarithmic_scaling_label = QLabel("Logarithmic Scaling")
        layout2.addWidget(logarithmic_scaling_label)
        self.logarithmic_scaling_box = QComboBox()
        self.logarithmic_scaling_box.addItems(["x2", "x3", "x4", "x5", "x6", "x7"])
        self.logarithmic_scaling_box.setCurrentText("x4")
        self.logarithmic_scaling_box.currentTextChanged.connect(self.on_logarithmic_scaling_change)
        layout2.addWidget(self.logarithmic_scaling_box)

        # Check box to switch between the UIs for projective distorted and corrected procedures
        self.check_box_proj_corr = QCheckBox('Projective correction', self)
        layout2.addWidget(self.check_box_proj_corr)
        self.check_box_proj_corr.setCheckState(Qt.CheckState.Checked)

        # Luminance Threshold Label + Line Box
        luminance_threshold_label = QLabel("Luminance Threshold [cd/m^2]")
        layout2.addWidget(luminance_threshold_label)
        self.luminance_threshold_line_box = QLineEdit()
        layout2.addWidget(self.luminance_threshold_line_box)
        self.luminance_threshold_line_box.setText(str(self.lum_th))
        self.luminance_threshold_line_box.textChanged.connect(self.on_luminance_threshold_change)

        # FWHM Label + Line Box
        FWHM_label = QLabel("Gauss Filter FWHM [px]")
        layout2.addWidget(FWHM_label)
        self.FWHM_line_box = QLineEdit()
        layout2.addWidget(self.FWHM_line_box)
        self.FWHM_line_box.setText(str(self.FWHM))
        self.FWHM_line_box.textChanged.connect(self.on_fwhm_change)

        # Rectification width Label + Line Box
        rectification_width_label = QLabel("Rectification Width [mm]")
        layout2.addWidget(rectification_width_label)
        self.rectification_width_line_box = QLineEdit()
        layout2.addWidget(self.rectification_width_line_box)
        self.rectification_width_line_box.setText(str(self.rectification_width))
        self.rectification_width_line_box.textChanged.connect(self.on_rectification_width_change)

        # Rectification height Label + Line Box
        rectification_height_label = QLabel("Rectification Height [mm]")
        layout2.addWidget(rectification_height_label)
        self.rectification_height_line_box = QLineEdit()
        layout2.addWidget(self.rectification_height_line_box)
        self.rectification_height_line_box.setText(str(self.rectification_height))
        self.rectification_height_line_box.textChanged.connect(self.on_rectification_height_change)

        # Checkbox to allow usage of the luminous area parameters
        self.check_box_use_luminous_area = QCheckBox("Use Luminous Area Parameters", self)
        layout2.addWidget(self.check_box_use_luminous_area)

        # Luminous area width label + Line Box
        luminous_area_width_label = QLabel("Luminous Area Width [mm]")
        layout2.addWidget(luminous_area_width_label)
        self.luminous_area_width_line_box = QLineEdit()
        layout2.addWidget(self.luminous_area_width_line_box)
        self.luminous_area_width_line_box.setText(str(self.luminous_area_width))
        self.luminous_area_width_line_box.textChanged.connect(self.on_luminous_area_width_change)

        # Luminous area height label + Line Box
        luminous_area_height_label = QLabel("Luminous Area Height [mm]")
        layout2.addWidget(luminous_area_height_label)
        self.luminous_area_height_line_box = QLineEdit()
        layout2.addWidget(self.luminous_area_height_line_box)
        self.luminous_area_height_line_box.setText(str(self.luminous_area_height))
        self.luminous_area_height_line_box.textChanged.connect(self.on_luminous_area_height_change)

        layout2.addStretch()

        # Button to start projective transformation (correction)
        button_projective_transformation = QPushButton("Projective Transformation")
        layout2.addWidget(button_projective_transformation)
        button_projective_transformation.clicked.connect(self.on_projective_transformation_click)

        layout2.addStretch()

        # Label and Combo Box for ROI shapes
        roi_shape_label = QLabel("ROI Shape")
        layout2.addWidget(roi_shape_label)
        self.roi_shape = QComboBox()
        self.roi_shape.addItems(["Rectangular", "Circular"])
        # self.roi_shape.addItems(["Rectangular", "Circular", "Polygonal"])
        self.roi_shape.currentTextChanged.connect(self.on_roi_shape_change)
        layout2.addWidget(self.roi_shape)

        # Button to safe the selected ROI
        button_safe_roi = QPushButton("Safe ROI")
        layout2.addWidget(button_safe_roi)
        button_safe_roi.clicked.connect(self.on_safe_roi_click)

        # Button to delete the last ROI selected
        button_delete_last_roi = QPushButton("Delete Last ROI")
        layout2.addWidget(button_delete_last_roi)
        button_delete_last_roi.clicked.connect(self.on_delete_last_roi)

        # Button to delete all of the ROIs
        button_delete_all_rois = QPushButton("Delete All ROIs")
        layout2.addWidget(button_delete_all_rois)
        button_delete_all_rois.clicked.connect(self.on_delete_all_rois)

        layout2.addStretch()

        # Button to perform gaussian filtering of the image
        button_filter_image = QPushButton("Filter Image")
        layout2.addWidget(button_filter_image)
        button_filter_image.clicked.connect(self.on_filter_image_click)

        layout2.addStretch()

        # Button to binarize the ROIs based on the threshold
        button_binarize = QPushButton("Binarize ROI(s)")
        layout2.addWidget((button_binarize))
        button_binarize.clicked.connect(self.on_binarize_click)

        # Create Tabs on the UI and Matplotlib Figures on them
        # Source Tab
        self.source_tab = QWidget()
        self.source_tab_layout = QVBoxLayout()
        self.source_tab_layout_h = QHBoxLayout()
        self.source_tab.setLayout(self.source_tab_layout)
        self.source_figure = FigureCanvas(Figure(figsize=(12, 8), layout='tight'))
        mpl_source_toolbar = NavigationToolbar2QT(self.source_figure, self)
        self.clear_poly_selection_button = QPushButton("Clear Polygon selection", self)
        self.clear_poly_selection_button.clicked.connect(self.callback_clear_poly_selection)
        self.source_tab_layout.addWidget(mpl_source_toolbar)
        self.source_tab_layout.addWidget(self.source_figure)
        self.source_tab_layout.addLayout(self.source_tab_layout_h)
        self.source_tab_layout_h.addWidget(self.clear_poly_selection_button)
        self.source_tab_layout_h.addStretch()

        # Projective Rectification Tab
        self.rectified_tab = QWidget()
        self.rectified_tab_layout = QVBoxLayout()
        self.rectified_tab_layout_h = QHBoxLayout()
        self.rectified_tab.setLayout(self.rectified_tab_layout)
        self.rectified_figure = FigureCanvas(Figure(figsize=(12, 8), layout='tight'))
        mpl_rectified_toolbar = NavigationToolbar2QT(self.rectified_figure, self)
        self.clear_roi_selection_button = QPushButton("Clear ROI selection", self)
        self.clear_roi_selection_button.clicked.connect(self.callback_clear_roi_selection)
        self.use_whole_image_button = QPushButton("Use the whole rectified image", self)
        self.use_whole_image_button.clicked.connect(self.on_use_whole_image_click)
        self.rectified_tab_layout.addWidget(mpl_rectified_toolbar)
        self.rectified_tab_layout.addWidget(self.rectified_figure)
        self.rectified_tab_layout.addLayout(self.rectified_tab_layout_h)
        self.rectified_tab_layout_h.addWidget(self.clear_roi_selection_button)
        self.rectified_tab_layout_h.addWidget(self.use_whole_image_button)
        self.rectified_tab_layout_h.addStretch()

        # Regions of interest Tab
        self.roi_tab = QWidget()
        self.roi_tab_layout = QVBoxLayout()
        self.roi_tab.setLayout(self.roi_tab_layout)
        self.roi_figure = FigureCanvas(Figure(figsize=(12, 8), layout='tight'))
        mpl_roi_toolbar = NavigationToolbar2QT(self.roi_figure, self)
        self.roi_tab_layout.addWidget(mpl_roi_toolbar)
        self.roi_tab_layout.addWidget(self.roi_figure)

        # Image filter tab
        self.filter_image_tab = QWidget()
        self.filter_image_tab_layout = QVBoxLayout()
        self.filter_image_tab.setLayout(self.filter_image_tab_layout)
        self.filter_image_figure = FigureCanvas(Figure(figsize=(12, 8), layout='tight'))
        mpl_filter_image_toolbar = NavigationToolbar2QT(self.filter_image_figure, self)
        self.filter_image_tab_layout.addWidget(mpl_filter_image_toolbar)
        self.filter_image_tab_layout.addWidget(self.filter_image_figure)

        # Binarization tab
        self.binarize_tab = QWidget()
        self.binarize_tab_layout = QVBoxLayout()
        self.binarize_tab.setLayout(self.binarize_tab_layout)
        self.binarize_figure = FigureCanvas(Figure(figsize=(12, 8), layout='tight'))
        mpl_binarize_toolbar = NavigationToolbar2QT(self.binarize_figure, self)
        self.binarize_tab_layout.addWidget(mpl_binarize_toolbar)
        self.binarize_tab_layout.addWidget(self.binarize_figure)

        # Result tab
        self.result_tab = QWidget()
        self.result_tab_layout = QVBoxLayout()
        self.result_tab_layout_h = QHBoxLayout()
        self.result_tab.setLayout(self.result_tab_layout)
        self.result_figure = FigureCanvas(Figure(figsize=(12, 8), layout='tight'))
        self.export_protocol_button = QPushButton("Export protocol", self)
        self.export_to_json_button = QPushButton("Export result to *.json", self)
        self.export_to_csv_button = QPushButton("Export result to *.csv")
        self.load_json_result_button = QPushButton("Load result data (*.json)", self)
        self.load_csv_result_button = QPushButton("Load result data (*.csv)", self)
        self.result_tab_layout.addWidget(self.result_figure)
        self.result_tab_layout.addLayout(self.result_tab_layout_h)
        self.result_tab_layout_h.addWidget(self.export_protocol_button)
        self.export_protocol_button.clicked.connect(self.on_export_protocol_click)
        self.result_tab_layout_h.addWidget(self.export_to_json_button)
        self.export_to_json_button.clicked.connect(self.on_export_to_json_click)
        self.result_tab_layout_h.addWidget(self.export_to_csv_button)
        self.export_to_csv_button.clicked.connect(self.on_export_to_csv_click)
        self.result_tab_layout_h.addStretch()
        self.result_tab_layout_h.addWidget(self.load_json_result_button)
        self.load_json_result_button.clicked.connect(self.on_load_json_result_click)
        self.result_tab_layout_h.addWidget(self.load_csv_result_button)
        self.load_csv_result_button.clicked.connect(self.on_load_csv_result_click)
        self.export_protocol_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        self.export_protocol_shortcut.activated.connect(self.on_export_protocol_click)

        self.clear_mpl_selection_shortcut = QShortcut(QKeySequence("Escape"), self)
        self.clear_mpl_selection_shortcut.activated.connect(self.clear_mpl_selection)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.source_tab, "Source")
        self.tabs.addTab(self.rectified_tab, "Rectified Image")
        self.tabs.addTab(self.roi_tab, "ROI")
        self.tabs.addTab(self.filter_image_tab, "Filtered Image")
        self.tabs.addTab(self.binarize_tab, 'Binarized Image')
        self.tabs.addTab(self.result_tab, "Result")
        layout3.addWidget(self.tabs)

        self._source_ax = self.source_figure.figure.subplots()
        self._rectified_ax = self.rectified_figure.figure.subplots()
        self._binarize_ax = self.binarize_figure.figure.subplots()
        self.poly = PolygonSelector(ax=self._source_ax, onselect=self.on_poly_select, useblit=True,
                                    props=dict(color='white', linestyle='-', linewidth=2, alpha=0.5))
        self.shape_selector = RectangleSelector(ax=self._rectified_ax, onselect=self.on_roi_select, useblit=True,
                                                button=[1, 3], interactive=True, spancoords='pixels')

        layout2.addStretch()

        # Button to calculate the DUGR value and print a result table on the result tab
        button_calculate_dugr = QPushButton("Calculate DUGR")
        layout2.addWidget(button_calculate_dugr)
        button_calculate_dugr.clicked.connect(self.on_calculate_dugr_click)

        layout2.addStretch()

        self.status_bar = QStatusBar(self)
        layout3.addWidget(self.status_bar)

        self.setLayout(layout)

    def on_file_open_click(self):

        image_path = QFileDialog.getOpenFileName(self, "Choose file")[0]

        if exists(image_path):
            if image_path[-2:] != 'pf' or image_path[-3:] != 'txt':
                self.status_bar.showMessage('File type is invalid.\nMake sure to load a *.pf  or *.txt File')
            if image_path[-2:] == 'pf':
                self.source_image, src_img_header = dugr_image_io.convert_tt_image_to_numpy_array(image_path)
                try:
                    self.vmin = np.max(self.source_image) / 10 ** int(self.logarithmic_scaling_flag[-1])
                except ValueError:
                    self.status_bar.showMessage('WARNING: The Image you want to load is in one of the supported file '
                                                'types, but the pixel information is not readable. '
                                                'Make sure the file is not corrupted')

                self.source_figure.figure.clf()
                self._source_ax = self.source_figure.figure.subplots()
                self.src_plot = self._source_ax.imshow(self.source_image,
                                                       norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                       cmap=custom_colormap.ls_cmap)
                self.source_figure.figure.colorbar(self.src_plot, ax=self._source_ax, fraction=0.04, pad=0.035,
                                                   label="cd/m^2")
                self.source_figure.draw()
                self.poly = PolygonSelector(ax=self._source_ax, onselect=self.on_poly_select, useblit=True,
                                            props=dict(color='white', linestyle='-', linewidth=2, alpha=0.5))
                self.shape_selector = RectangleSelector(ax=self._rectified_ax, onselect=self.on_roi_select,
                                                        useblit=True,
                                                        button=[1, 3], interactive=True, spancoords='pixels')
                self.status_bar.showMessage('File import successful')

            elif image_path[-3:] == "txt":
                self.source_image = dugr_image_io.convert_ascii_image_to_numpy_array(image_path)
                try:
                    self.vmin = np.max(self.source_image) / 10 ** int(self.logarithmic_scaling_flag[-1])
                except ValueError:
                    self.status_bar.showMessage('WARNING: The Image you want to load is in one of the supported file '
                                                'types, but the pixel information is not readable. '
                                                'Make sure the file is not corrupted')

                self.source_figure.figure.clf()
                self._source_ax = self.source_figure.figure.subplots()
                self.src_plot = self._source_ax.imshow(self.source_image,
                                                       norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                       cmap=custom_colormap.ls_cmap)
                self.source_figure.figure.colorbar(self.src_plot, ax=self._source_ax, fraction=0.04, pad=0.035,
                                                   label="cd/m^2")
                self.source_figure.draw()
                self.status_bar.showMessage('File import successful')
        else:
            self.status_bar.showMessage('No File selected')

    def on_poly_select(self, vertices):
        self.vertices = vertices
        vertices = np.array(vertices)

        sum_vertices = vertices.sum(axis=1)
        self.projective_rect[0] = self.vertices[np.argmin(sum_vertices)]  # top left = smallest sum
        self.projective_rect[2] = self.vertices[np.argmax(sum_vertices)]  # bottom right = largest sum

        diff = np.diff(vertices, axis=1)
        self.projective_rect[1] = self.vertices[np.argmin(diff)]  # top right = smallest difference
        self.projective_rect[3] = self.vertices[np.argmax(diff)]  # bottom left = largest difference

        edge_width_front = round(self.rectification_width / (self.projective_rect[2][0] - self.projective_rect[3][0]), 4)
        edge_width_back = round(self.rectification_width / (self.projective_rect[1][0] - self.projective_rect[0][0]), 4)

        cathete_left1 = self.projective_rect[3][1] - self.projective_rect[0][1]
        cathete_left2 = self.projective_rect[0][0] - self.projective_rect[3][0]
        edge_height_left = round(self.rectification_height / sqrt(cathete_left1 ** 2 + cathete_left2 ** 2), 4)

        cathete_right1 = self.projective_rect[2][1] - self.projective_rect[1][1]
        cathete_right2 = self.projective_rect[2][0] - self.projective_rect[1][0]
        edge_height_right = round(self.rectification_height / sqrt(cathete_right1 ** 2 + cathete_right2 ** 2), 4)

        if max(edge_width_front, edge_width_back, edge_height_left, edge_height_right) <= 1.2:
            self.status_bar.showMessage("Polygon selection successful     " +
                                        "Edge width front: " + str(edge_width_front) + "[mm/px]     " +
                                        "Edge width back: " + str(edge_width_back) + "[mm/px]     " +
                                        "Edge height left: " + str(edge_height_left) + "[mm/px]     " +
                                        "Edge height right: " + str(edge_height_right) + "[mm/px]")
        else:
            self.status_bar.showMessage("WARNING: The required measuring resolution of 1.2[mm/px] is not met")

    def on_roi_select(self, eclick, erelease):
        self.click[:] = round(eclick.xdata), round(eclick.ydata)
        self.release[:] = round(erelease.xdata), round(erelease.ydata)

    def clear_mpl_selection(self):
        if self.tabs.currentIndex() == 0:
            self.callback_clear_poly_selection()
        elif self.tabs.currentIndex() == 1:
            self.callback_clear_roi_selection()

    def callback_clear_poly_selection(self):
        self.poly._xs, self.poly._ys = [0], [0]
        self.poly._selection_completed = False
        self.poly.set_visible(True)
        self.status_bar.showMessage("Polygon selection deleted")

    def callback_clear_roi_selection(self):
        self.click = [None, None]
        self.release = [None, None]
        self.shape_selector.set_active(False)
        self.shape_selector.set_visible(False)
        self.shape_selector.update()
        self.shape_selector.set_active(True)
        self.status_bar.showMessage("ROI selection deleted")

    def on_projective_transformation_click(self):
        if self.source_image is None:
            self.status_bar.showMessage('In order to execute the projective correction a source image has to be opened'
                                        ' first')
            return
        if not np.any(self.projective_rect):
            self.status_bar.showMessage('In order to execute the projective correction the edge points have to be drawn'
                                        ' onto the source image first')
            return
        if self.rectification_width == 0:
            self.status_bar.showMessage('In order to execute the projective correction the luminaire width has to be'
                                        ' defined')
            return
        if self.rectification_height == 0:
            self.status_bar.showMessage('In order to execute the projective correction the luminaire height has to be'
                                        ' defined')
            return
        self.rectified_image = dugr_image_processing.projective_rectification(self.source_image,
                                                                              self.projective_rect,
                                                                              self.rectification_width,
                                                                              self.rectification_height)
        if self.rect_plot is None:
            self.rect_plot = self._rectified_ax.imshow(self.rectified_image,
                                                       norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                       cmap=custom_colormap.ls_cmap)
            self.rectified_figure.figure.colorbar(self.rect_plot, ax=self._rectified_ax, fraction=0.04, pad=0.035,
                                                  label="cd/m^2")
        else:
            self.rectified_figure.figure.clf()
            self.rect_plot = self._rectified_ax.imshow(self.rectified_image,
                                                       norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                       cmap=custom_colormap.ls_cmap)
            self.rectified_figure.figure.colorbar(self.rect_plot, ax=self._rectified_ax, fraction=0.04, pad=0.035,
                                                  label="cd/m^2")
        self.rectified_figure.draw()
        self.status_bar.showMessage("Projective Transformation successful")

    def on_use_whole_image_click(self):
        if self.rectified_image is not None:
            self.click = [0, 0]
            self.release = [self.rectified_image.shape[1], self.rectified_image.shape[0]]
            self.on_safe_roi_click()
        else:
            self.status_bar.showMessage("In order to use the whole rectified image, the rectification has to be "
                                        "executed first")

    def on_luminance_threshold_change(self):
        userinput = self.luminance_threshold_line_box.text()
        if userinput.isdigit():
            self.lum_th = int(userinput)
            self.status_bar.showMessage("Luminance threshold successfully changed to: " + str(self.lum_th))

    def on_fwhm_change(self):
        userinput = self.FWHM_line_box.text()
        if userinput.isdigit():
            self.FWHM = int(userinput)
            self.sigma = round((self.FWHM / 2.3548), 4)
            self.filter_width = 2 * ceil(3 * self.sigma) + 1
            self.border_size = floor(self.filter_width / 2)
            self.status_bar.showMessage("FWHM successfully changed to: " + str(self.FWHM) + ";     Sigma: " +
                                        str(self.sigma) + ";     Filter width: " +
                                        str(self.filter_width) + ";     Border size: " + str(self.border_size))

    def on_rectification_width_change(self):
        userinput = self.rectification_width_line_box.text()
        if userinput.isdigit():
            self.rectification_width = int(userinput)
            self.status_bar.showMessage("Luminaire width successfully changed to: " + str(self.rectification_width))

    def on_rectification_height_change(self):
        userinput = self.rectification_height_line_box.text()
        if userinput.isdigit():
            self.rectification_height = int(userinput)
            self.status_bar.showMessage("Luminaire height successfully changed to: " + str(self.rectification_height))

    def on_luminous_area_width_change(self):
        userinput = self.luminous_area_width_line_box.text()
        if userinput.isdigit():
            self.luminous_area_width = int(userinput)
            self.status_bar.showMessage("Luminaire width successfully changed to: " + str(self.luminous_area_width))

    def on_luminous_area_height_change(self):
        userinput = self.luminous_area_height_line_box.text()
        if userinput.isdigit():
            self.luminous_area_height = int(userinput)
            self.status_bar.showMessage("Luminaire height successfully changed to: " + str(self.luminous_area_height))

    def on_roi_shape_change(self, shape):
        self.roi_shape_flag = shape
        if self.roi_shape_flag == "Rectangular":
            self.shape_selector = RectangleSelector(ax=self._rectified_ax, onselect=self.on_roi_select, useblit=True,
                                                    button=[1, 3], interactive=True, spancoords='pixels')
        elif self.roi_shape_flag == "Circular":
            self.shape_selector = EllipseSelector(ax=self._rectified_ax, onselect=self.on_roi_select, useblit=True,
                                                  button=[1, 3], interactive=True, spancoords='pixels')

    def on_logarithmic_scaling_change(self, scaling):
        self.logarithmic_scaling_flag = scaling
        if self.source_image is not None:
            self.vmin = np.max(self.source_image) / 10 ** int(self.logarithmic_scaling_flag[-1])
            self.source_figure.figure.clf()
            self._source_ax = self.source_figure.figure.subplots()
            self.src_plot = self._source_ax.imshow(self.source_image,
                                                   norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                   cmap=custom_colormap.ls_cmap)
            self.source_figure.figure.colorbar(self.src_plot, ax=self._source_ax, fraction=0.04, pad=0.035,
                                               label="cd/m^2")
            self.source_figure.draw()
            self.poly = PolygonSelector(ax=self._source_ax, onselect=self.on_poly_select, useblit=True,
                                                  props=dict(color='white', linestyle='-', linewidth=2, alpha=0.5))

    def on_safe_roi_click(self):
        if self.rectified_image is None:
            self.status_bar.showMessage("In order to safe ROIs you have to execute the projective correction first")
            return
        if not np.any(self.click) and not np.any(self.release):
            self.status_bar.showMessage("In order to safe ROIs you have to draw them onto the projective corrected "
                                        "image first")
            return

        if self.roi_shape_flag == "Rectangular":
            ROI = RectangularRoi(self.rectified_image[self.click[1]:self.release[1], self.click[0]:self.release[0]],
                                 np.array([self.click, self.release]))
        elif self.roi_shape_flag == "Circular":
            ROI = CircularRoi(self.rectified_image[self.click[1]:self.release[1], self.click[0]:self.release[0]],
                              np.array([self.click, self.release]))
        else:
            self.status_bar.showMessage("There was no ROI of an expected type detected."
                                        "The rectified image is used as ROI")
            ROI = self.rectified_image
        self.rois.append(ROI)

        if len(self.rois) == 1:
            if isinstance(self.rois[0], RectangularRoi):
                self._roi_axs = self.roi_figure.figure.subplots()
                roi_plot = self._roi_axs.imshow(self.rois[0].roi_array,
                                                norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                cmap=custom_colormap.ls_cmap)
                self.roi_figure.figure.colorbar(roi_plot, ax=self._roi_axs, fraction=0.04, pad=0.035,
                                                label="cd/m^2")

            elif isinstance(self.rois[0], CircularRoi):
                self._roi_axs = self.roi_figure.figure.subplots()
                roi_plot = self._roi_axs.imshow(self.rois[0].bounding_box,
                                                norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                cmap=custom_colormap.ls_cmap)
                self.roi_figure.figure.colorbar(roi_plot, ax=self._roi_axs, fraction=0.04, pad=0.035,
                                                label="cd/m^2")

                t = np.linspace(0, 2 * pi, 100)
                self._roi_axs.plot(self.rois[0].width / 2 + (self.rois[0].width - 1) / 2 * np.cos(t),
                                   self.rois[0].height / 2 + (self.rois[0].height - 1) / 2 * np.sin(t),
                                   color='red')

            self.status_bar.showMessage("Successfully saved: 1 ROI")

        elif len(self.rois) > 1:

            self.roi_figure.figure.clf()

            self._roi_axs = self.roi_figure.figure.subplots(len(self.rois))
            for i in range(len(self.rois)):
                if isinstance(self.rois[i], RectangularRoi):
                    roi_plot = self._roi_axs[i].imshow(self.rois[i].roi_array,
                                                       norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                       cmap=custom_colormap.ls_cmap)
                    self.roi_figure.figure.colorbar(roi_plot, ax=self._roi_axs[i], fraction=0.04, pad=0.035,
                                                    label="cd/m^2")
                elif isinstance(self.rois[i], CircularRoi):
                    roi_plot = self._roi_axs[i].imshow(self.rois[i].bounding_box,
                                                       norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                       cmap=custom_colormap.ls_cmap)
                    self.roi_figure.figure.colorbar(roi_plot, ax=self._roi_axs[i], fraction=0.04, pad=0.035,
                                                    label="cd/m^2")

                    t = np.linspace(0, 2 * pi, 100)
                    self._roi_axs[i].plot(self.rois[i].width / 2 + (self.rois[i].width - 1) / 2 * np.cos(t),
                                          self.rois[i].height / 2 + (self.rois[i].height - 1) / 2 * np.sin(t),
                                          color='red')

            self.status_bar.showMessage("Successfully saved: " + str(len(self.rois)) + " ROIs")
        else:
            self.status_bar.showMessage("No ROI to safe selected")
        self.roi_figure.draw()
        self.callback_clear_roi_selection()

    def on_delete_last_roi(self):
        if len(self.rois) > 0:
            self.rois.pop()
            self.roi_figure.figure.clf()

            if len(self.rois) == 1:
                if isinstance(self.rois[0], RectangularRoi):
                    self._roi_axs = self.roi_figure.figure.subplots()
                    roi_plot = self._roi_axs.imshow(self.rois[0].roi_array,
                                                    norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                    cmap=custom_colormap.ls_cmap)
                    self.roi_figure.figure.colorbar(roi_plot, ax=self._roi_axs, fraction=0.04, pad=0.035,
                                                    label="cd/m^2")

                elif isinstance(self.rois[0], CircularRoi):
                    self._roi_axs = self.roi_figure.figure.subplots()
                    roi_plot = self._roi_axs.imshow(self.rois[0].bounding_box,
                                                    norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                    cmap=custom_colormap.ls_cmap)
                    self.roi_figure.figure.colorbar(roi_plot, ax=self._roi_axs, fraction=0.04, pad=0.035,
                                                    label="cd/m^2")

                    t = np.linspace(0, 2 * pi, 100)
                    self._roi_axs.plot(self.rois[0].width / 2 + (self.rois[0].width - 1) / 2 * np.cos(t),
                                       self.rois[0].height / 2 + (self.rois[0].height - 1) / 2 * np.sin(t),
                                       color='red')

            if len(self.rois) > 1:
                self._roi_axs = self.roi_figure.figure.subplots(len(self.rois))
                for i in range(len(self.rois)):
                    if isinstance(self.rois[i], RectangularRoi):
                        roi_plot = self._roi_axs[i].imshow(self.rois[i].roi_array,
                                                           norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                           cmap=custom_colormap.ls_cmap)
                        self.roi_figure.figure.colorbar(roi_plot, ax=self._roi_axs[i], fraction=0.04, pad=0.035,
                                                        label="cd/m^2")
                    elif isinstance(self.rois[i], CircularRoi):
                        roi_plot = self._roi_axs[i].imshow(self.rois[i].bounding_box,
                                                           norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                           cmap=custom_colormap.ls_cmap)
                        self.roi_figure.figure.colorbar(roi_plot, ax=self._roi_axs[i], fraction=0.04, pad=0.035,
                                                        label="cd/m^2")

                        t = np.linspace(0, 2 * pi, 100)
                        self._roi_axs[i].plot(self.rois[i].width / 2 + (self.rois[i].width - 1) / 2 * np.cos(t),
                                              self.rois[i].height / 2 + (self.rois[i].height - 1) / 2 * np.sin(t),
                                              color='red')

            self.roi_figure.draw()
            self.status_bar.showMessage("Successfully deleted the last ROI.     " + str(len(self.rois)) + " remaining")
        else:
            self.status_bar.showMessage("All of the ROIs have been removed successfully, none left.")

    def on_delete_all_rois(self):
        self.rois = []
        self.roi_coords = []
        self.roi_figure.figure.clf()
        self.roi_figure.draw()
        self.status_bar.showMessage("Successfully deleted all of the ROIs.")

    def on_filter_image_click(self):
        if len(self.rois) == 0:
            self.status_bar.showMessage("In order to filter the image you need to select ROIs first!")
            return
        self.on_poly_select(self.vertices)
        warped_border_image = dugr_image_processing.projective_rectification_with_borders(self.source_image,
                                                                                          self.projective_rect,
                                                                                          self.border_size,
                                                                                          self.rectification_width,
                                                                                          self.rectification_height)
        self.filtered_image = dugr_image_processing.filter_image(warped_border_image, self.filter_width, self.sigma)
        self.filter_image_figure.figure.clf()
        self._filter_ax = self.filter_image_figure.figure.subplots()
        filter_plot = self._filter_ax.imshow(self.filtered_image,
                                             norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                             cmap=custom_colormap.ls_cmap)

        self.roi_coords_with_border = []

        for i in range(len(self.rois)):
            if isinstance(self.rois[i], RectangularRoi):
                patch_anchor = (self.rois[i].top_left[0] + self.border_size,
                                self.rois[i].top_left[1] + self.border_size)
                ROI_draw = Rectangle(patch_anchor, self.rois[i].width, self.rois[i].height, linewidth=1,
                                     edgecolor='r', facecolor='none')
                self._filter_ax.add_patch(ROI_draw)

                y1 = self.rois[i].top_left[1] + self.border_size
                y2 = self.rois[i].bottom_left[1] + self.border_size
                x1 = self.rois[i].top_left[0] + self.border_size
                x2 = self.rois[i].top_right[0] + self.border_size

                bordered_roi = RectangularRoi(self.filtered_image[y1:y2, x1:x2],
                                              self.rois[i].roi_coordinates + self.border_size)

                self.roi_coords_with_border.append(bordered_roi)

            elif isinstance(self.rois[i], CircularRoi):
                patch_anchor = (self.rois[i].middle_point_coordinates[0] + self.border_size,
                                self.rois[i].middle_point_coordinates[1] + self.border_size)
                ROI_draw = Ellipse(patch_anchor, self.rois[i].width, self.rois[i].height, linewidth=1, edgecolor='r',
                                   facecolor='none')
                self._filter_ax.add_patch(ROI_draw)

                y1 = self.rois[i].top_left[1] + self.border_size
                y2 = self.rois[i].bottom_left[1] + self.border_size
                x1 = self.rois[i].top_left[0] + self.border_size
                x2 = self.rois[i].top_right[0] + self.border_size

                bordered_roi = CircularRoi(self.filtered_image[y1:y2, x1:x2],
                                           self.rois[i].bounding_box_coordinates + self.border_size)
                self.roi_coords_with_border.append(bordered_roi)

        self.filter_image_figure.figure.colorbar(filter_plot, ax=self._filter_ax, fraction=0.04, pad=0.035,
                                                 label="cd/m^2")
        self.filter_image_figure.draw()
        if len(self.filtered_image != 0):
            self.status_bar.showMessage("Image filtering successful with FWHM: " + str(self.FWHM) +
                                        "     Sigma: " + str(self.sigma) +
                                        "     Filter width: " + str(self.filter_width) +
                                        "     Border size: " + str(self.border_size))

    def on_binarize_click(self):
        if not hasattr(self, 'roi_coords_with_border'):
            self.status_bar.showMessage('In order to do binarization you need to filter the image first.')
            return
        self.binarized_rois = []
        for i in range(len(self.roi_coords_with_border)):
            if isinstance(self.roi_coords_with_border[i], RectangularRoi):
                binarize_roi = self.roi_coords_with_border[i].roi_array
                binarize_roi[binarize_roi < self.lum_th] = 0
                self.binarized_rois.append(binarize_roi)
            elif isinstance(self.roi_coords_with_border[i], CircularRoi):
                binarize_roi = self.roi_coords_with_border[i].bounding_box

                middle_point_x = self.roi_coords_with_border[i].middle_point_coordinates[0] - \
                                 self.roi_coords_with_border[i].bounding_box_coordinates[0][0]
                middle_point_y = self.roi_coords_with_border[i].middle_point_coordinates[1] - \
                                 self.roi_coords_with_border[i].bounding_box_coordinates[0][1]

                for y in range(binarize_roi.shape[0]):
                    for x in range(binarize_roi.shape[1]):
                        if ((((x - middle_point_x) ** 2) / ((self.roi_coords_with_border[i].width / 2) ** 2)) +
                           (((y - middle_point_y) ** 2) / (self.roi_coords_with_border[i].height / 2) ** 2)) >= 1:
                            binarize_roi[y][x] = 0
                binarize_roi[binarize_roi < self.lum_th] = 0
                self.binarized_rois.append(binarize_roi)

            if len(self.binarized_rois) == 1:
                self.binarize_figure.figure.clf()
                self._binarize_ax = self.binarize_figure.figure.subplots()
                binarize_plot = self._binarize_ax.imshow(self.binarized_rois[0],
                                                         norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                         cmap=custom_colormap.ls_cmap)
                self.binarize_figure.figure.colorbar(binarize_plot, ax=self._binarize_ax, fraction=0.04, pad=0.035,
                                                     label="cd/m^2")
                self.status_bar.showMessage("Binarization of 1 ROI successful")
            elif len(self.binarized_rois) > 1:
                self.binarize_figure.figure.clf()
                self._binarize_ax = self.binarize_figure.figure.subplots(len(self.binarized_rois))
                for i in range(len(self.binarized_rois)):
                    binarize_plot = self._binarize_ax[i].imshow(self.binarized_rois[i],
                                                                norm=LogNorm(
                                                                    vmin=self.vmin, vmax=np.max(self.source_image)),
                                                                cmap=custom_colormap.ls_cmap)
                    self.binarize_figure.figure.colorbar(binarize_plot, ax=self._binarize_ax[i], fraction=0.04,
                                                         pad=0.035, label="cd/m^2")
                self.status_bar.showMessage("Binarization of " + str(len(self.binarized_rois)) + " ROIs successful")
            elif len(self.binarized_rois) == 0:
                self.status_bar.showMessage("No ROI for the binarization found")
            self.binarize_figure.draw()

    def on_calculate_dugr_click(self):

        if len(self.binarized_rois) == 0:
            self.status_bar.showMessage("There is no binarized ROI, make sure to execute each step before starting"
                                        "the calculation.")
        else:
            self.A = 0
            self.Ls = 0

            for i in range(len(self.rois)):
                A_roi = self.rois[i].area
                L_roi = self.rois[i].mean_luminance

                self.Ls = self.Ls + L_roi
                self.A = round((self.A + A_roi), 2)
            self.Ls = round((self.Ls / len(self.rois)), 2)

            self.Aeff = 0
            self.Leff = 0
            for roi in self.binarized_rois:
                Aeff_roi = np.count_nonzero(roi) * self.R_mm
                Leff_roi = np.mean(roi[roi > 0])
                self.Aeff = round((self.Aeff + Aeff_roi), 2)
                self.Leff = self.Leff + Leff_roi
            self.Leff = round((self.Leff / len(self.binarized_rois)), 2)

            if self.check_box_use_luminous_area.isChecked():
                self.A = self.luminous_area_width * self.luminous_area_height
                self.rectification_width = self.luminous_area_width
                self.rectification_height = self.luminous_area_height

            self.k_square = round(((self.Leff ** 2 * self.Aeff) / (self.Ls ** 2 * self.A)), 4)

            self.A_new = round((self.A / self.k_square), 2)

            self.DUGR = round((8 * log(self.k_square, 10)), 4)

            table_data = [
                ["DUGR", str(self.DUGR)],
                ["A_new", str(self.A_new) + " [mm^2]"],
                ["k^2", str(self.k_square)],
                ["A_eff", str(self.Aeff) + " [mm^2]"],
                ["L_eff", str(self.Leff) + " [cd/m^2]"],
                ["A", str(self.A) + " [mm^2]"],
                ["Mean Luminaire luminance", str(self.Ls) + " [cd/m^2]"],
                ["lum_th", str(self.lum_th) + " [cd/m^2]"],
                ["FWHM", str(self.FWHM) + " [mm]"],
                ["Filter width", str(self.filter_width) + " [mm]"],
                ["Filter sigma", str(self.sigma) + " [mm]"],
                ["Border size", str(self.border_size) + " [mm]"],
                ["luminous area width", str(self.rectification_width) + " [mm]"],
                ["luminous area height", str(self.rectification_height) + " [mm]"],
            ]

            data = {'Parameter': ['DUGR', 'A_new', 'k^2', 'A_eff', 'L_eff', 'A', 'L_s', 'lum_th', 'FWHM',
                                  'Filter_width', 'Filter_sigma', 'Border_size', 'rectification_width', 'rectification_height'],

                    'Value': [self.DUGR, self.A_new, self.k_square, self.Aeff, self.Leff, self.A, self.Ls, self.lum_th,
                              self.FWHM, self.filter_width, self.sigma, self.border_size, self.rectification_width,
                              self.rectification_height],

                    'Unit': ['None', 'mm^2', 'None', 'mm^2', 'cd/m^2', 'mm^2', 'cd/m^2', 'cd/m^2', 'px', 'px', 'px',
                             'px', 'mm', 'mm']
                    }

            self.df = pd.DataFrame(data)

            self.result_figure.figure.clf()
            self._result_ax = self.result_figure.figure.subplots()
            self._result_ax.axis('off')
            self.result_table = self._result_ax.table(cellText=table_data, loc='center', cellLoc='center')
            self.result_table.set_fontsize(13)
            self.result_table.scale(1, 2)
            self.result_figure.draw()

            self.status_bar.showMessage("DUGR calculation successful")

    def on_export_protocol_click(self):
        protocol_file = QFileDialog.getSaveFileName(self, "Export File", "", "*.pdf")[0]
        if protocol_file:
            pdf = PdfPages(protocol_file)
            pdf.savefig(self.source_figure.figure)
            pdf.savefig(self.rectified_figure.figure)
            pdf.savefig(self.roi_figure.figure)
            pdf.savefig(self.filter_image_figure.figure)
            pdf.savefig(self.binarize_figure.figure)
            pdf.savefig(self.result_figure.figure)
            pdf.close()

    def on_export_to_json_click(self):
        json_file = QFileDialog.getSaveFileName(self, "Export File", "", "*.json")[0]
        if json_file:
            self.df.to_json(json_file)
            self.status_bar.showMessage('Export to *.json File successful')

    def on_export_to_csv_click(self):
        csv_file = QFileDialog.getSaveFileName(self, "Export File", "", "*.csv")[0]
        if csv_file:
            self.df.to_csv(csv_file, encoding='utf-8-sig')
            self.status_bar.showMessage('Export to *.csv File successful')

    def on_load_json_result_click(self):
        result_path = QFileDialog.getOpenFileName(self, "Choose file")[0]
        if exists(result_path):
            if result_path[-4:] != 'json':
                self.status_bar.showMessage('Parameter file type is invalid.\nMake sure to load a *.json File')
                return
            else:
                with open(result_path, encoding='utf-8-sig') as f:
                    data = load(f)
        table_data = []
        for i in data['Parameter']:
            table_data.append([data['Parameter'][i], data['Value'][i]])
        self.result_figure.figure.clf()
        self._result_ax = self.result_figure.figure.subplots()
        self._result_ax.axis('off')
        self.result_table = self._result_ax.table(cellText=table_data, loc='center', cellLoc='center')
        self.result_table.set_fontsize(13)
        self.result_table.scale(1, 2)
        self.result_figure.draw()

    def on_load_csv_result_click(self):
        result_path = QFileDialog.getOpenFileName(self, "Choose file")[0]
        if exists(result_path):
            if result_path[-3:] != 'csv':
                self.status_bar.showMessage('Parameter file type is invalid.\nMake sure to load a *.csv File')
                return
            else:
                table_data = []
                with open(result_path) as f:
                    csv_reader = reader(f)
                    for index, row in enumerate(csv_reader):
                        if index >= 1:
                            table_data.append(row[1:-1])
        self.result_figure.figure.clf()
        self._result_ax = self.result_figure.figure.subplots()
        self._result_ax.axis('off')
        self.result_table = self._result_ax.table(cellText=table_data, loc='center', cellLoc='center')
        self.result_table.set_fontsize(13)
        self.result_table.scale(1, 2)
        self.result_figure.draw()