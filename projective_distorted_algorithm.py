"""
Script that contains the functionality for the DUGR calculation approach without projective correction
"""

import numpy as np
import pandas as pd
import dugr_image_io
import dugr_image_processing
import custom_colormap

from roi_definitions import RectangularRoi, CircularRoi, TrapezoidRoi
from os.path import exists
from math import pi, log
from matplotlib.colors import LogNorm
from json import load, dump

# Changed "Import Figure Canvas" to import "FigureCanvasQTAgg as Figure Canvas" -> Undo if this raises errors
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.widgets import PolygonSelector, RectangleSelector, EllipseSelector

from PyQt6.QtGui import QKeySequence, QShortcut
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


class ProjectiveDistUi(QWidget):
    def __init__(self, parent=None):
        super(ProjectiveDistUi, self).__init__(parent)
        layout = QHBoxLayout()
        layout2 = QVBoxLayout()
        layout3 = QVBoxLayout()
        layout.addLayout(layout2)
        layout.addLayout(layout3)

        # Parameters
        self.lum_th = 500
        self.d = 12
        self.focal_length = 0.0
        self.pixel_size = 0.0
        self.viewing_angle = 0.0
        self.viewing_distance = 0.0
        self.optical_resolution = 0.0
        self.luminous_intensity = 0.0
        self.luminaire_height = 0.0
        self.luminaire_width = 0.0

        self.rb_min = 0.0
        self.ro_min = 0.0
        self.fwhm = 0.0
        self.sigma = 0.0
        self.filter_width = 0.0
        self.DUGR_I = 0.0
        self.DUGR_L = 0.0
        self.k_square_I = 0.0
        self.k_square_L = 0.0
        self.l_b = 0.0
        self.l_eff = 0.0
        self.l_s = 0.0
        self.p_i = 0.0
        self.solid_angle_eff = 0.0
        self.omega_l = 0.0
        self.A_new = 0.0
        self.A_p = 0.0

        self.click = [None, None]
        self.release = [None, None]
        self.poly_edge_points = np.zeros((4, 2), dtype="float32")
        self.vertices = [
            [None, None],
            [None, None],
            [None, None],
            [None, None]
        ]

        self.logarithmic_scaling_flag = 'x4'
        self.vmin = 0
        self.df = None

        self.rois = []
        self.roi_shape_flag = "Trapezoid"
        self.filter_only_roi_flag = False

        self.source_image = None
        self.src_plot = None
        self._roi_axs = None
        self._filtered_image_ax = None
        self.filtered_image_plot = None
        self._binarized_image_ax = None
        self._result_ax = None
        self.result_table = None

        self.binarized_image_plot = None
        self.filtered_image = None
        self.binarized_image = None

        self.A_eff = None
        self.A_p_new_I = None
        self.A_p_new_L = None
        self.A = None
        self.A_new_L = None

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

        # Button to load parameters from json file
        self.load_parameter_button = QPushButton("Load Parameters", self)
        layout2.addWidget(self.load_parameter_button)
        self.load_parameter_button.clicked.connect(self.on_load_parameter_click)
        self.load_file_shortcut = QShortcut(QKeySequence("Ctrl+P"), self)
        self.load_file_shortcut.activated.connect(self.on_load_parameter_click)

        # Safe parameters to json file
        self.safe_parameter_button = QPushButton("Safe Parameters", self)
        layout2.addWidget(self.safe_parameter_button)
        self.safe_parameter_button.clicked.connect(self.on_safe_parameter_click)

        # Checkbox to switch between projective corrected or distorted algorithm
        self.check_box_proj_corr = QCheckBox("Projective correction", self)
        layout2.addWidget(self.check_box_proj_corr)

        # Checkbox to filter only ROI around
        self.checkbox_filter_only_roi = QCheckBox("Filter only ROI")
        layout2.addWidget(self.checkbox_filter_only_roi)
        self.checkbox_filter_only_roi.stateChanged.connect(self.on_filter_only_roi_change)

        # Luminance Threshold Label + Line Box
        luminance_threshold_label = QLabel("Luminance Threshold [cd/m^2]")
        layout2.addWidget(luminance_threshold_label)
        self.luminance_threshold_line_box = QLineEdit()
        layout2.addWidget(self.luminance_threshold_line_box)
        self.luminance_threshold_line_box.setText(str(self.lum_th))
        self.luminance_threshold_line_box.textChanged.connect(self.on_luminance_threshold_change)

        # Focal Length Label + Line Box
        focal_length_label = QLabel("Focal Length [mm]")
        layout2.addWidget(focal_length_label)
        self.focal_length_line_box = QLineEdit()
        layout2.addWidget(self.focal_length_line_box)
        self.focal_length_line_box.setText(str(self.focal_length))
        self.focal_length_line_box.textChanged.connect(self.on_focal_length_change)

        # Pixel Size Label + Line Box
        pixel_size_label = QLabel("Pixel Size [mm]")
        layout2.addWidget(pixel_size_label)
        self.pixel_size_line_box = QLineEdit()
        layout2.addWidget(self.pixel_size_line_box)
        self.pixel_size_line_box.setText(str(self.pixel_size))
        self.pixel_size_line_box.textChanged.connect(self.on_pixel_size_change)

        # Worst case resolution d Label + Line Box
        d_label = QLabel('"Eye Resolution: d [mm]"')
        layout2.addWidget(d_label)
        self.d_line_box = QLineEdit()
        layout2.addWidget(self.d_line_box)
        self.d_line_box.setText(str(self.d))
        self.d_line_box.textChanged.connect(self.on_worst_case_resolution_change)

        # Viewing angle Label + Line Box
        viewing_angle_label = QLabel("Measurement angle \u03B1E [°]")
        layout2.addWidget(viewing_angle_label)
        self.viewing_angle_line_box = QLineEdit()
        layout2.addWidget(self.viewing_angle_line_box)
        self.viewing_angle_line_box.setText(str(self.viewing_angle))
        self.viewing_angle_line_box.textChanged.connect(self.on_viewing_angle_change)

        # Viewing distance Label + Line Box
        viewing_distance_label = QLabel("Viewing distance [mm]")
        layout2.addWidget(viewing_distance_label)
        self.viewing_distance_line_box = QLineEdit()
        layout2.addWidget(self.viewing_distance_line_box)
        self.viewing_distance_line_box.setText(str(self.viewing_distance))
        self.viewing_distance_line_box.textChanged.connect(self.on_viewing_distance_change)

        # Luminous Intensity Label + Line Box
        luminous_intensity_label = QLabel("Luminous Intensity I [cd]")
        layout2.addWidget(luminous_intensity_label)
        self.luminous_intensity_line_box = QLineEdit()
        layout2.addWidget(self.luminous_intensity_line_box)
        self.luminous_intensity_line_box.setText(str(self.luminous_intensity))
        self.luminous_intensity_line_box.textChanged.connect(self.on_luminous_intensity_change)

        # Luminaire width Label + Line Box
        luminaire_width_label = QLabel("Luminous area width [mm]")
        layout2.addWidget(luminaire_width_label)
        self.luminaire_width_line_box = QLineEdit()
        layout2.addWidget(self.luminaire_width_line_box)
        self.luminaire_width_line_box.setText(str(self.luminaire_width))
        self.luminaire_width_line_box.textChanged.connect(self.on_luminaire_width_change)

        # Luminaire height Label + Line Box
        luminaire_height_label = QLabel("Luminous area height [mm]")
        layout2.addWidget(luminaire_height_label)
        self.luminaire_height_line_box = QLineEdit()
        layout2.addWidget(self.luminaire_height_line_box)
        self.luminaire_height_line_box.setText(str(self.luminaire_height))
        self.luminaire_height_line_box.textChanged.connect(self.on_luminaire_height_change)

        layout2.addStretch()

        # Label and Combo Box for ROI shapes
        roi_shape_label = QLabel("ROI Shape")
        layout2.addWidget(roi_shape_label)
        self.roi_shape = QComboBox()
        self.roi_shape.addItems(["Trapezoid", "Rectangular", "Circular"])
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

        layout2.addStretch()

        self.calculate_dugr_button = QPushButton("Calculate DUGR", self)
        layout2.addWidget(self.calculate_dugr_button)
        self.calculate_dugr_button.clicked.connect(self.on_calculate_dugr_click)

        layout2.addStretch()

        # Create Tabs on the UI and Matplotlib Figures on them
        self.source_figure_tab = QWidget()
        self.tab_layout = QVBoxLayout()
        self.source_figure_tab.setLayout(self.tab_layout)
        self.source_figure = FigureCanvas(Figure(figsize=(12, 8), layout='tight'))
        mpl_source_figure_toolbar = NavigationToolbar2QT(self.source_figure, self)
        self.tab_layout.addWidget(mpl_source_figure_toolbar)
        self.tab_layout.addWidget(self.source_figure)

        # Regions of interest Tab
        self.roi_tab = QWidget()
        self.roi_tab_layout = QVBoxLayout()
        self.roi_tab.setLayout(self.roi_tab_layout)
        self.roi_figure = FigureCanvas(Figure(figsize=(12, 8), layout='tight'))
        mpl_roi_toolbar = NavigationToolbar2QT(self.roi_figure, self)
        self.roi_tab_layout.addWidget(mpl_roi_toolbar)
        self.roi_tab_layout.addWidget(self.roi_figure)

        # Filtered image Tab
        self.filtered_image_tab = QWidget()
        self.filtered_image_tab_layout = QVBoxLayout()
        self.filtered_image_tab.setLayout(self.filtered_image_tab_layout)
        self.filtered_image_figure = FigureCanvas(Figure(figsize=(12, 8), layout='tight'))
        mpl_filtered_image_toolbar = NavigationToolbar2QT(self.filtered_image_figure, self)
        self.filtered_image_tab_layout.addWidget(mpl_filtered_image_toolbar)
        self.filtered_image_tab_layout.addWidget(self.filtered_image_figure)

        # Binarized image Tab
        self.binarized_image_tab = QWidget()
        self.binarized_image_tab_layout = QVBoxLayout()
        self.binarized_image_tab.setLayout(self.binarized_image_tab_layout)
        self.binarized_image_figure = FigureCanvas(Figure(figsize=(12, 8), layout='tight'))
        mpl_binarized_image_toolbar = NavigationToolbar2QT(self.binarized_image_figure, self)
        self.binarized_image_tab_layout.addWidget(mpl_binarized_image_toolbar)
        self.binarized_image_tab_layout.addWidget(self.binarized_image_figure)

        self.result_figure_tab = QWidget()
        self.result_tab_layout = QVBoxLayout()
        self.result_tab_layout_h = QHBoxLayout()
        self.result_figure_tab.setLayout(self.result_tab_layout)
        self.result_figure = FigureCanvas(Figure(figsize=(12, 8), layout='tight'))
        self.export_protocol_button = QPushButton("Export protocol", self)
        self.export_to_json_button = QPushButton("Export to *.json", self)
        self.load_result_button = QPushButton("Load result data (*.json)", self)
        self.result_tab_layout.addWidget(self.result_figure)
        self.result_tab_layout.addLayout(self.result_tab_layout_h)
        self.result_tab_layout_h.addWidget(self.export_protocol_button)
        self.result_tab_layout_h.addWidget(self.export_to_json_button)
        self.export_protocol_button.clicked.connect(self.on_export_protocol_click)
        self.export_to_json_button.clicked.connect(self.on_export_to_json_click)
        self.result_tab_layout_h.addStretch()
        self.result_tab_layout_h.addWidget(self.load_result_button)
        self.load_result_button.clicked.connect(self.on_load_result_click)
        self.export_protocol_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        self.export_protocol_shortcut.activated.connect(self.on_export_protocol_click)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.source_figure_tab, "Source")
        self.tabs.addTab(self.roi_tab, "ROI")
        self.tabs.addTab(self.filtered_image_tab, "Filtered Image")
        self.tabs.addTab(self.binarized_image_tab, "Binarized Image")
        self.tabs.addTab(self.result_figure_tab, "Result")
        layout3.addWidget(self.tabs)

        self.clear_mpl_selection_shortcut = QShortcut(QKeySequence("Escape"), self)
        self.clear_mpl_selection_shortcut.activated.connect(self.clear_mpl_selection)

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
                    return

                self.source_figure.figure.clf()
                self._source_ax = self.source_figure.figure.subplots()
                self.src_plot = self._source_ax.imshow(self.source_image,
                                                       norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                       cmap=custom_colormap.ls_cmap)
                self.source_figure.figure.colorbar(self.src_plot, ax=self._source_ax, fraction=0.04, pad=0.035,
                                                   label="cd/m^2")
                self.source_figure.draw()
                self.shape_selector = PolygonSelector(ax=self._source_ax, onselect=self.on_poly_select, useblit=True,
                                                      props=dict(color='white', linestyle='-', linewidth=2, alpha=0.5))
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
                self.shape_selector = PolygonSelector(ax=self._source_ax, onselect=self.on_poly_select, useblit=True,
                                                      props=dict(color='white', linestyle='-', linewidth=2, alpha=0.5))
                self.status_bar.showMessage('File import successful')
        else:
            self.status_bar.showMessage('No File selected')

    def on_load_parameter_click(self):
        parameter_path = QFileDialog.getOpenFileName(self, "Choose file")[0]
        if exists(parameter_path):
            if parameter_path[-4:] != 'json':
                self.status_bar.showMessage('Parameter file type is invalid.\nMake sure to load a *.json File')
            else:
                with open(parameter_path) as f:
                    data = load(f)
                self.luminance_threshold_line_box.setText(str(data["lum_th"]))
                self.focal_length_line_box.setText(str(data["focal_length"]))
                self.pixel_size_line_box.setText(str(data["pixel_size"]))
                self.d_line_box.setText(str(data["d"]))
                self.viewing_angle_line_box.setText(str(data["viewing_angle"]))
                self.viewing_distance_line_box.setText(str(data["viewing_distance"]))
                self.luminaire_width_line_box.setText(str(data["luminaire_width"]))
                self.luminaire_height_line_box.setText(str(data["luminaire_height"]))

                self.status_bar.showMessage("Parameter import successfull")

    def on_safe_parameter_click(self):
        parameter_json_file = QFileDialog.getSaveFileName(self, "Export File", "", "*.json")[0]
        if parameter_json_file:
            data = {
                "lum_th": self.luminance_threshold_line_box.text(),
                "focal_length": self.focal_length_line_box.text(),
                "pixel_size": self.pixel_size_line_box.text(),
                "d": self.d_line_box.text(),
                "viewing_angle": self.viewing_angle_line_box.text(),
                "viewing_distance": self.viewing_distance_line_box.text(),
                "luminaire_width": self.luminaire_width_line_box.text(),
                "luminaire_height": self.luminaire_height_line_box.text()
            }
            with open(parameter_json_file, 'w') as f:
                dump(data, f)
        self.status_bar.showMessage("Parameter saved successfully")

    def on_filter_only_roi_change(self):
        if self.checkbox_filter_only_roi.isChecked():
            self.filter_only_roi_flag = True

        if not self.checkbox_filter_only_roi.isChecked():
            self.filter_only_roi_flag = False

    def on_roi_select(self, eclick, erelease):
        self.click[:] = round(eclick.xdata), round(eclick.ydata)
        self.release[:] = round(erelease.xdata), round(erelease.ydata)

    def on_poly_select(self, vertices):
        self.vertices = np.array(vertices)

    def clear_mpl_selection(self):
        if isinstance(self.shape_selector, RectangleSelector):
            self.click = [None, None]
            self.release = [None, None]
            self.shape_selector.set_active(False)
            self.shape_selector.set_visible(False)
            self.shape_selector.update()
            self.shape_selector.set_active(True)
            self.status_bar.showMessage("Rectangular ROI selection deleted")
        elif isinstance(self.shape_selector, PolygonSelector):
            self.shape_selector._xs, self.shape_selector._ys = [0], [0]
            self.shape_selector._selection_completed = False
            self.shape_selector.set_visible(True)
            self.status_bar.showMessage("Polygon selection deleted")

    def on_roi_shape_change(self, shape):
        self.roi_shape_flag = shape
        if self.roi_shape_flag == "Rectangular":
            self.shape_selector = RectangleSelector(ax=self._source_ax, onselect=self.on_roi_select, useblit=True,
                                                    button=[1, 3], interactive=True, spancoords='pixels')
        elif self.roi_shape_flag == "Circular":
            self.shape_selector = EllipseSelector(ax=self._source_ax, onselect=self.on_roi_select, useblit=True,
                                                  button=[1, 3], interactive=True, spancoords='pixels')
        elif self.roi_shape_flag == "Trapezoid":
            self.shape_selector = PolygonSelector(ax=self._source_ax, onselect=self.on_poly_select, useblit=True,
                                                  props=dict(color='white', linestyle='-', linewidth=2, alpha=0.5))

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
            self.shape_selector = PolygonSelector(ax=self._source_ax, onselect=self.on_poly_select, useblit=True,
                                                  props=dict(color='white', linestyle='-', linewidth=2, alpha=0.5))

    def on_safe_roi_click(self):
        if not hasattr(self, 'source_image'):
            self.status_bar.showMessage('In order to safe ROIs you need to open a source image first!')
            return
        if isinstance(self.shape_selector, PolygonSelector) and not np.any(self.vertices):
            self.status_bar.showMessage('In order to safe ROIs you need to draw them on the source image first!')
            return
        if (isinstance(self.shape_selector, EllipseSelector) or isinstance(self.shape_selector, RectangleSelector))\
                and self.click == [None, None]:
            self.status_bar.showMessage('In order to safe ROIs you need to draw them on the source image first!')
            return

        if self.roi_shape_flag == "Rectangular":
            ROI = RectangularRoi(self.source_image[self.click[1]:self.release[1], self.click[0]:self.release[0]],
                                 np.array([self.click, self.release]))
        elif self.roi_shape_flag == "Circular":
            ROI = CircularRoi(self.source_image[self.click[1]:self.release[1], self.click[0]:self.release[0]],
                              np.array([self.click, self.release]))
        elif self.roi_shape_flag == "Trapezoid":
            ROI = TrapezoidRoi(src_image=self.source_image, vertices=self.vertices)
        else:
            self.status_bar.showMessage("There was no ROI of an expected type detected."
                                        "The rectified image is used as ROI")
            ROI = self.source_image

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

            elif isinstance(self.rois[0], TrapezoidRoi):
                self._roi_axs = self.roi_figure.figure.subplots()
                self._roi_axs.plot(self.rois[0].d1_x, self.rois[0].d1_y, color='red', linewidth=3)
                self._roi_axs.plot(self.rois[0].d2_x, self.rois[0].d2_y, color='red', linewidth=3)
                self._roi_axs.plot(self.rois[0].d3_x, self.rois[0].d3_y, color='red', linewidth=3)
                self._roi_axs.plot(self.rois[0].d4_x, self.rois[0].d4_y, color='red', linewidth=3)
                roi_plot = self._roi_axs.imshow(self.rois[0].bounding_box,
                                                norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                cmap=custom_colormap.ls_cmap)
                self.roi_figure.figure.colorbar(roi_plot, ax=self._roi_axs, fraction=0.04, pad=0.035,
                                                label="cd/m^2")

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

                elif isinstance(self.rois[i], TrapezoidRoi):
                    roi_plot = self._roi_axs[i].imshow(self.rois[i].bounding_box,
                                                       norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                       cmap=custom_colormap.ls_cmap)
                    self.roi_figure.figure.colorbar(roi_plot, ax=self._roi_axs[i], fraction=0.04, pad=0.035,
                                                    label="cd/m^2")

            self.status_bar.showMessage("Successfully saved: " + str(len(self.rois)) + " ROIs")
        else:
            self.status_bar.showMessage("No ROI to safe selected")
        self.roi_figure.draw()

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

                elif isinstance(self.rois[0], TrapezoidRoi):
                    self._roi_axs = self.roi_figure.figure.subplots()
                    roi_plot = self._roi_axs.imshow(self.rois[0].bounding_box,
                                                    norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                    cmap=custom_colormap.ls_cmap)
                    self.roi_figure.figure.colorbar(roi_plot, ax=self._roi_axs, fraction=0.04, pad=0.035,
                                                    label="cd/m^2")

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
                    elif isinstance(self.rois[i], TrapezoidRoi):
                        roi_plot = self._roi_axs[i].imshow(self.rois[i].bounding_box,
                                                           norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                           cmap=custom_colormap.ls_cmap)
                        self.roi_figure.figure.colorbar(roi_plot, ax=self._roi_axs[i], fraction=0.04, pad=0.035,
                                                        label="cd/m^2")
            self.roi_figure.draw()
            self.status_bar.showMessage("Successfully deleted the last ROI.     " + str(len(self.rois)) + " remaining")
        else:
            self.status_bar.showMessage("All of the ROIs have already been removed, none left.")

    def on_luminance_threshold_change(self):
        userinput = self.luminance_threshold_line_box.text()
        try:
            self.lum_th = float(userinput)
            self.status_bar.showMessage("Luminance threshold changed successfully")
        except ValueError:
            self.status_bar.showMessage("Userinput to float conversion not possible")

    def on_focal_length_change(self):
        userinput = self.focal_length_line_box.text()
        try:
            self.focal_length = float(userinput)
            self.status_bar.showMessage("Focal Length changed successfully")
        except ValueError:
            self.status_bar.showMessage("Userinput to float conversion not possible")

    def on_pixel_size_change(self):
        userinput = self.pixel_size_line_box.text()
        try:
            self.pixel_size = float(userinput)
            self.status_bar.showMessage("Pixel Size changed successfully")
        except ValueError:
            self.status_bar.showMessage("Userinput to float conversion not possible")

    def on_worst_case_resolution_change(self):
        userinput = self.d_line_box.text()
        try:
            self.d = float(userinput)
            self.status_bar.showMessage("Worst case resolution changed successfully")
        except ValueError:
            self.status_bar.showMessage("Userinput to float conversion not possible")

    def on_luminous_intensity_change(self):
        userinput = self.luminous_intensity_line_box.text()
        try:
            self.luminous_intensity = float(userinput)
            self.status_bar.showMessage("Luminous intensity changed successfully")
        except ValueError:
            self.status_bar.showMessage("Userinput to float conversion not possible")

    def on_viewing_angle_change(self):
        userinput = self.viewing_angle_line_box.text()
        try:
            self.viewing_angle = float(userinput)
            self.status_bar.showMessage("Viewing angle changed successfully")
        except ValueError:
            self.status_bar.showMessage("Userinput to float conversion not possible")

    def on_viewing_distance_change(self):
        userinput = self.viewing_distance_line_box.text()
        try:
            self.viewing_distance = float(userinput)
            self.status_bar.showMessage("Viewing distance changed successfully")
        except ValueError:
            self.status_bar.showMessage("Userinput to float conversion not possible")

    def on_luminaire_width_change(self):
        userinput = self.luminaire_width_line_box.text()
        try:
            self.luminaire_width = float(userinput)
            self.status_bar.showMessage("luminaire width changed successfully")
        except ValueError:
            self.status_bar.showMessage("Userinput to float conversion not possible")

    def on_luminaire_height_change(self):
        userinput = self.luminaire_height_line_box.text()
        try:
            self.luminaire_height = float(userinput)
            self.status_bar.showMessage("Luminaire height changed successfully")
        except ValueError:
            self.status_bar.showMessage("Userinput to float conversion not possible")

    def on_calculate_dugr_click(self):
        if len(self.rois) == 0:
            self.status_bar.showMessage("No ROI found! Make sure to safe a ROI before executing the calculation")
            return

        self.DUGR_L, self.k_square_L, self.l_eff, self.l_s, self.solid_angle_eff, self.omega_l, \
            self.optical_resolution, self.rb_min, self.ro_min, self.fwhm, self.sigma, self.filter_width, \
            self.filtered_image, self.binarized_image = \
            dugr_image_processing.execute_projective_dist_algorithm(src_image=self.source_image,
                                                                    focal_length=self.focal_length,
                                                                    pixel_size=self.pixel_size,
                                                                    viewing_distance=self.viewing_distance,
                                                                    luminous_area_height=self.luminaire_height,
                                                                    viewing_angle=self.viewing_angle,
                                                                    rois=self.rois,
                                                                    filter_flag=self.filter_only_roi_flag)

        self.A_eff = self.solid_angle_eff * self.viewing_distance**2
        if self.roi_shape_flag != "Circular":
            self.A_p = (self.luminaire_height * self.luminaire_width) * np.cos(np.radians(90 - self.viewing_angle))
        else:
            self.A_p = (np.pi * (self.luminaire_height/2) * (self.luminaire_width/2))\
                       * np.cos(np.radians(90 - self.viewing_angle))

        self.A_p_new_I = (self.luminous_intensity ** 2) / ((self.l_eff * 10 ** -6) ** 2 * self.A_eff)
        self.A_p_new_L = self.A_p / self.k_square_L

        self.A_new_L = self.A_p_new_L / np.cos(np.radians(90 - self.viewing_angle))
        self.A = self.luminaire_width * self.luminaire_height

        if self.A_p_new_I != 0:
            self.k_square_I = self.A_p / self.A_p_new_I
            self.DUGR_I = 8 * log(self.k_square_I, 10)

            table_data = [
                ["DUGR_I", f"{self.DUGR_I:.1f}"],
                ["k^2_I", f"{self.k_square_I:.1f}"],
                ["A_p_new_I", f"{self.A_p_new_I:.0f} [mm^2]"],
                ["DUGR_L", f"{self.DUGR_L:.1f}"],
                ["k^2_L", f"{self.k_square_L:.1f}"],
                ["A_p_new_L", f"{self.A_p_new_L:.0f} [mm^2]"],
                ["A_p", f"{self.A_p:.0f} [mm^2]"],
                ["A_new_L", f"{self.A_new_L:.0f} [mm^2]"],
                ["A", f"{self.A:.0f} [mm^2]"],
                ["A_eff", f"{self.A_eff:.0f} [mm^2]"],
                ["L_eff", f"{self.l_eff:.2f} [cd/m^2]"],
                ["L_mean", f"{self.l_s:.2f} [cd/m^2]"],
                ["\u03C9_eff", f"{self.solid_angle_eff:.6f} [sr]"],
                ["\u03C9_luminaire", f"{self.omega_l:.6f} [sr]"],
                ["Measurement angle \u03B1E", f"{self.viewing_angle} [°]"],
                ["Measurement distance", f"{self.viewing_distance} [mm]"],
                ["Luminous area width", f"{self.luminaire_width} [mm]"],
                ["Luminous area height", f"{self.luminaire_height} [mm]"],
                ["I", f"{self.luminous_intensity:.1f} [cd]"],
                ["lum_th", f"{self.lum_th} [cd/m^2]"],
                ["d", f"{self.d} [mm]"],
                ["Calculated optical resolution", f"{self.optical_resolution:.5f} [°/px]"],
                ["FWHM", f"{self.fwhm:.2f} [px]"],
                ["Filter width", f"{self.filter_width} [px]"],
                ["Filter sigma", f"{self.sigma:.3f} [px]"],
                ["rb min", f"{self.rb_min:.2f} [mm]"],
                ["ro min", f"{self.ro_min:.5f} [°/px]"]
            ]

            data = {'Parameter': ['DUGR_I', 'k^2_I', 'A_p_new_I', 'DUGR_L', 'k^2_L', 'A_p_new_L', 'A_p', 'A_new_L', 'A',
                                  'A_eff', 'L_eff', 'L_mean', '\u03C9_eff', '\u03C9_luminaire',
                                  'Measurement angle \u03B1E', 'Measurement distance', 'Luminous area width',
                                  'Luminous area height', 'I', 'lum_th', 'd', 'Calculated optical resolution', 'FWHM',
                                  'Filter_width', 'Filter \u03C3', 'rb_min', 'ro_min'],
                    'Value': [self.DUGR_I, self.k_square_I, self.A_p_new_I, self.DUGR_L, self.k_square_L,
                              self.A_p_new_L, self.A_p, self.A_new_L, self.A, self.A_eff, self.l_eff, self.l_s,
                              self.solid_angle_eff, self.omega_l, self.viewing_angle, self.viewing_distance,
                              self.luminaire_width, self.luminaire_height, self.luminous_intensity, self.lum_th, self.d,
                              self.optical_resolution, self.fwhm, self.filter_width, self.sigma, self.rb_min,
                              self.ro_min],
                    'Unit': ['None', 'None', 'mm^2', 'None', 'None', 'mm^2', 'mm^2', 'mm^2', 'mm^2', 'mm^2', 'cd/m^2',
                             'cd/m^2', 'sr', 'sr', '°', 'mm', 'mm', 'mm', 'cd', 'cd/m^2', 'mm', '°/px', 'px', 'px',
                             'px', 'mm', '°/px']}

            self.df = pd.DataFrame(data)
        else:
            table_data = [
                ["DUGR_L", f"{self.DUGR_L:.1f}"],
                ["k^2_L", f"{self.k_square_L:.1f}"],
                ["A_p_new_L", f"{self.A_p_new_L:.0f} [mm^2]"],
                ["A_p", f"{self.A_p:.0f} [mm^2]"],
                ["A_new_L", f"{self.A_new_L:.0f} [mm^2]"],
                ["A", f"{self.A:.0f} [mm^2]"],
                ["A_eff", f"{self.A_eff:.0f} [mm^2]"],
                ["Effective luminance", f"{self.l_eff:.2f} [cd/m^2]"],
                ["Mean Luminaire luminance", f"{self.l_s:.2f} [cd/m^2]"],
                ["Effective solid angle", f"{self.solid_angle_eff:.6f} [sr]"],
                ["Luminaire solid angle", f"{self.omega_l:.6f} [sr]"],
                ["Measurement angle \u03B1E", f"{self.viewing_angle} [°]"],
                ["Viewing distance", f"{self.viewing_distance} [mm]"],
                ["Luminous area width", f"{self.luminaire_width} [mm]"],
                ["Luminous area height", f"{self.luminaire_height} [mm]"],
                ["lum_th", f"{self.lum_th} [cd/m^2]"],
                ["d", f"{self.d} [mm]"],
                ["Calculated optical resolution", f"{self.optical_resolution:.5f} [°/px]"],
                ["FWHM", f"{self.fwhm:.2f} [px]"],
                ["Filter width", f"{self.filter_width} [px]"],
                ["Filter sigma", f"{self.sigma:.3f} [px]"],
                ["rb min", f"{self.rb_min:.2f} [mm]"],
                ["ro min", f"{self.ro_min:.5f} [°/px]"]
            ]

            data = {'Parameter': ['DUGR_L', 'k^2_L', 'A_p_new_L', 'A_p', 'A_new_L', 'A', 'A_eff', 'L_eff', 'L_mean',
                                  '\u03C9_eff', '\u03C9_luminaire', 'Measurement angle \u03B1E', 'Measurement distance',
                                  'Luminous area width', 'Luminous area height', 'lum_th', 'd',
                                  'Calculated optical resolution', 'FWHM', 'Filter_width', 'Filter \u03C3', 'rb_min',
                                  'ro_min'],
                    'Value': [self.DUGR_L, self.k_square_L, self.A_p_new_L, self.A_p, self.A_new_L, self.A, self.A_eff,
                              self.l_eff, self.l_s, self.solid_angle_eff, self.omega_l, self.viewing_angle,
                              self.viewing_distance, self.luminaire_width, self.luminaire_height, self.lum_th, self.d,
                              self.optical_resolution, self.fwhm, self.filter_width, self.sigma, self.rb_min,
                              self.ro_min],
                    'Unit': ['None', 'None', 'mm^2', 'mm^2', 'mm^2', 'mm^2', 'mm^2', 'cd/m^2', 'cd/m^2', 'sr', 'sr',
                             '°', 'mm', 'mm', 'mm', 'cd/m^2', 'mm', '°/px', 'px', 'px', 'px', 'mm', '°/px']}

            self.df = pd.DataFrame(data)

        self.filtered_image_figure.figure.clf()
        self._filtered_image_ax = self.filtered_image_figure.figure.subplots(len(self.filtered_image))

        if len(self.filtered_image) == 1:
            self.filtered_image_plot = self._filtered_image_ax.imshow(self.filtered_image[0],
                                                                      norm=LogNorm(
                                                                          vmin=self.vmin,
                                                                          vmax=np.max(self.source_image)),
                                                                      cmap=custom_colormap.ls_cmap)
            self.filtered_image_figure.figure.colorbar(self.filtered_image_plot, ax=self._filtered_image_ax,
                                                       fraction=0.04, pad=0.035, label="[cd/m^2]")

        else:
            for i in range(len(self.filtered_image)):
                self.filtered_image_plot = self._filtered_image_ax[i].imshow(self.filtered_image[i],
                                                                             norm=LogNorm(
                                                                                vmin=self.vmin,
                                                                                vmax=np.max(self.source_image)),
                                                                             cmap=custom_colormap.ls_cmap)
                self.filtered_image_figure.figure.colorbar(self.filtered_image_plot, ax=self._filtered_image_ax[i],
                                                           fraction=0.04, pad=0.035, label="[cd/m^2]")

        self.filtered_image_figure.draw()

        self.binarized_image_figure.figure.clf()
        self._binarized_image_ax = self.binarized_image_figure.figure.subplots(len(self.binarized_image))

        if len(self.binarized_image) == 1:
            self.binarized_image_plot = self._binarized_image_ax.imshow(self.binarized_image[0],
                                                                        norm=LogNorm(
                                                                            vmin=self.vmin,
                                                                            vmax=np.max(self.source_image)),
                                                                        cmap=custom_colormap.ls_cmap)
            self.binarized_image_figure.figure.colorbar(self.binarized_image_plot, ax=self._binarized_image_ax,
                                                        fraction=0.04, pad=0.035, label="[cd/m^2]")

        else:
            for i in range(len(self.binarized_image)):
                self.binarized_image_plot = self._binarized_image_ax[i].imshow(self.binarized_image[i],
                                                                               norm=LogNorm(
                                                                                    vmin=self.vmin,
                                                                                    vmax=np.max(self.source_image)),
                                                                               cmap=custom_colormap.ls_cmap)
                self.binarized_image_figure.figure.colorbar(self.binarized_image_plot, ax=self._binarized_image_ax[i],
                                                            fraction=0.04, pad=0.035, label="[cd/m^2]")

        self.binarized_image_figure.draw()

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
            pdf.savefig(self.filtered_image_figure.figure)
            pdf.savefig(self.binarized_image_figure.figure)
            pdf.savefig(self.result_figure.figure)
            pdf.close()

    def on_export_to_json_click(self):
        json_file = QFileDialog.getSaveFileName(self, "Export File", "", "*.json")[0]
        if json_file:
            self.df.to_json(json_file)
            self.status_bar.showMessage('Export to *.json File successful')

    def on_load_result_click(self):
        result_path = QFileDialog.getOpenFileName(self, "Choose file")[0]
        if exists(result_path):
            if result_path[-4:] != 'json':
                self.status_bar.showMessage('Parameter file type is invalid.\nMake sure to load a *.json File')
            else:
                with open(result_path) as f:
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