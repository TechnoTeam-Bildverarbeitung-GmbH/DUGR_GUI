import sys
import numpy as np
import os.path
import math
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import dugr_image_io
import dugr_image_processing
import json
import pandas as pd

from matplotlib.pyplot import register_cmap
from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.widgets import PolygonSelector, RectangleSelector, EllipseSelector
from matplotlib.patches import Rectangle, Ellipse

from PyQt6.QtGui import QIcon, QKeySequence, QShortcut
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
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

basedir = os.path.dirname(__file__)

color_values = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
color_names = ["black", "blue", "lime", "red", "yellow", "white"]
color_map = list(zip(color_values, color_names))
ls_cmap = LinearSegmentedColormap.from_list('ls_cmap', color_map)
ls_cmap.set_bad((0, 0, 0))
register_cmap(cmap=ls_cmap)


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.resize(1400, 900)
        self.start_projective_distorted_ui()

        self.setWindowTitle("DUGR GUI")
        self.setWindowIcon(QIcon(os.path.join(basedir, "light-bulb.ico")))

        self.projective_dist_tab = None
        self.projective_corr_tab = None

    def start_projective_distorted_ui(self):
        self.projective_dist_tab = ProjectiveDistUi(self)
        self.setCentralWidget(self.projective_dist_tab)
        self.projective_dist_tab.check_box_proj_corr.stateChanged.connect(self.start_projective_corrected_ui)
        self.show()

    def start_projective_corrected_ui(self):
        self.projective_corr_tab = ProjectiveCorrUi(self)
        self.setCentralWidget(self.projective_corr_tab)
        self.projective_corr_tab.check_box_proj_corr.stateChanged.connect(self.start_projective_distorted_ui)
        self.show()


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

        self.src_plot = None
        self._roi_axs = None
        self._filtered_image_ax = None
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
        self.result_tab_layout.addWidget(self.result_figure)
        self.result_tab_layout.addLayout(self.result_tab_layout_h)
        self.result_tab_layout_h.addWidget(self.export_protocol_button)
        self.result_tab_layout_h.addWidget(self.export_to_json_button)
        self.export_protocol_button.clicked.connect(self.on_export_protocol_click)
        self.export_to_json_button.clicked.connect(self.on_export_to_json_click)
        self.result_tab_layout_h.addStretch()
        self.export_protocol_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        self.export_protocol_shortcut.activated.connect(self.on_export_protocol_click)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.source_figure_tab, "Source")
        self.tabs.addTab(self.roi_tab, "ROI")
        self.tabs.addTab(self.filtered_image_tab, "Filtered Image")
        self.tabs.addTab(self.binarized_image_tab, "Binarized Image")
        self.tabs.addTab(self.result_figure_tab, "Result")
        layout3.addWidget(self.tabs)

        self._source_ax = self.source_figure.figure.subplots()
        # self.shape_selector = RectangleSelector(ax=self._source_ax, onselect=self.on_roi_select, useblit=True,
        #                                         button=[1, 3], interactive=True, spancoords='pixels')
        self.shape_selector = PolygonSelector(ax=self._source_ax, onselect=self.on_poly_select, useblit=True,
                                              props=dict(color='white', linestyle='-', linewidth=2, alpha=0.5))

        self.clear_mpl_selection_shortcut = QShortcut(QKeySequence("Escape"), self)
        self.clear_mpl_selection_shortcut.activated.connect(self.clear_mpl_selection)

        self.status_bar = QStatusBar(self)
        layout3.addWidget(self.status_bar)

        self.setLayout(layout)

    def on_file_open_click(self):

        image_path = QFileDialog.getOpenFileName(self, "Choose file")[0]

        if os.path.exists(image_path):
            if image_path[-2:] != 'pf' or image_path[-3:] != 'txt':
                self.status_bar.showMessage('File type is invalid.\nMake sure to load a *.pf  or *.txt File')
            if image_path[-2:] == 'pf':
                self.source_image, src_img_header = dugr_image_io.convert_tt_image_to_numpy_array(image_path)
                self.vmin = np.max(self.source_image) / 10 ** int(self.logarithmic_scaling_flag[-1])

                if self.src_plot is None:
                    self.src_plot = self._source_ax.imshow(self.source_image,
                                                           norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                           cmap=ls_cmap)
                    self.source_figure.figure.colorbar(self.src_plot, ax=self._source_ax, fraction=0.04, pad=0.035,
                                                       label="cd/m^2")
                else:
                    self.source_figure.figure.clf()
                    self._source_ax = self.source_figure.figure.subplots()
                    self.src_plot = self._source_ax.imshow(self.source_image,
                                                           norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                           cmap=ls_cmap)
                    self.source_figure.figure.colorbar(self.src_plot, ax=self._source_ax, fraction=0.04, pad=0.035,
                                                       label="cd/m^2")
                self.source_figure.draw()
                self.shape_selector = PolygonSelector(ax=self._source_ax, onselect=self.on_poly_select, useblit=True,
                                                      props=dict(color='white', linestyle='-', linewidth=2, alpha=0.5))
                self.status_bar.showMessage('File import successful')

            elif image_path[-3:] == "txt":
                self.source_image = dugr_image_io.convert_ascii_image_to_numpy_array(image_path)
                self.vmin = np.max(self.source_image) / 10 ** int(self.logarithmic_scaling_flag[-1])

                if self.src_plot is None:
                    self.src_plot = self._source_ax.imshow(self.source_image,
                                                           norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                           cmap=ls_cmap)
                    self.source_figure.figure.colorbar(self.src_plot, ax=self._source_ax, fraction=0.04, pad=0.035,
                                                       label="cd/m^2")
                else:
                    self.src_plot.set_data(self.source_image)
                    self.src_plot.autoscale()
                self.source_figure.draw()
                self.shape_selector = PolygonSelector(ax=self._source_ax, onselect=self.on_poly_select, useblit=True,
                                                      props=dict(color='white', linestyle='-', linewidth=2, alpha=0.5))
                self.status_bar.showMessage('File import successful')
        else:
            self.status_bar.showMessage('No File selected')

    def on_load_parameter_click(self):
        parameter_path = QFileDialog.getOpenFileName(self, "Choose file")[0]
        if os.path.exists(parameter_path):
            if parameter_path[-4:] != 'json':
                self.status_bar.showMessage('Parameter file type is invalid.\nMake sure to load a *.json File')
            else:
                with open(parameter_path) as f:
                    data = json.load(f)
                self.luminance_threshold_line_box.setText(str(data["lum_th"]))
                self.focal_length_line_box.setText(str(data["focal_length"]))
                self.pixel_size_line_box.setText(str(data["pixel_size"]))
                self.d_line_box.setText(str(data["d"]))
                self.viewing_angle_line_box.setText(str(data["viewing_angle"]))
                self.viewing_distance_line_box.setText(str(data["viewing_distance"]))
                self.luminaire_width_line_box.setText(str(data["luminaire_width"]))
                self.luminaire_height_line_box.setText(str(data["luminaire_height"]))

                self.status_bar.showMessage("Parameter import successfull")

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
                                                   cmap=ls_cmap)
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

        self.rois.append(ROI)

        if len(self.rois) == 1:
            if isinstance(self.rois[0], RectangularRoi):
                self._roi_axs = self.roi_figure.figure.subplots()
                roi_plot = self._roi_axs.imshow(self.rois[0].roi_array,
                                                norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                cmap=ls_cmap)
                self.roi_figure.figure.colorbar(roi_plot, ax=self._roi_axs, fraction=0.04, pad=0.035,
                                                label="cd/m^2")

            elif isinstance(self.rois[0], CircularRoi):
                self._roi_axs = self.roi_figure.figure.subplots()
                roi_plot = self._roi_axs.imshow(self.rois[0].bounding_box,
                                                norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                cmap=ls_cmap)
                self.roi_figure.figure.colorbar(roi_plot, ax=self._roi_axs, fraction=0.04, pad=0.035,
                                                label="cd/m^2")

                t = np.linspace(0, 2 * math.pi, 100)
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
                                                cmap=ls_cmap)
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
                                                       cmap=ls_cmap)
                    self.roi_figure.figure.colorbar(roi_plot, ax=self._roi_axs[i], fraction=0.04, pad=0.035,
                                                    label="cd/m^2")

                elif isinstance(self.rois[i], CircularRoi):
                    roi_plot = self._roi_axs[i].imshow(self.rois[i].bounding_box,
                                                       norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                       cmap=ls_cmap)
                    self.roi_figure.figure.colorbar(roi_plot, ax=self._roi_axs[i], fraction=0.04, pad=0.035,
                                                    label="cd/m^2")

                    t = np.linspace(0, 2 * math.pi, 100)
                    self._roi_axs[i].plot(self.rois[i].width / 2 + (self.rois[i].width - 1) / 2 * np.cos(t),
                                          self.rois[i].height / 2 + (self.rois[i].height - 1) / 2 * np.sin(t),
                                          color='red')

                elif isinstance(self.rois[i], TrapezoidRoi):
                    roi_plot = self._roi_axs[i].imshow(self.rois[i].bounding_box,
                                                       norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                       cmap=ls_cmap)
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
                                                    cmap=ls_cmap)
                    self.roi_figure.figure.colorbar(roi_plot, ax=self._roi_axs, fraction=0.04, pad=0.035,
                                                    label="cd/m^2")

                elif isinstance(self.rois[0], CircularRoi):
                    self._roi_axs = self.roi_figure.figure.subplots()
                    roi_plot = self._roi_axs.imshow(self.rois[0].bounding_box,
                                                    norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                    cmap=ls_cmap)
                    self.roi_figure.figure.colorbar(roi_plot, ax=self._roi_axs, fraction=0.04, pad=0.035,
                                                    label="cd/m^2")

                    t = np.linspace(0, 2 * math.pi, 100)
                    self._roi_axs.plot(self.rois[0].width / 2 + (self.rois[0].width - 1) / 2 * np.cos(t),
                                       self.rois[0].height / 2 + (self.rois[0].height - 1) / 2 * np.sin(t),
                                       color='red')

                elif isinstance(self.rois[0], TrapezoidRoi):
                    self._roi_axs = self.roi_figure.figure.subplots()
                    roi_plot = self._roi_axs.imshow(self.rois[0].bounding_box,
                                                       norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                       cmap=ls_cmap)
                    self.roi_figure.figure.colorbar(roi_plot, ax=self._roi_axs, fraction=0.04, pad=0.035,
                                                    label="cd/m^2")

            if len(self.rois) > 1:
                self._roi_axs = self.roi_figure.figure.subplots(len(self.rois))
                for i in range(len(self.rois)):
                    if isinstance(self.rois[i], RectangularRoi):
                        roi_plot = self._roi_axs[i].imshow(self.rois[i].roi_array,
                                                           norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                           cmap=ls_cmap)
                        self.roi_figure.figure.colorbar(roi_plot, ax=self._roi_axs[i], fraction=0.04, pad=0.035,
                                                        label="cd/m^2")
                    elif isinstance(self.rois[i], CircularRoi):
                        roi_plot = self._roi_axs[i].imshow(self.rois[i].bounding_box,
                                                           norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                           cmap=ls_cmap)
                        self.roi_figure.figure.colorbar(roi_plot, ax=self._roi_axs[i], fraction=0.04, pad=0.035,
                                                        label="cd/m^2")

                        t = np.linspace(0, 2 * math.pi, 100)
                        self._roi_axs[i].plot(self.rois[i].width / 2 + (self.rois[i].width - 1) / 2 * np.cos(t),
                                              self.rois[i].height / 2 + (self.rois[i].height - 1) / 2 * np.sin(t),
                                              color='red')
                    elif isinstance(self.rois[i], TrapezoidRoi):
                        roi_plot = self._roi_axs[i].imshow(self.rois[i].bounding_box,
                                                           norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                           cmap=ls_cmap)
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
            self.DUGR_I = 8 * math.log(self.k_square_I, 10)

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
        self._filtered_image_ax = self.filtered_image_figure.figure.subplots()
        if self.filter_only_roi_flag:
            if self.roi_shape_flag == "Trapezoid":
                self._filtered_image_ax.plot((self.rois[0].top_left[0], self.rois[0].top_right[0]),
                                             (self.rois[0].top_left[1], self.rois[0].top_right[1]), color='red')
                self._filtered_image_ax.plot((self.rois[0].top_right[0], self.rois[0].bottom_right[0]),
                                             (self.rois[0].top_right[1], self.rois[0].bottom_right[1]), color='red')
                self._filtered_image_ax.plot((self.rois[0].bottom_right[0], self.rois[0].bottom_left[0]),
                                             (self.rois[0].bottom_right[1], self.rois[0].bottom_left[1]), color='red')
                self._filtered_image_ax.plot((self.rois[0].bottom_left[0], self.rois[0].top_left[0]),
                                             (self.rois[0].bottom_left[1], self.rois[0].top_left[1]), color='red')
        self.filtered_image_plot = self._filtered_image_ax.imshow(self.filtered_image,
                                                                  norm=LogNorm(
                                                                      vmin=self.vmin, vmax=np.max(self.source_image)),
                                                                  cmap=ls_cmap)
        self.filtered_image_figure.figure.colorbar(self.filtered_image_plot, ax=self._filtered_image_ax, fraction=0.04,
                                                   pad=0.035, label="[cd/m^2]")

        self.binarized_image_figure.figure.clf()
        self._binarized_image_ax = self.binarized_image_figure.figure.subplots()
        if self.filter_only_roi_flag:
            if self.roi_shape_flag == "Trapezoid":
                self._binarized_image_ax.plot((self.rois[0].top_left[0], self.rois[0].top_right[0]),
                                             (self.rois[0].top_left[1], self.rois[0].top_right[1]), color='red')
                self._binarized_image_ax.plot((self.rois[0].top_right[0], self.rois[0].bottom_right[0]),
                                             (self.rois[0].top_right[1], self.rois[0].bottom_right[1]), color='red')
                self._binarized_image_ax.plot((self.rois[0].bottom_right[0], self.rois[0].bottom_left[0]),
                                             (self.rois[0].bottom_right[1], self.rois[0].bottom_left[1]), color='red')
                self._binarized_image_ax.plot((self.rois[0].bottom_left[0], self.rois[0].top_left[0]),
                                             (self.rois[0].bottom_left[1], self.rois[0].top_left[1]), color='red')
        self.binarized_image_plot = self._binarized_image_ax.imshow(self.binarized_image,
                                                                    norm=LogNorm(
                                                                        vmin=self.vmin, vmax=np.max(self.source_image)),
                                                                    cmap=ls_cmap)
        self.binarized_image_figure.figure.colorbar(self.binarized_image_plot, ax=self._binarized_image_ax,
                                                    fraction=0.04, pad=0.035, label="[cd/m^2]")

        self.result_figure.figure.clf()
        self._result_ax = self.result_figure.figure.subplots()
        self._result_ax.axis('off')
        self.result_table = self._result_ax.table(cellText=table_data, loc='center', cellLoc='center')
        self.result_table.set_fontsize(13)
        self.result_table.scale(1, 2)
        self.result_figure.draw()

        self.status_bar.showMessage("DUGR calculation successful")

    def on_export_protocol_click(self):

        image_file = QFileDialog.getSaveFileName(self, "Export File", "", "*.pdf")[0]
        if image_file:
            pdf = PdfPages(image_file)

            self.source_figure.figure.suptitle("Source Image", fontsize=12)
            self.result_figure.figure.suptitle("Result", fontsize=12)

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
        self.luminaire_width = 0
        self.luminaire_height = 0
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
        self.filter_width = 2 * math.ceil(3 * self.sigma) + 1
        self.border_size = math.floor(self.filter_width / 2)

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

        # Luminaire width Label + Line Box
        luminaire_width_label = QLabel("Width [mm]")
        layout2.addWidget(luminaire_width_label)
        self.luminaire_width_line_box = QLineEdit()
        layout2.addWidget(self.luminaire_width_line_box)
        self.luminaire_width_line_box.setText(str(self.luminaire_width))
        self.luminaire_width_line_box.textChanged.connect(self.on_luminaire_width_change)

        # Luminaire height Label + Line Box
        luminaire_height_label = QLabel("Height [mm]")
        layout2.addWidget(luminaire_height_label)
        self.luminaire_height_line_box = QLineEdit()
        layout2.addWidget(self.luminaire_height_line_box)
        self.luminaire_height_line_box.setText(str(self.luminaire_height))
        self.luminaire_height_line_box.textChanged.connect(self.on_luminaire_height_change)

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
        self.rectified_tab_layout.addWidget(mpl_rectified_toolbar)
        self.rectified_tab_layout.addWidget(self.rectified_figure)
        self.rectified_tab_layout.addLayout(self.rectified_tab_layout_h)
        self.rectified_tab_layout_h.addWidget(self.clear_roi_selection_button)
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
        self.export_to_json_button = QPushButton("Export to *.json", self)
        self.result_tab_layout.addWidget(self.result_figure)
        self.result_tab_layout.addLayout(self.result_tab_layout_h)
        self.result_tab_layout_h.addWidget(self.export_protocol_button)
        self.export_protocol_button.clicked.connect(self.on_export_protocol_click)
        self.result_tab_layout_h.addWidget(self.export_to_json_button)
        self.export_to_json_button.clicked.connect(self.on_export_to_json_click)
        self.result_tab_layout_h.addStretch()
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

        if os.path.exists(image_path):
            if image_path[-2:] != 'pf' or image_path[-3:] != 'txt':
                self.status_bar.showMessage('File type is invalid.\nMake sure to load a *.pf  or *.txt File')
            if image_path[-2:] == 'pf':
                self.source_image, src_img_header = dugr_image_io.convert_tt_image_to_numpy_array(image_path)
                self.vmin = np.max(self.source_image) / 10 ** int(self.logarithmic_scaling_flag[-1])

                if self.src_plot is None:
                    self.src_plot = self._source_ax.imshow(self.source_image,
                                                           norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                           cmap=ls_cmap)
                    self.source_figure.figure.colorbar(self.src_plot, ax=self._source_ax, fraction=0.04, pad=0.035,
                                                       label="cd/m^2")
                else:
                    self.source_figure.figure.clf()
                    self._source_ax = self.source_figure.figure.subplots()
                    self.src_plot = self._source_ax.imshow(self.source_image,
                                                           norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                           cmap=ls_cmap)
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
                self.vmin = np.max(self.source_image) / 10 ** int(self.logarithmic_scaling_flag[-1])

                if self.src_plot is None:
                    self.src_plot = self._source_ax.imshow(self.source_image,
                                                           norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                           cmap=ls_cmap)
                    self.source_figure.figure.colorbar(self.src_plot, ax=self._source_ax, fraction=0.04, pad=0.035,
                                                       label="cd/m^2")
                else:
                    self.src_plot.set_data(self.source_image)
                    self.src_plot.autoscale()
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

        edge_width_front = round(self.luminaire_width / (self.projective_rect[2][0] - self.projective_rect[3][0]), 4)
        edge_width_back = round(self.luminaire_width / (self.projective_rect[1][0] - self.projective_rect[0][0]), 4)

        cathete_left1 = self.projective_rect[3][1] - self.projective_rect[0][1]
        cathete_left2 = self.projective_rect[0][0] - self.projective_rect[3][0]
        edge_height_left = round(self.luminaire_height / math.sqrt(cathete_left1 ** 2 + cathete_left2 ** 2), 4)

        cathete_right1 = self.projective_rect[2][1] - self.projective_rect[1][1]
        cathete_right2 = self.projective_rect[2][0] - self.projective_rect[1][0]
        edge_height_right = round(self.luminaire_height / math.sqrt(cathete_right1 ** 2 + cathete_right2 ** 2), 4)

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
        if self.luminaire_width == 0:
            self.status_bar.showMessage('In order to execute the projective correction the luminaire width has to be'
                                        ' defined')
            return
        if self.luminaire_height == 0:
            self.status_bar.showMessage('In order to execute the projective correction the luminaire height has to be'
                                        ' defined')
            return
        self.rectified_image = dugr_image_processing.projective_rectification(self.source_image,
                                                                              self.projective_rect,
                                                                              self.luminaire_width,
                                                                              self.luminaire_height)
        if self.rect_plot is None:
            self.rect_plot = self._rectified_ax.imshow(self.rectified_image,
                                                       norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                       cmap=ls_cmap)
            self.rectified_figure.figure.colorbar(self.rect_plot, ax=self._rectified_ax, fraction=0.04, pad=0.035,
                                                  label="cd/m^2")
        else:
            self.rectified_figure.figure.clf()
            self.rect_plot = self._rectified_ax.imshow(self.rectified_image,
                                                       norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                       cmap=ls_cmap)
            self.rectified_figure.figure.colorbar(self.rect_plot, ax=self._rectified_ax, fraction=0.04, pad=0.035,
                                                  label="cd/m^2")
        self.rectified_figure.draw()
        self.status_bar.showMessage("Projective Transformation successful")

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
            self.filter_width = 2 * math.ceil(3 * self.sigma) + 1
            self.border_size = math.floor(self.filter_width / 2)
            self.status_bar.showMessage("FWHM successfully changed to: " + str(self.FWHM) + ";     Sigma: " +
                                        str(self.sigma) + ";     Filter width: " +
                                        str(self.filter_width) + ";     Border size: " + str(self.border_size))

    def on_luminaire_width_change(self):
        userinput = self.luminaire_width_line_box.text()
        if userinput.isdigit():
            self.luminaire_width = int(userinput)
            self.status_bar.showMessage("Luminaire width successfully changed to: " + str(self.luminaire_width))

    def on_luminaire_height_change(self):
        userinput = self.luminaire_height_line_box.text()
        if userinput.isdigit():
            self.luminaire_height = int(userinput)
            self.status_bar.showMessage("Luminaire height successfully changed to: " + str(self.luminaire_height))

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
                                                   cmap=ls_cmap)
            self.source_figure.figure.colorbar(self.src_plot, ax=self._source_ax, fraction=0.04, pad=0.035,
                                               label="cd/m^2")
            self.source_figure.draw()
            self.poly = PolygonSelector(ax=self._source_ax, onselect=self.on_poly_select, useblit=True,
                                                  props=dict(color='white', linestyle='-', linewidth=2, alpha=0.5))

    def on_safe_roi_click(self):
        if self.rectified_image is None:
            self.status_bar.showMessage("In order to safe ROIs you have to execute the projective correction first")
            return
        if not np.any(self.click):
            self.status_bar.showMessage("In order to safe ROIs you have to draw them onto the projective corrected "
                                        "image first")
            return

        if self.roi_shape_flag == "Rectangular":
            ROI = RectangularRoi(self.rectified_image[self.click[1]:self.release[1], self.click[0]:self.release[0]],
                                 np.array([self.click, self.release]))
        elif self.roi_shape_flag == "Circular":
            ROI = CircularRoi(self.rectified_image[self.click[1]:self.release[1], self.click[0]:self.release[0]],
                              np.array([self.click, self.release]))

        self.rois.append(ROI)

        if len(self.rois) == 1:
            if isinstance(self.rois[0], RectangularRoi):
                self._roi_axs = self.roi_figure.figure.subplots()
                roi_plot = self._roi_axs.imshow(self.rois[0].roi_array,
                                                norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                cmap=ls_cmap)
                self.roi_figure.figure.colorbar(roi_plot, ax=self._roi_axs, fraction=0.04, pad=0.035,
                                                label="cd/m^2")

            elif isinstance(self.rois[0], CircularRoi):
                self._roi_axs = self.roi_figure.figure.subplots()
                roi_plot = self._roi_axs.imshow(self.rois[0].bounding_box,
                                                norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                cmap=ls_cmap)
                self.roi_figure.figure.colorbar(roi_plot, ax=self._roi_axs, fraction=0.04, pad=0.035,
                                                label="cd/m^2")

                t = np.linspace(0, 2 * math.pi, 100)
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
                                                       cmap=ls_cmap)
                    self.roi_figure.figure.colorbar(roi_plot, ax=self._roi_axs[i], fraction=0.04, pad=0.035,
                                                    label="cd/m^2")
                elif isinstance(self.rois[i], CircularRoi):
                    roi_plot = self._roi_axs[i].imshow(self.rois[i].bounding_box,
                                                       norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                       cmap=ls_cmap)
                    self.roi_figure.figure.colorbar(roi_plot, ax=self._roi_axs[i], fraction=0.04, pad=0.035,
                                                    label="cd/m^2")

                    t = np.linspace(0, 2 * math.pi, 100)
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
                                                    cmap=ls_cmap)
                    self.roi_figure.figure.colorbar(roi_plot, ax=self._roi_axs, fraction=0.04, pad=0.035,
                                                    label="cd/m^2")

                elif isinstance(self.rois[0], CircularRoi):
                    self._roi_axs = self.roi_figure.figure.subplots()
                    roi_plot = self._roi_axs.imshow(self.rois[0].bounding_box,
                                                    norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                    cmap=ls_cmap)
                    self.roi_figure.figure.colorbar(roi_plot, ax=self._roi_axs, fraction=0.04, pad=0.035,
                                                    label="cd/m^2")

                    t = np.linspace(0, 2 * math.pi, 100)
                    self._roi_axs.plot(self.rois[0].width / 2 + (self.rois[0].width - 1) / 2 * np.cos(t),
                                       self.rois[0].height / 2 + (self.rois[0].height - 1) / 2 * np.sin(t),
                                       color='red')

            if len(self.rois) > 1:
                self._roi_axs = self.roi_figure.figure.subplots(len(self.rois))
                for i in range(len(self.rois)):
                    if isinstance(self.rois[i], RectangularRoi):
                        roi_plot = self._roi_axs[i].imshow(self.rois[i].roi_array,
                                                           norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                           cmap=ls_cmap)
                        self.roi_figure.figure.colorbar(roi_plot, ax=self._roi_axs[i], fraction=0.04, pad=0.035,
                                                        label="cd/m^2")
                    elif isinstance(self.rois[i], CircularRoi):
                        roi_plot = self._roi_axs[i].imshow(self.rois[i].bounding_box,
                                                           norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                           cmap=ls_cmap)
                        self.roi_figure.figure.colorbar(roi_plot, ax=self._roi_axs[i], fraction=0.04, pad=0.035,
                                                        label="cd/m^2")

                        t = np.linspace(0, 2 * math.pi, 100)
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
                                                                                          self.luminaire_width,
                                                                                          self.luminaire_height)
        self.filtered_image = dugr_image_processing.filter_image(warped_border_image, self.filter_width, self.sigma)
        self.filter_image_figure.figure.clf()
        self._filter_ax = self.filter_image_figure.figure.subplots()
        filter_plot = self._filter_ax.imshow(self.filtered_image,
                                             norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                             cmap=ls_cmap)

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
                        if ((((x - middle_point_x) ** 2) /\
                             ((self.roi_coords_with_border[i].width / 2) ** 2)) +\
                            (((y - middle_point_y) ** 2) / (self.roi_coords_with_border[i].height / 2) ** 2)) >= 1:
                            binarize_roi[y][x] = 0
                binarize_roi[binarize_roi < self.lum_th] = 0
                self.binarized_rois.append(binarize_roi)

            if len(self.binarized_rois) == 1:
                self.binarize_figure.figure.clf()
                self._binarize_ax = self.binarize_figure.figure.subplots()
                binarize_plot = self._binarize_ax.imshow(self.binarized_rois[0],
                                                         norm=LogNorm(vmin=self.vmin, vmax=np.max(self.source_image)),
                                                         cmap=ls_cmap)
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
                                                                cmap=ls_cmap)
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

            self.k_square = round(((self.Leff ** 2 * self.Aeff) / (self.Ls ** 2 * self.A)), 4)

            self.A_new = round((self.A / self.k_square), 2)

            self.DUGR = round((8 * math.log(self.k_square, 10)), 4)

            table_data = [
                ["DUGR", str(self.DUGR)],
                ["A_new", str(self.A_new) + " [mm^2]"],
                ["k^2", str(self.k_square)],
                ["A_eff", str(self.Aeff) + " [mm^2]"],
                ["L_eff", str(self.Leff) + " [cd/m^2]"],
                ["A", str(self.A) + " [mm^2]"],
                ["L_s", str(self.Ls) + " [cd/m^2]"],
                ["lum_th", str(self.lum_th) + " [cd/m^2]"],
                ["FWHM", str(self.FWHM) + " [mm]"],
                ["Filter width", str(self.filter_width) + " [mm]"],
                ["Filter sigma", str(self.sigma) + " [mm]"],
                ["Border size", str(self.border_size) + " [mm]"],
                ["luminaire width", str(self.luminaire_width) + " [mm]"],
                ["luminaire height", str(self.luminaire_height) + " [mm]"],
            ]

            data = {'Parameter': ['DUGR', 'A_new', 'k^2', 'A_eff', 'L_eff', 'A', 'L_s', 'lum_th', 'FWHM',
                                  'Filter_width', 'Filter_sigma', 'Border_size', 'luminaire_width', 'luminaire_height'],

                    'Value': [self.DUGR, self.A_new, self.k_square, self.Aeff, self.Leff, self.A, self.Ls, self.lum_th,
                              self.FWHM, self.filter_width, self.sigma, self.border_size, self.luminaire_width,
                              self.luminaire_height],

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
        image_file = QFileDialog.getSaveFileName(self, "Export File", "", "*.pdf")[0]
        if image_file:
            pdf = PdfPages(image_file)
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


class RectangularRoi:
    def __init__(self, roi_array, roi_coordinates):
        self.roi_array = roi_array
        self.roi_coordinates = roi_coordinates
        self.top_left = (roi_coordinates[0][0], roi_coordinates[0][1])
        self.top_right = (roi_coordinates[1][0], roi_coordinates[0][1])
        self.bottom_left = (roi_coordinates[0][0], roi_coordinates[1][1])
        self.bottom_right = (roi_coordinates[1][0], roi_coordinates[1][1])

        self.width = self.top_right[0] - self.top_left[0]
        self.height = self.bottom_right[1] - self.top_right[1]

        self.area = self.width * self.height
        self.mean_luminance = np.mean(roi_array)


class CircularRoi:
    def __init__(self, bounding_box, bounding_box_coordinates):
        self.bounding_box = bounding_box
        self.bounding_box_coordinates = bounding_box_coordinates
        self.top_left = (bounding_box_coordinates[0][0], bounding_box_coordinates[0][1])
        self.top_right = (bounding_box_coordinates[1][0], bounding_box_coordinates[0][1])
        self.bottom_left = (bounding_box_coordinates[0][0], bounding_box_coordinates[1][1])
        self.bottom_right = (bounding_box_coordinates[1][0], bounding_box_coordinates[1][1])

        self.width = self.top_right[0] - self.top_left[0]
        self.height = self.bottom_right[1] - self.top_right[1]

        self.middle_point_coordinates = (bounding_box_coordinates[0][0] + self.width / 2,
                                         bounding_box_coordinates[0][1] + self.height / 2)

        self.area = math.pi * (self.width / 2) * (self.height / 2)

        shifted_middle_point_x = self.middle_point_coordinates[0] - self.bounding_box_coordinates[0][0]
        shifted_middle_point_y = self.middle_point_coordinates[1] - self.bounding_box_coordinates[0][1]

        luminance_values = []
        for y in range(self.bounding_box.shape[0]):
            for x in range(bounding_box.shape[1]):
                if (((x - shifted_middle_point_x) ** 2) / ((self.width / 2) ** 2) + (
                        (y - shifted_middle_point_y) ** 2) / (self.height / 2) ** 2) <= 1:
                    luminance_values.append(bounding_box[y][x])

        self.luminance_values = np.array(luminance_values)
        self.mean_luminance = np.mean(self.luminance_values)


class TrapezoidRoi:
    def __init__(self, src_image, vertices):
        self.roi_vertices = vertices
        self.vertices_count = len(self.roi_vertices)

        sum_vertices = self.roi_vertices.sum(axis=1)
        self.top_left = self.roi_vertices[np.argmin(sum_vertices)]  # top left = smallest sum
        self.bottom_right = self.roi_vertices[np.argmax(sum_vertices)]  # bottom right = largest sum

        diff = np.diff(vertices, axis=1)
        self.top_right = self.roi_vertices[np.argmin(diff)]  # top right = smallest difference
        self.bottom_left = self.roi_vertices[np.argmax(diff)]  # bottom left = largest difference

        self.width_bottom = self.bottom_right[0] - self.bottom_left[0]
        self.width_top = self.top_right[0] - self.top_left[0]

        if self.width_top > self.width_bottom:
            self.bounding_box = src_image[int(self.top_left[1]):int(self.bottom_left[1]),
                                          int(self.top_left[0]):int(self.top_right[0])]

            self.d1_x = [0, self.top_right[0] - self.top_left[0]]
            self.d1_y = [0, 0]
            self.d2_x = [self.top_right[0] - self.top_left[0], self.bottom_right[0] - self.top_left[0]]
            self.d2_y = [0, self.bottom_left[1] - self.top_left[1]]
            self.d3_x = [self.bottom_right[0] - self.top_left[0], self.bottom_left[0] - self.top_left[0]]
            self.d3_y = [self.bottom_left[1] - self.top_left[1], self.bottom_left[1] - self.top_left[1]]
            self.d4_x = [self.bottom_left[0] - self.top_left[0], 0]
            self.d4_y = [self.bottom_left[1] - self.top_left[1], 0]

        elif self.width_top <= self.width_bottom:
            self.bounding_box = src_image[int(self.top_left[1]):int(self.bottom_left[1]),
                                          int(self.bottom_left[0]):int(self.bottom_right[0])]

            self.d1_x = [self.top_left[0] - self.bottom_left[0], self.top_right[0] - self.bottom_left[0]]
            self.d1_y = [0, 0]
            self.d2_x = [self.top_right[0] - self.bottom_left[0], self.bottom_right[0] - self.bottom_left[0]]
            self.d2_y = [0, self.bottom_left[1] - self.top_left[1]]
            self.d3_x = [self.bottom_right[0] - self.bottom_left[0], 0]
            self.d3_y = [self.bottom_left[1] - self.top_left[1], self.bottom_left[1] - self.top_left[1]]
            self.d4_x = [0, self.top_left[0] - self.bottom_left[0]]
            self.d4_y = [self.bottom_left[1] - self.top_left[1], 0]

        self.height_left = self.bottom_left[1] - self.top_left[1]
        self.height_right = self.bottom_right[1] - self.top_right[1]

        mean_height = (self.height_right + self.height_left)/2

        self.area = (mean_height * (self.width_top + self.width_bottom))/2


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = MainWindow()
    sys.exit(app.exec())
