"""
"""

from dugr_image_processing import *
from image_view import ImageView
import pickle

import locale
from math import log
from csv import reader

# Changed "Import Figure Canvas" to import "FigureCanvasQTAgg as Figure Canvas" -> Undo if this raises errors
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import pandas as pd

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
    QSizePolicy,
    QMessageBox,
)

from roi_definitions import saveROIListToTar, loadROIListFromTar
from dugr_image_io import *
from params import *

class ProjectiveDistEvalUi(QWidget):
    def __init__(self, parent, c_angle:int, view_angle:float):
        super(ProjectiveDistEvalUi, self).__init__(parent)

        self.status_bar = parent.status_bar
        self.main_settings = parent.main_settings

        self.title_changed_callback = None

        ### Parameters
        self.params = ParamGroup('eval', "")

        self.params.addParam(key='use_flag', name="Use", default=True)
        self.params.addParam(key='viewing_angle', name="Viewing angle", unit="°", default=view_angle)
        self.params.addParam(key='c_angle', name="C angle", unit="°", default=c_angle)
        self.params.addParam(key='luminous_intensity', name="Luminous intensity", unit="cd", default=0.0)

        # get all parameters from the settings
        self.settingsChanged()

        self.luminous_intensity = 0.0

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
        self.optical_resolution = 0.0

        self.df = None

        self.rois = []
        self.results = None

        self._result_ax = None
        self.result_table = None

        self.A_eff = None
        self.A_p_new_I = None
        self.A_p_new_L = None
        self.A = None
        self.A_new_L = None

        self.roi_img = None
        self.filtered_img = None
        self.threshold_img = None

        ### GUI
        layout = QVBoxLayout()

        toolbar = QWidget()
        layout_toolbar = QHBoxLayout()
        toolbar.setLayout(layout_toolbar)
        layout.addWidget(toolbar)

        ## Toolbar
        # Use Checkbox
        self.use_cb = QCheckBox("Use")
        self.use_cb.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.use_cb.toggled.connect(self.on_use_change)
        layout_toolbar.addWidget(self.use_cb)
        layout_toolbar.addSpacing(10)

        ## C-Plane
        label = QLabel("View from C-")
        label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        layout_toolbar.addWidget(label)
        self.c_angles_cb = QComboBox()
        self.c_angles_cb.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.c_angles_cb.addItems(["0", "90", "180", "270"])
        self.c_angles_cb.currentTextChanged.connect(self.on_c_angle_change)
        layout_toolbar.addWidget(self.c_angles_cb)
        layout_toolbar.addSpacing(10)

        # Viewing angle Label + Line Box
        label = QLabel("View angle \u03B1E [°]")
        layout_toolbar.addWidget(label)
        label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.viewing_angle_le = QLineEdit()
        self.viewing_angle_le.setFixedWidth(40)
        self.viewing_angle_le.editingFinished.connect(self.on_viewing_angle_change)
        layout_toolbar.addWidget(self.viewing_angle_le)
        layout_toolbar.addSpacing(10)

        # Luminous Intensity Label + Line Box
        label = QLabel("I [cd] (optional)")
        label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        layout_toolbar.addWidget(label)
        self.luminous_intensity_le = QLineEdit()
        self.luminous_intensity_le.setFixedWidth(40)
        layout_toolbar.addWidget(self.luminous_intensity_le)
        self.luminous_intensity_le.editingFinished.connect(self.on_luminous_intensity_change)

        # Spacer
        spacer = QWidget()
        layout_toolbar.addWidget(spacer)


        # Create Tabs on the UI and Matplotlib Figures on them
        self.source_img_tab = ImageView(use_roi=True)
        self.source_img_tab.setStatusBar(self.status_bar)
        self.source_img_tab.setROISaveCallback(self.on_safe_roi_click)
        self.source_img_tab.setROIDeleteLastCallback(self.on_delete_last_roi)

        # Regions of interest Tab
        self.roi_img_tab = ImageView(use_roi=False)
        self.roi_img_tab.setStatusBar(self.status_bar)

        # Filtered image Tab
        self.filtered_img_tab = ImageView(use_roi=False)
        self.filtered_img_tab.setStatusBar(self.status_bar)

        # Threshold image Tab
        self.threshold_img_tab = ImageView(use_roi=False)
        self.threshold_img_tab.setStatusBar(self.status_bar)

        self.result_figure_tab = QWidget()
        self.result_tab_layout = QVBoxLayout()
        self.result_tab_layout_h = QHBoxLayout()
        self.result_figure_tab.setLayout(self.result_tab_layout)
        self.result_figure = FigureCanvas(Figure(figsize=(12, 8), layout='constrained'))

        self.export_protocol_button = QPushButton("Generate PDF report", self)
        self.export_protocol_button.setStyleSheet("padding-left: 10px; padding-right: 10px; padding-top: 3px; padding-bottom: 3px;")
        self.export_to_json_button = QPushButton("Export to *.json", self)
        self.export_to_json_button.setStyleSheet("padding-left: 10px; padding-right: 10px; padding-top: 3px; padding-bottom: 3px;")
        self.export_to_csv_button = QPushButton("Export to *.csv", self)
        self.export_to_csv_button.setStyleSheet("padding-left: 10px; padding-right: 10px; padding-top: 3px; padding-bottom: 3px;")

        self.result_tab_layout.addWidget(self.result_figure)
        self.result_tab_layout.addLayout(self.result_tab_layout_h)
        self.result_tab_layout_h.addWidget(self.export_protocol_button)
        self.result_tab_layout_h.addWidget(self.export_to_json_button)
        self.result_tab_layout_h.addWidget(self.export_to_csv_button)
        self.export_protocol_button.clicked.connect(self.on_export_protocol_click)
        self.export_to_json_button.clicked.connect(self.on_export_to_json_click)
        self.export_to_csv_button.clicked.connect(self.on_export_to_csv_click)
        self.result_tab_layout_h.addStretch()
        self.export_protocol_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        self.export_protocol_shortcut.activated.connect(self.on_export_protocol_click)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.source_img_tab, "Source")
        self.tabs.addTab(self.roi_img_tab, "ROI")
        self.tabs.addTab(self.filtered_img_tab, "Filtered Image")
        self.tabs.addTab(self.threshold_img_tab, "Threshold Image")
        self.tabs.addTab(self.result_figure_tab, "Result")
        layout.addWidget(self.tabs)

        self._updateUI()

        self.setLayout(layout)

    def setTitleChangedCallback(self, title_changed_callback):
        self.title_changed_callback = title_changed_callback

    def _updateUI(self):
        self.use_cb.setChecked(self.params.getValue('use_flag'))
        self.c_angles_cb.setCurrentText(self.params.getFmtValue('c_angle'))
        self.viewing_angle_le.setText(self.params.getFmtValue('viewing_angle'))
        self.luminous_intensity_le.setText(self.params.getFmtValue('luminous_intensity'))

    def getTitle(self):
        return "C%s (%s)" % (self.params.getFmtValue('c_angle'), self.params.getFmtValue('viewing_angle'))

    def settingsChanged(self):
        self.lum_th = self.main_settings.getParameterValue('cie232Param', 'lum_th')
        self.d = self.main_settings.getParameterValue('cie232Param', 'eye_res_d')
        self.use_only_roi = self.main_settings.getParameterValue('cie232Param', 'use_only_roi')

    def setCommonParameter(self, cparams:ParamGroup):
        self.cparams = cparams

    def saveViewDataToTar(self, tar, fname_prefix):
        self.params.writeToTar(tar, "%s_param" % (fname_prefix))
        self.source_img_tab.saveToTar(tar, "%s_srcimg" % (fname_prefix))
        saveROIListToTar(self.rois, tar, fname_prefix)

    def loadViewDataFromTar(self, tar, tar_content, fname_prefix):
        if not self.params.readFromTar(tar, tar_content, "%s_param" % (fname_prefix)):
            return
        if not self.source_img_tab.loadFromTar(tar, tar_content, "%s_srcimg" % (fname_prefix)):
            return

        self.rois = loadROIListFromTar(tar, tar_content, fname_prefix, self.source_img_tab.getImage())

        self._updateUI()
        self.update_roi_plot()

    def setImage(self, image):
        self.clear()
        self.source_img_tab.setImage(image)

        self.update_roi_plot()

    def clear(self):
        self.rois = []
        self.source_img_tab.clear()
        self.roi_img_tab.clear()
        self.deleteResults()

    def setAngles(self, c_angle:int, view_angle:int):
        self.params.setValue('c_angle', c_angle)
        self.params.setValue('viewing_angle', view_angle)
        self._updateUI()
        if self.title_changed_callback is not None:
            self.title_changed_callback(self)

    def on_use_change(self):
        self.setUse(self.sender().isChecked())

    def setUse(self, use_flag):
        self.params.setValue('use_flag', use_flag)
        self._updateUI()

    def deleteResults(self):
        self.results = None

        if self.filtered_img != None:
            self.filtered_img = None
            self.filtered_img_tab.clear()
        if self.threshold_img != None:
            self.threshold_img = None
            self.threshold_img_tab.clear()

        self.result_figure.figure.clf()
        self.result_figure.draw()

    def on_safe_roi_click(self, roi):
        self.rois.append(roi)

        self.update_roi_plot()
        self.deleteResults()

        if len(self.rois) == 1:
            self.status_bar.showMessage("Successfully saved: 1 ROI")
        elif len(self.rois) > 1:
            self.status_bar.showMessage("Successfully saved: " + str(len(self.rois)) + " ROIs")

    def on_delete_last_roi(self):
        if len(self.rois) > 0:
            self.rois.pop()
            self.update_roi_plot()
            self.deleteResults()

            self.status_bar.showMessage("Successfully deleted the last ROI.     " + str(len(self.rois)) + " remaining")
        else:
            self.status_bar.showMessage("All of the ROIs have already been removed, none left.")

    def on_c_angle_change(self, c_angle):
        self.params.setValue('c_angle', c_angle)
        self._updateUI()
        if self.title_changed_callback is not None:
            self.title_changed_callback(self)

    def on_luminous_intensity_change(self):
        self.params.setValue('luminous_intensity', self.luminous_intensity_le.text())
        self._updateUI()

    def on_viewing_angle_change(self):
        self.params.setValue('viewing_angle', self.viewing_angle_le.text())
        self._updateUI()
        if self.title_changed_callback is not None:
            self.title_changed_callback(self)

    def update_roi_plot(self):
        if len(self.rois) == 0:
            if self.roi_img != None:
                self.roi_img = None
                self.roi_img_tab.clear()
            return

        ## Bounding box of bounding boxes
        bboxes = []
        for roi in self.rois:
            bboxes.append(roi.bbox)

        bbox = Bbox()
        bbox.fromBboxes(bboxes)
        self.roi_img = DUGRImage(bbox)
        for roi in self.rois:
            self.roi_img.data[roi.bbox.getSlice(bbox)] = roi.img.data

        self.roi_img_tab.setImage(self.roi_img, self.rois)


    def calculate(self, only_plausibility_test=False):
        if not self.params.getValue('use_flag'):
            return

        # When no ROI exists, save ROI in image view when there is one
        if len(self.rois) == 0:
            self.source_img_tab.saveROIWhenExist()

        if len(self.rois) == 0:
            self.status_bar.showMessage("No ROI found! Make sure to safe a ROI before executing the calculation")
            return

        self.results = ParamGroup('results', "Result data")

        # Get common parameters
        lum_area_type = self.cparams.getValue('lum_area_type')
        lum_area_C0 = self.cparams.getValue('lum_area_C0')
        lum_area_C90 = self.cparams.getValue('lum_area_C90')
        lum_area_distributed = self.cparams.getValue('lum_area_distributed')
        lum_area_span_C0 = self.cparams.getValue('lum_area_span_C0')
        lum_area_span_C90 = self.cparams.getValue('lum_area_span_C90')
        lum_area_d = self.cparams.getValue('lum_area_d')
        viewing_distance = self.cparams.getValue('viewing_distance')

        # Get view dependent parameters
        viewing_angle = self.params.getValue('viewing_angle')
        luminous_intensity = self.params.getValue('luminous_intensity')
        c_angle = self.params.getValue('c_angle')

        ## Get luminous area height from this observer position depending on C-Plane and luminous area sizes in
        ## C0 and C90 direction.
        if lum_area_type == 1:         # circular
            luminous_area_height_span = lum_area_d
        else:
            # When there are distributed areas, use the maximum span. This is necessary for calculating rb_min.
            if lum_area_distributed:
                sizes = [lum_area_span_C0, lum_area_span_C90]
            else:
                sizes = [lum_area_C0, lum_area_C90]
            if c_angle in (90, 270):
                sizes.reverse()

            luminous_area_height_span = sizes[0]

        # Get image center from original source image
        img_center = self.source_img_tab.getImage().getBBox().getCenter()

        # Create algorithm instance
        algo = DUGR_ProjectiveDistAlgorithm(src_img=self.roi_img, cparams=self.cparams,
                                                                    luminous_area_height=luminous_area_height_span,
                                                                    viewing_angle=viewing_angle,
                                                                    rois=self.rois,
                                                                    lum_th=self.lum_th,
                                                                    use_only_roi=self.use_only_roi,
                                                                    d=self.d,
                                                                    img_center=img_center,
                                                                    only_plausibility_test=only_plausibility_test)
        # Execute algorithm
        self.l_eff, self.l_s, self.omega_eff, self.omega_l, \
            self.optical_resolution, self.rb_min, self.ro_min, self.fwhm, self.sigma, self.filter_width, \
            self.filtered_img, self.threshold_img = algo.execute()

        # size of luminous area
        if lum_area_type == 1:  # circular
            self.A = np.pi * (lum_area_d / 2) ** 2
        else:
            self.A = lum_area_C0 * lum_area_C90

        # When there are distributed areas and only one ROI given, assume that the ROI envelops the distributed
        # luminous areas (with dark areas in between). In this case, the mean luminance must be corrected by the
        # ratio of the area sizes!
        if len(self.rois) == 1 and lum_area_type == 0 and lum_area_distributed:
            a_span = lum_area_span_C0 * lum_area_span_C90
            ratio = a_span / self.A
            self.l_s *= ratio       # mean luminance gets higher
            self.omega_l /= ratio   # solid angle gets lower
            #print("Mean luminance and solid angle of luminous ara corrected due to distributed areas and only one ROI given: l_s = %s" % str(self.l_s))
            #print("New: omega_l = %s, l_s = %s" % (str(self.omega_l), str(self.l_s)))

        # ----- DUGR values -----
        # k_square: k^2 Value
        self.k_square_L = (self.l_eff ** 2 * self.omega_eff) / (self.l_s ** 2 * self.omega_l)
        # DUGR value
        self.DUGR_L= 8 * log(self.k_square_L, 10)


        self.filtered_img_tab.setImage(self.filtered_img, self.rois)
        self.threshold_img_tab.setImage(self.threshold_img, self.rois)

        self.A_eff = self.omega_eff * viewing_distance**2

        self.A_p = self.A * np.cos(np.radians(90 - viewing_angle))

        self.A_p_new_I = (luminous_intensity ** 2) / ((self.l_eff * 10 ** -6) ** 2 * self.A_eff)
        self.A_p_new_L = self.A_p / self.k_square_L

        self.A_new_L = self.A_p_new_L / np.cos(np.radians(90 - viewing_angle))

        # When plausibility test is requested, calculate A_p from solid angle and radius and show MessageBox.
        if only_plausibility_test:
            A_p_L = self.omega_l * viewing_distance**2
            err = A_p_L / self.A_p * 100.0 - 100.0
            msg = "Difference of projected area and calculated area\n from ROI and camera model is %.1f %%.\n" % err
            msg += "A = %d mm², A_p = A * cos(90° - %d°) = %d.\n" % (int(self.A), int(viewing_angle), int(self.A_p))
            msg += "Recalculated from ROI and solid angle integral: A_p_L = %d mm²" % A_p_L
            QMessageBox.information(self, "Plausibility test", msg)
        else:
            if self.A_p_new_I != 0:
                self.k_square_I = self.A_p / self.A_p_new_I
                self.DUGR_I = 8 * log(self.k_square_I, 10)

                self.results.addParam(key='DUGR_I', name="DUGR_I", unit="", fmt="%.1f", default=self.DUGR_I)
                self.results.addParam(key='k2_I', name="k^2_I", unit="", fmt="%.1f", default=self.k_square_I)
                self.results.addParam(key='A_p_new_I', name="A_p_new_I", unit="", fmt="%.0f", default=self.A_p_new_I)
                self.results.addParam(key='I', name="I", unit="", fmt="%.1f", default=luminous_intensity)

            self.results.addParam(key='DUGR_L', name="DUGR_L", unit="", fmt="%.1f", default=self.DUGR_L)
            self.results.addParam(key='"k2_L', name="k^2_L", unit="", fmt="%.1f", default=self.k_square_L)
            self.results.addParam(key='A_p_new_L', name="A_p_new_L", unit="mm²", fmt="%.0f", default=self.A_p_new_L)
            self.results.addParam(key='A_p', name="A_p", unit="mm²", fmt="%.0f", default=self.A_p)
            self.results.addParam(key='A_new_L', name="A_new_L", unit="mm²", fmt="%.0f", default=self.A_new_L)
            self.results.addParam(key='A', name="A", unit="mm²", fmt="%.0f", default=self.A)
            self.results.addParam(key='A_eff', name="A_eff", unit="mm²", fmt="%.0f", default=self.A_eff)
            self.results.addParam(key='L_eff', name="Effective luminance (L_eff)", unit="cd/m²", fmt="%.2f", default=self.l_eff)
            self.results.addParam(key='L_mean', name="Mean Luminaire luminance (L_mean)", unit="cd/m²", fmt="%.2f", default=self.l_s)
            self.results.addParam(key='omega_eff', name="Effective solid angle (\u03C9_eff)", unit="sr", fmt="%.6f", default=self.omega_eff)
            self.results.addParam(key='omega_luminaire', name="Luminaire solid angle (\u03C9_luminaire)", unit="sr", fmt="%.6f", default=self.omega_l)
            self.results.addParam(key='alpha_E', name="Measurement angle \u03B1E", unit="°", fmt="%.0f", default=viewing_angle)
            self.results.addParam(key='distance', name="Viewing distance", unit="mm", fmt="%.0f", default=viewing_distance)
            self.results.addParam(key='lum_area_C0', name="Luminous area size in C0", unit="mm", fmt="%.0f", default=lum_area_C0)
            self.results.addParam(key='lum_area_C90', name="Luminous area size in C90", unit="mm", fmt="%.0f", default=lum_area_C90)
            self.results.addParam(key='lum_th', name="Luminance threshold", unit="cd/m²", fmt="%.0f", default=self.lum_th)
            self.results.addParam(key='d_eye', name="Eye resolution", unit="mm", fmt="%.0f", default=self.d)
            self.results.addParam(key='optical_resolution', name="Calculated optical resolution", unit="°/px", fmt="%.5f", default=self.optical_resolution)
            self.results.addParam(key='fwhm', name="FWHM", unit="px", fmt="%.2f", default=self.fwhm)
            self.results.addParam(key='filter_width', name="Filter width", unit="px", fmt="%.0f", default=self.filter_width)
            self.results.addParam(key='sigma', name="Filter sigma", unit="px", fmt="%.3f", default=self.sigma)
            self.results.addParam(key='rb_min', name="rb min", unit="mm", fmt="%.2f", default=self.rb_min)
            self.results.addParam(key='ro_min', name="ro min", unit="°/px", fmt="%.5f", default=self.ro_min)

            self.result_figure.figure.clf()
            self._result_ax = self.result_figure.figure.subplots()
            self._result_ax.axis('off')
            self._result_ax.axis('tight')

            table_data = self.results.getTable()
            self.result_table = self._result_ax.table(cellText=table_data, loc='center', cellLoc='center')

            self.result_table.auto_set_font_size(True)
            self.result_table.set_fontsize(12)
            self.result_table.scale(1, 1.5)

            self.result_figure.draw()

            self.status_bar.showMessage("DUGR calculation successful")

    def on_export_protocol_click(self):
        if self.results is None:
            return

        protocol_file = QFileDialog.getSaveFileName(self, "PDF report file", "", "*.pdf")[0]
        if protocol_file:
            self.generate_pdf(protocol_file, [self.source_img_tab.figure.figure,
                                    self.roi_img_tab.figure.figure, self.filtered_img_tab.figure.figure,
                                    self.threshold_img_tab.figure.figure, self.result_figure.figure])

    def generate_pdf(self, filename, figures):
        pdf = PdfPages(filename)

        for fig in figures:
            # FigureCanvasPdf (used by PdfPages) modifies the figure object so that it is not usable by the GUI
            # anymore. Temporarily copying of the figure object is the solution but deepcopy is not supports.
            # As workaround for deepcopy, pickle is used.
            fig2 = pickle.loads(pickle.dumps(fig))
            pdf.savefig(fig2)
        pdf.close()

    def on_export_to_json_click(self):
        if self.results is None:
            return

        json_file = QFileDialog.getSaveFileName(self, "Export File", "", "*.json")[0]
        if json_file:
            self.results.getPD().to_json(json_file)
            self.status_bar.showMessage('Export to *.json File successful')

    def on_export_to_csv_click(self):
        if self.results is None:
            return

        csv_file = QFileDialog.getSaveFileName(self, "Export File", "", "*.csv")[0]
        if csv_file:
            self.results.getPD().to_csv(csv_file, encoding='utf-8-sig')
            self.status_bar.showMessage('Export to *.csv File successful')


