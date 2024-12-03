"""
Script that contains the functionality for the DUGR calculation approach without projective correction
"""
from PyQt6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QBoxLayout,
    QWidget,
    QLineEdit,
    QLabel,
    QTabWidget,
    QComboBox,
    QGroupBox,
    QFormLayout,
    QRadioButton,
    QCheckBox,
)

from PyQt6.QtGui import QDoubleValidator, QIntValidator

from ProjectiveDistEvalUi import ProjectiveDistEvalUi

from dugr_image_io import *
from params import *
import tarfile
import io
import io
import locale

class ProjectiveDistUi(QWidget):
    PARAM_CAMERA_ID = 'camera_id'
    PARAM_PIXEL_SIZE = 'pixel_size'
    PARAM_FOCAL_LENGTH = 'focal_length'
    PARAM_VIEWING_DISTANCE = 'viewing_distance'
    PARAM_LUM_AREA_TYPE = 'lum_area_type'
    PARAM_LUM_AREA_C0 = 'lum_area_C0'
    PARAM_LUM_AREA_C90 = 'lum_area_C90'
    PARAM_LUM_AREA_DISTRIBUTED = 'lum_area_distributed'
    PARAM_LUM_AREA_SPAN_C0 = 'lum_area_span_C0'
    PARAM_LUM_AREA_SPAN_C90 = 'lum_area_span_C90'
    PARAM_LUM_AREA_D = 'lum_area_d'

    def __init__(self, parent=None):
        super(ProjectiveDistUi, self).__init__(parent)
        layout = QHBoxLayout()
        layout2 = QVBoxLayout()
        layout.addLayout(layout2)

        self.status_bar = parent.status_bar
        self.main_settings = parent.main_settings

        self.eval_tabs = None

        ### Parameters
        self._param_edit_fields = {}
        self._initParams()

        # Pixel Size Label + Dropdown + Line Box
        label = QLabel("Camera, Pixel Size [mm]")
        layout2.addWidget(label)
        self.cameras_dd = QComboBox()
        for key, camera_data in self.cameras.items():
            self.cameras_dd.addItem(camera_data[0], key)
        self.cameras_dd.activated.connect(self.on_cameras_change)
        layout2.addWidget(self.cameras_dd)
        self.addParamEditField(self.PARAM_PIXEL_SIZE, layout2, 60, use_label=False)

        # Focal Length Label + Line Box
        self.addParamEditField(self.PARAM_FOCAL_LENGTH, layout2, 60)
        # Viewing distance Label + Line Box
        self.addParamEditField(self.PARAM_VIEWING_DISTANCE, layout2, 60)

        ## Luminous area dimensions
        layout2.addWidget(self._initLumAreaGB())
        layout2.addStretch()

        # Create Tabs on the UI and Matplotlib Figures on them
        self.eval_tabs = []
        self.eval_tabs.append(ProjectiveDistEvalUi(self, 0, 25))
        self.eval_tabs.append(ProjectiveDistEvalUi(self, 0, 40))
        self.eval_tabs.append(ProjectiveDistEvalUi(self, 90, 25))
        self.eval_tabs.append(ProjectiveDistEvalUi(self, 90, 40))

        self.tab_widget = QTabWidget()
        for tab in self.eval_tabs:
            self.tab_widget.addTab(tab, tab.getTitle())
            tab.setTitleChangedCallback(self.on_tab_title_changed)

        layout.addWidget(self.tab_widget, 1)
        self.setLayout(layout)

    def addParamEditField(self, param_key, layout, width, use_label=True):
        lb = None
        if use_label:
            lb = QLabel(self.cparams.getName(param_key))

        le = QLineEdit()

        le.setStyleSheet("""QLineEdit:disabled { background-color: rgb(240, 240, 240); color: darkGray }""")
        le.setFixedWidth(width)
        le.setValidator(QDoubleValidator())
        le.editingFinished.connect(lambda:self.on_line_edit_change(param_key))

        if isinstance(layout, QBoxLayout):
            if lb:
                layout.addWidget(lb)
            layout.addWidget(le)
        elif isinstance(layout, QFormLayout):
            if lb:
                layout.addRow(lb, le)
            else:
                layout.addRow(le)

        self._param_edit_fields[param_key] = {'label': lb, 'line_edit': le}
        return le

    def _enableEditFields(self, param_keys:[], flag):
        for param_key in param_keys:
            edit_field = self._param_edit_fields[param_key]
            edit_field['line_edit'].setEnabled(flag)
            label = edit_field['label']
            if label:
                label.setEnabled(flag)

    def _updateUIValues(self):
        self.cameras_dd.setCurrentText(self.cparams.getFmtValue(self.PARAM_CAMERA_ID))

        # Update all QLineEdits connected with params
        for key, edit_field in self._param_edit_fields.items():
            edit_field['line_edit'].setText(self.cparams.getFmtValue(key))

        # Enable / Disable edit fields depending on the luminous area type (0=rectangular, 1=circular)
        lum_area_type = self.cparams.getValue(self.PARAM_LUM_AREA_TYPE)
        area_rect_flag = lum_area_type == 0      # rectangular

        self.lum_area_type_rect_rb.setChecked(area_rect_flag)
        self.lum_area_type_circ_rb.setChecked(not area_rect_flag)
        self._enableEditFields([self.PARAM_LUM_AREA_C0, self.PARAM_LUM_AREA_C90], area_rect_flag)
        self._enableEditFields([self.PARAM_LUM_AREA_D], not area_rect_flag)

        self.lum_area_distributed_ckb.setEnabled(area_rect_flag)

        distributed_flag = self.cparams.getValue(self.PARAM_LUM_AREA_DISTRIBUTED)
        self.lum_area_distributed_ckb.setChecked(distributed_flag)
        enabled_flag = distributed_flag and area_rect_flag
        self._enableEditFields([self.PARAM_LUM_AREA_SPAN_C0, self.PARAM_LUM_AREA_SPAN_C90], enabled_flag)

    def _cameraFmtFunc(self, val):
        return self.cameras[val][0]

    def _lumAreaFmtFunc(self, val):
        return "Circular" if val else "Rectangular"

    def _initParams(self):
        # Init list of cameras as dict of {key : [name, pixel size]}
        self.cameras = {
            'LMK6': ['LMK6 - 5/12', 0.00345],
            'LMK5-5': ['LMK5-5', 0.00345],
            'LMK5-1': ['LMK5-1', 0.00645],
            'LMK98-4': ['LMK98-4', 0.00645],
            'EOS-70d': ['Canon EOS 70d', 0.00409*2],
            'EOS-80d': ['Canon EOS 80d', 0.00373*2],
            'EOS-350d': ['Canon EOS 350d', 0.00641*2],
            'EOS-450d': ['Canon EOS 450d', 0.00519*2],
            'EOS-550d': ['Canon EOS 550d', 0.00429*2],
            'EOS-650d': ['Canon EOS 650d', 0.00429*2],
            'EOS-RP': ['Canon EOS RP', 0.00573*2],
            'custom': ['Custom camera', 0.0]
        }

        self.cparams = ParamGroup('common', "Common parameters")

        camera_id = "LMK6"
        self.cparams.addParam(key=self.PARAM_CAMERA_ID, name="Camera", default=camera_id, fmtFunc=self._cameraFmtFunc)
        self.cparams.addParam(key=self.PARAM_PIXEL_SIZE, name="Pixel size", unit="mm", fmt="%.5f", default=self.cameras[camera_id][1])
        self.cparams.addParam(key=self.PARAM_FOCAL_LENGTH, name="Focal length", unit="mm", default=0.0)
        self.cparams.addParam(key=self.PARAM_VIEWING_DISTANCE, name="Viewing distance", unit="mm", default=0.0)

        self.cparams.addParam(key=self.PARAM_LUM_AREA_TYPE, name="Luminous area shape", default=0, fmtFunc=self._lumAreaFmtFunc)
        self.cparams.addParam(key=self.PARAM_LUM_AREA_C0, name="Size in C0", unit="mm", default=0.0)
        self.cparams.addParam(key=self.PARAM_LUM_AREA_C90, name="Size in C90", unit="mm", default=0.0)
        self.cparams.addParam(key=self.PARAM_LUM_AREA_DISTRIBUTED, name="Distributed areas", default=False)
        self.cparams.addParam(key=self.PARAM_LUM_AREA_SPAN_C0, name="Max span in C0", unit="mm", default=0.0)
        self.cparams.addParam(key=self.PARAM_LUM_AREA_SPAN_C90, name="Max span in C90", unit="mm", default=0.0)
        self.cparams.addParam(key=self.PARAM_LUM_AREA_D, name="Diameter", unit="mm", default=0.0)

    def _initLumAreaGB(self):
        gb = QGroupBox("Luminous area size")
        layout_gb = QFormLayout()
        # Type
        self.lum_area_type_rect_rb = QRadioButton("Rectangular")
        self.lum_area_type_rect_rb.toggled.connect(self.lum_area_type_rect_change)
        self.lum_area_type_circ_rb = QRadioButton("Circular")
        self.lum_area_type_circ_rb.toggled.connect(self.lum_area_type_circ_change)
        layout_gb.addRow(self.lum_area_type_rect_rb, self.lum_area_type_circ_rb)

        # In C0
        self.addParamEditField(self.PARAM_LUM_AREA_C0, layout_gb, 60)
        # In C90
        self.addParamEditField(self.PARAM_LUM_AREA_C90, layout_gb, 60)
        # Distributed areas
        self.lum_area_distributed_ckb = QCheckBox("Distributed areas")
        self.lum_area_distributed_ckb.toggled.connect(self.on_lum_area_distributed_change)
        layout_gb.addRow(self.lum_area_distributed_ckb)
        # Maximum span in C0/90
        self.addParamEditField(self.PARAM_LUM_AREA_SPAN_C0, layout_gb, 60)
        self.addParamEditField(self.PARAM_LUM_AREA_SPAN_C90, layout_gb, 60)
        # Diameter
        self.addParamEditField(self.PARAM_LUM_AREA_D, layout_gb, 60)
        #
        gb.setLayout(layout_gb)

        return gb

    def on_tab_title_changed(self, eval_tab:ProjectiveDistEvalUi):
        idx = self.tab_widget.indexOf(eval_tab)
        if idx != -1:
            self.tab_widget.setTabText(idx, eval_tab.getTitle())

    def saveParams(self, fname):
        self.cparams.saveJSON(fname)

    def loadParams(self, fname):
        self.cparams.loadJSON(fname)
        self._updateUIValues()

    def settingsChanged(self):
        for eval_tab in self.eval_tabs:
            eval_tab.settingsChanged()

    def setImage(self, image):
        idx = self.tab_widget.currentIndex()
        self.eval_tabs[idx].setImage(image)

    def plausibilityTest(self):
        idx = self.tab_widget.currentIndex()
        self.eval_tabs[idx].calculate(only_plausibility_test=True)

    def setImageSeries(self, image_series):
        for eval_tab in self.eval_tabs:
            eval_tab.setUse(False)

        for idx, image_info in enumerate(image_series):
            if idx < 4:
                self.eval_tabs[idx].setImage(image_info['image'])
                self.eval_tabs[idx].setAngles(int(image_info['positions'][0]), int(image_info['positions'][1]))
                self.eval_tabs[idx].setUse(True)
                self.tab_widget.setTabText(idx, self.eval_tabs[idx].getTitle())

    def loadProject(self, fname):
        # reset all tabs
        for eval_tab in self.eval_tabs:
            eval_tab.setUse(False)
            eval_tab.clear()

        with tarfile.open(fname) as tar:
            tar_content = tar.getnames()

            if 'projective_dist' in tar_content:
                # common parameters
                self.cparams.readFromTar(tar, tar_content, "projective_dist/common_param")

                # parameters and data of each view
                for idx, eval_tab in enumerate(self.eval_tabs):
                    eval_tab.loadViewDataFromTar(tar, tar_content, "projective_dist/%d" % (idx))
                    self.tab_widget.setTabText(idx, eval_tab.getTitle())


        self._updateUIValues()

    def saveProject(self, fname):
        # Write to memory Bytes stream first and then save as file
        fh = io.BytesIO()
        with tarfile.open(fileobj=fh, mode='w') as tar:
            # create directory for projective distorted algorithm data
            info = tarfile.TarInfo('projective_dist')
            info.type = tarfile.DIRTYPE
            tar.addfile(info)

            # common parameters
            self.cparams.writeToTar(tar, "projective_dist/common_param")
            # parameters and data of each view
            for idx, eval_tab in enumerate(self.eval_tabs):
                eval_tab.saveViewDataToTar(tar, "projective_dist/%d" % (idx))

        with open(fname, 'wb') as f:
            f.write(fh.getvalue())

    def on_line_edit_change(self, key):
        self.cparams.setValue(key, self.sender().text())
        self._paramChanged()
        self._updateUIValues()

    def on_cameras_change(self, index):
        cam_id = self.cameras_dd.itemData(index)
        self.cparams.setValue(self.PARAM_CAMERA_ID, cam_id)
        self.cparams.setValue(self.PARAM_PIXEL_SIZE, self.cameras[cam_id][1])
        self._paramChanged()
        self._updateUIValues()

    def lum_area_type_rect_change(self):
        if self.sender().isChecked():
            self.cparams.setValue(self.PARAM_LUM_AREA_TYPE, 0)
            self._paramChanged()
            self._updateUIValues()

    def on_lum_area_distributed_change(self):
        self.cparams.setValue(self.PARAM_LUM_AREA_DISTRIBUTED, self.sender().isChecked())
        self._paramChanged()
        self._updateUIValues()

    def lum_area_type_circ_change(self):
        if self.sender().isChecked():
            self.cparams.setValue(self.PARAM_LUM_AREA_TYPE, 1)
            self._paramChanged()
            self._updateUIValues()

    def execute(self):
        self.status_bar.showMessage("Calculate ...")
        self.status_bar.repaint(0, 0, -1, -1)

        for eval_tab in self.eval_tabs:
            eval_tab.calculate()

    def _paramChanged(self):
        if self.eval_tabs is not None:
            for eval_tab in self.eval_tabs:
                eval_tab.setCommonParameter(self.cparams)

