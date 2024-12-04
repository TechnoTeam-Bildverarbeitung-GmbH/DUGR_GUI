"""
MAIN SCRIPT of the GUI

Defines the main window and calls the different algorithm files
"""
import os
from ProjectiveDistUi import ProjectiveDistUi
from projective_corrected_algorithm import ProjectiveCorrUi
from sys import argv, exit
from os.path import dirname, join
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMenu,
    QDialog,
    QFileDialog,
    QToolBar,
    QVBoxLayout,
    QDialogButtonBox,
    QLabel,
    QFormLayout,
    QLineEdit,
    QGroupBox,
    QCheckBox,
    QMessageBox,
)
import sys
import traceback
import copy
from dugr_image_io import *
from tt_image_series import LoadTTImageSeries
import locale

basedir = dirname(__file__)


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        locale.setlocale(locale.LC_ALL, '')

        self.eval_tab = None

        self.main_settings = MainSettings()

        self.resize(1400, 900)

        self.setWindowTitle("DUGR Calc 1.5.0 - TechnoTeam Bildverarbeitung GmbH")
        self.setWindowIcon(QIcon("icons/light-bulb_16x16.png"))

        self.createMenuBar()

        self.status_bar = self.statusBar()

        # Activate projective distorted GUI
        self.onSwitchProjRectMode(False)
        self.show()

    def createMenuBar(self):
        menuBar = self.menuBar()

        # ------ File -------
        menu = QMenu("&File", self)
        # Load project
        action = QAction("&Load project ...", self)
        menu.addAction(action)
        action.triggered.connect(self.onLoadProjectClick)
        # Save project
        action = QAction("&Save project ...", self)
        menu.addAction(action)
        action.triggered.connect(self.onSaveProjectClick)
        # Open image
        menu.addSeparator()
        action = QAction("&Open image ...", self)
        menu.addAction(action)
        action.triggered.connect(self.onFileOpenImageClick)
        # Open TT image series
        action = QAction("Open TT image series ...", self)
        menu.addAction(action)
        action.triggered.connect(self.onFileOpenImageSeriesClick)
        # Settings
        menu.addSeparator()
        action = QAction("Settings ...", self)
        menu.addAction(action)
        action.triggered.connect(self.onFileSettingsClick)
        #
        menuBar.addMenu(menu)

        # ------ Calculation -------
        menu = QMenu("&Calculation", self)
        # Use projective rectification
        action = QAction("Projective rectification mode", self)
        action.setCheckable(True)
        menu.addAction(action)
        action.triggered.connect(self.onSwitchProjRectMode)

        # Execute
        action = QAction("&Execute", self)
        menu.addAction(action)
        action.triggered.connect(self.onCalcExecuteClick)
        # Execute
        menu.addSeparator()
        action = QAction("&Plausibility test", self)
        menu.addAction(action)
        action.triggered.connect(self.onPlausibilityTestClick)
        #
        menuBar.addMenu(menu)

        # ------ Help -------
        menu = menuBar.addMenu("&Help")
        # Manual
        action = QAction("&Software manual", self)
        menu.addAction(action)
        action.triggered.connect(self.onHelpManual)
        # About
        action = QAction("&About", self)
        menu.addAction(action)
        action.triggered.connect(self.onHelpAbout)

        # -------- Toolbar ----------
        toolbar = QToolBar("")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        # Load project
        action = QAction(QIcon("icons/Open.png"), "Load project", self)
        action.setStatusTip("Load project")
        action.triggered.connect(self.onLoadProjectClick)
        toolbar.addAction(action)
        # Save project
        action = QAction(QIcon("icons/Save.png"), "Save project", self)
        action.setStatusTip("Load project")
        action.triggered.connect(self.onSaveProjectClick)
        toolbar.addAction(action)
        # Load image
        toolbar.addSeparator()
        action = QAction(QIcon("icons/Open image.png"), "Open image", self)
        action.setStatusTip("Open image")
        action.triggered.connect(self.onFileOpenImageClick)
        toolbar.addAction(action)
        # Load image series
        action = QAction(QIcon("icons/open-image-folder.png"), "Open image folder", self)
        action.setStatusTip("Open image folder")
        action.triggered.connect(self.onFileOpenImageSeriesClick)
        toolbar.addAction(action)
        # Execute
        toolbar.addSeparator()
        action = QAction(QIcon("icons/run.png"), "Execute calculation", self)
        action.setStatusTip("Execute calculation")
        action.triggered.connect(self.onCalcExecuteClick)
        toolbar.addAction(action)

    def onLoadProjectClick(self):
        fname = QFileDialog.getOpenFileName(self, "Load project", "*.tar")[0]
        if not fname:
            return
        self.eval_tab.loadProject(fname)

    def onSaveProjectClick(self):
        fname = QFileDialog.getSaveFileName(self, "Save project", "", "*.tar")[0]
        if not fname:
            return
        self.eval_tab.saveProject(fname)

    def onFileOpenImageClick(self):
        image_path = QFileDialog.getOpenFileName(self, "Load image")[0]
        if len(image_path) == 0:
            self.status_bar.showMessage('No file selected')
            return

        source_image = DUGRImage()
        source_image.load(image_path)
        if source_image.isError():
            self.status_bar.showMessage(source_image.errmsg)
            return

        self.eval_tab.setImage(source_image)
        self.status_bar.showMessage('File import successful')

    def onFileOpenImageSeriesClick(self):
        dlg = QFileDialog(self, "Image series folder")
        dlg.setFileMode(QFileDialog.FileMode.Directory)
        if not dlg.exec():
            return

        path_name = dlg.selectedFiles()[0]

        dlg = LoadTTImageSeries(self, path_name)
        if not dlg.exec():
            return

        image_series = dlg.getSelected()

        # Load image data and insert into data structure
        for image_info in image_series:
            image_info['image'] = DUGRImage()
            success = image_info['image'].load(image_info['fname_full'])
            if not success:
                msg = "Error reading file %s:\n%s" % (image_info['fname'], image_info['image'].getErrorMessage())
                QMessageBox.critical(self, "Error", msg)
                return

        self.eval_tab.setImageSeries(image_series)

    def onFileSettingsClick(self):
        dlg = SettingsDialog(self.main_settings)
        if dlg.exec():
            settings = dlg.getSettings()
            self.main_settings.setValues(settings)
            self.eval_tab.settingsChanged()

    def onCalcExecuteClick(self):
        self.eval_tab.execute()

    def onPlausibilityTestClick(self):
        self.eval_tab.plausibilityTest()

    def onHelpManual(self):
        os.system("doc\\Softwaremanual-DURG-Calculator.pdf")

    def onHelpAbout(self):
        QMessageBox.about(self, "About", "DUGR Calculator v.1.5.0\n\n"
                        "Home: https://github.com/TechnoTeam-Bildverarbeitung-GmbH/DUGR_GUI\n\n"
                        "This program is free software under the\n"
                        "GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007.\n"
                        "Refer to doc/LICENSE.txt or https://www.gnu.org/licenses/")

    def onSwitchProjRectMode(self, flag):
        if flag:
            # Activate projective rectification GUI
            self.eval_tab = ProjectiveCorrUi(self)
            self.setCentralWidget(self.eval_tab)
        else:
            self.eval_tab = ProjectiveDistUi(self)
            self.loadParams()
            self.setCentralWidget(self.eval_tab)

        self.show()

    def _getSettingsFname(self):
        homedir = os.path.expanduser("~")
        fname = homedir + "\\AppData\\Local\\TechnoTeam\\DUGR\\settings.json"
        return fname

    def saveParams(self):
        self.eval_tab.saveParams(self._getSettingsFname())

    def loadParams(self):
        self.eval_tab.loadParams(self._getSettingsFname())


class MainSettings:
    def __init__(self):
        cie232Param = {
            'key': 'cie232Param', 'name': "CIE 232 Parameter",
            'params_list': [
                {'key': 'lum_th', 'name': "Luminance threshold", 'unit': "cd/m²", 'default': 500.0},
                {'key': 'eye_res_d', 'name': "Eye Resolution d", 'unit': "mm", 'default': 12.0},
                {'key': 'use_only_roi', 'name': "Evaluation only inside ROI", 'default': False}
            ]
        }
        # settings data is a list of groups
        self.data = {'groups_list': [cie232Param]}

        ## Build a dict of groups and for each group a dict of params. Complete structure when necessary.
        groups_dict = {}
        for group in self.data['groups_list']:
            params_dict = {}
            for param in group['params_list']:
                # add value item when it doesn't exists
                if 'value' not in param:
                    param['value'] = param['default']

                params_dict[param['key']] = param
            group['params_dict'] = params_dict

            groups_dict[group['key']] = group
        self.data['groups_dict'] = groups_dict

    def get(self):
        return self.data

    def getGroupsList(self):
        return self.get()['groups_list']

    def getGroupsDict(self):
        return self.get()['groups_dict']

    def getParameter(self, group_key, param_key):
        groups_dict = self.getGroupsDict()
        return groups_dict[group_key]['params_dict'][param_key]

    def getParameterValue(self, group_key, param_key):
        return self.getParameter(group_key, param_key)['value']

    def setParameterValue(self, group_key, param_key, value):
        param = self.getParameter(group_key, param_key)
        param['value'] = value

    def setValues(self, other):
        for group in other.getGroupsList():
            for param in group['params_list']:
                self.setParameterValue(group['key'], param['key'], param['value'])

    def getJSON(self):
        json_data = {}
        for group in self.getGroupsList():
            params = {}
            for param in group['params_list']:
                params[param['key']] = param['value']
            json_data[group['key']] = params

        return json_data


class SettingsDialog(QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)

        self.settings = copy.deepcopy(settings)

        self.setWindowTitle("Settings")

        mainLayout = QVBoxLayout()

        self.lines = []

        ###
        for group in self.settings.getGroupsList():
            gb = QGroupBox(group['name'])

            layout = QFormLayout()

            #{'key': 'lum_th', 'name': "Luminance threshold", 'unit': "cd/m²", 'default': 500.0, 'value': 500.0},

            for param in group['params_list']:
                ## Label
                label = param['name']
                if 'unit' in param:
                    label += " [%s]" % param['unit']
                ## Input element depending on type
                value = param['value']
                if isinstance(value, bool):         # Bool -> Checkbox
                    widget = QCheckBox()
                    widget.setChecked(value)
                else:                               # All other -> LineEdit
                    widget = QLineEdit()
                    widget.setText(str(value))

                layout.addRow(QLabel(label), widget)

                self.lines.append({'widget': widget, 'param': param})

            gb.setLayout(layout)
            mainLayout.addWidget(gb)

        ###
        QBtn = QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        mainLayout.addWidget(self.buttonBox)
        self.setLayout(mainLayout)

    def getSettings(self):
        for line in self.lines:
            widget = line['widget']
            param = line["param"]
            value = param['value']
            if isinstance(value, float):
                value = float(widget.text())
            elif isinstance(value, bool):
                value = widget.isChecked()
            elif isinstance(value, int):
                value = int(widget.text())
            else:
                value = widget.text()
            param['value'] = value

        return self.settings


if __name__ == '__main__':
    app = QApplication(argv)
    #app.setStyle('Fusion')
    window = MainWindow()

    def excepthook(exc_type, exc_value, exc_tb):
        tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print("error catched!:")
        print("error message:\n", tb)
    sys.excepthook = excepthook
    app.exec()
    window.saveParams()
    exit()
