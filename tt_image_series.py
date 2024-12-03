"""

"""


from PyQt6.QtWidgets import (
    QVBoxLayout,
    QLabel,
    QRadioButton,
    QDialog,
    QGroupBox,
    QDialogButtonBox,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
)
import configparser
import pathlib
import codecs
import copy


class LoadTTImageSeries(QDialog):
    def __init__(self, parent, dname):
        super().__init__(parent)

        # Interpretation of angles: 0: Alpha/Phi (Alpha is horizontal angle), 1: spherical coordinates (Theta, Phi)
        self.angle_mode = 0

        self.setWindowTitle("TechnoTeam image series")
        self.setMinimumWidth(400)

        mainLayout = QVBoxLayout()
        ##
        gb = QGroupBox("Interpretation of angles")
        layout_gb = QVBoxLayout()
        self.angles_no_change_rb = QRadioButton("No change (Alpha, Phi)")
        self.angles_no_change_rb.toggled.connect(self.angles_no_change_rb_change)
        layout_gb.addWidget(self.angles_no_change_rb)
        self.angles_theta_phi_rb = QRadioButton("Spherical coordinates (Theta, Phi)")
        self.angles_theta_phi_rb.toggled.connect(self.angles_theta_phi_rb_change)
        layout_gb.addWidget(self.angles_theta_phi_rb)
        gb.setLayout(layout_gb)
        mainLayout.addWidget(gb)

        ##
        label = QLabel("Image series")
        mainLayout.addWidget(label)

        self.positions_tbl = QTableWidget()
        mainLayout.addWidget(self.positions_tbl)

        ###
        QBtn = QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        mainLayout.addWidget(self.buttonBox)
        self.setLayout(mainLayout)

        self.image_infos = self.scanDirectory(dname)
        # Try to find angle mode from positions
        self._determineAngleMode()

        self._updateUI()


    def scanDirectory(self, dname):
        path = pathlib.Path(dname)
        ini_files = path.glob("*.ini")
        image_infos = []
        for ini_file in ini_files:
            fname = str(ini_file)

            # Solve problems with the possible BOM of Windows files
            with open(fname, "rb") as file:
                content = file.read()
            if content.startswith(codecs.BOM_UTF8):
                content = content[len(codecs.BOM_UTF8):].decode("utf-8")  # remove the BOMb
            elif content.startswith(codecs.BOM_UTF16):
                content = content[len(codecs.BOM_UTF16):].decode("utf-16")  # remove the BOMb

            # Read INI
            config = configparser.ConfigParser()
            config.read_string(content)
            positions_str = config.get('Description', 'SAVEIMAGE_CAPTUREPOS', fallback='')
            positions = positions_str.split(' ')

            fname_img = config.get('Description', 'SAVEIMAGE_FILENAME', fallback='')

            # Dict with some infos about the image
            info = {}
            info['positions'] = [float(positions[0]), float(positions[1])]      # Phi, Theta/Alpha
            info['fname'] = fname_img
            info['fname_full'] = dname + '/' + fname_img

            # Add to results list
            image_infos.append(info)

        return image_infos

    def _determineAngleMode(self):
        # Determine what coordinates are used. Usually C angles are 0/90 or also 180/270 that can easily be
        # differentiated from theta angles.
        for image_info in self.image_infos:
            c = image_info['positions'][0]
            if not (abs(c) < 0.1 or abs(c - 90.0) < 0.1 or abs(c - 180.0) < 0.1 or abs(c - 270.0) < 0.1):
                self.angle_mode = 1  # switch to Theta / Phi mode
                break

    def _updateUI(self):
        self.angles_no_change_rb.setChecked(self.angle_mode == 0)
        self.angles_theta_phi_rb.setChecked(self.angle_mode == 1)

        self.positions_tbl.clear()

        if self.angle_mode == 0:
            self.positions_tbl.setColumnCount(3)
            self.positions_tbl.setHorizontalHeaderLabels(["Alpha", "C", "Filename"])
            self.positions_tbl.setColumnWidth(2, 200)
        else:
            self.positions_tbl.setColumnCount(4)
            self.positions_tbl.setHorizontalHeaderLabels(["Theta", "Alpha", "C", "Filename"])
            self.positions_tbl.setColumnWidth(2, 50)
            self.positions_tbl.setColumnWidth(3, 200)

        self.positions_tbl.setColumnWidth(0, 50)
        self.positions_tbl.setColumnWidth(1, 50)


        self.positions_tbl.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.positions_tbl.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)

        self.positions_tbl.setRowCount(len(self.image_infos))
        for i, image_info in enumerate(self.image_infos):
            if self.angle_mode == 0:
                alpha = image_info['positions'][1]
                c = image_info['positions'][0]
                self.positions_tbl.setItem(i, 0, QTableWidgetItem(str(alpha)))
                self.positions_tbl.setItem(i, 1, QTableWidgetItem(str(c)))
                self.positions_tbl.setItem(i, 2, QTableWidgetItem(image_info['fname']))
            else:
                theta = image_info['positions'][0]
                alpha = theta  - 90.0
                c = image_info['positions'][1]
                self.positions_tbl.setItem(i, 0, QTableWidgetItem(str(theta)))
                self.positions_tbl.setItem(i, 1, QTableWidgetItem(str(alpha)))
                self.positions_tbl.setItem(i, 2, QTableWidgetItem(str(c)))
                self.positions_tbl.setItem(i, 3, QTableWidgetItem(image_info['fname']))

            self.positions_tbl.resizeColumnsToContents()

            # preselect 25 and 40 deg viewing directions
            if abs(alpha-25.0) < 0.1 or abs(alpha-40.0) < 0.1:      # handle position tolerance
                self.positions_tbl.selectRow(i)

    def angles_no_change_rb_change(self):
        if self.sender().isChecked():
            self.angle_mode = 0
            self._updateUI()

    def angles_theta_phi_rb_change(self):
        if self.sender().isChecked():
            self.angle_mode = 1
            self._updateUI()

    def getSelected(self):
        rows = sorted(set([idx.row() for idx in self.positions_tbl.selectedIndexes()]))
        selection = []
        for row in rows:
            image_info = copy.deepcopy(self.image_infos[row])
            # For Theta/Phi image series, the sequence of angles is different.
            if self.angle_mode == 1:
                theta = image_info['positions'][0]
                alpha = theta  - 90.0
                c = image_info['positions'][1]

                image_info['positions'][0] = c
                image_info['positions'][1] = alpha

            selection.append(image_info)

        return selection

