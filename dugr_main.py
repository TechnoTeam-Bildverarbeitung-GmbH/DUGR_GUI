"""
MAIN SCRIPT of the GUI

Defines the main window and calls the different algorithm files
"""
from projective_distorted_algorithm import ProjectiveDistUi
from projective_corrected_algorithm import ProjectiveCorrUi
from sys import argv, exit
from os.path import dirname, join
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
)

basedir = dirname(__file__)


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.resize(1400, 900)
        self.start_projective_distorted_ui()

        self.setWindowTitle("DUGR GUI")
        self.setWindowIcon(QIcon(join(basedir, "assets/light-bulb.ico")))

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


if __name__ == '__main__':
    app = QApplication(argv)
    app.setStyle('Fusion')

    window = MainWindow()
    exit(app.exec())
