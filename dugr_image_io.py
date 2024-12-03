"""
Functions to handle common image formats for DUGR calculation

Currently available:
    - TechnoTeam Image formats: *.pus, *.pf, *.pcf
    - Ascii image: A textfile containing the pixel information
"""
import numpy as np
import re
from os.path import exists
from roi_definitions import Bbox
from params import *


def convert_tt_image_to_numpy_array(image_file: str):
    """
    Function that converts an image from the TechnoTeam image format to a numpy array

    Args:
        image_file: image file in one of the TechnoTeam image formats (*.pus, *.pf, *.pcf)

    Returns:
         image_array: Numpy array containing the pixel information
         header dict: Dictionary with all the image information stored in the TechnoTeam image headers
    """

    if not exists(image_file):
        print("\nThe entered image file does not exist!")
        return np.empty((0, 0)), {}

    if image_file[-3:] != 'pus' and image_file[-2:] != 'pf' and image_file[-3:] != 'pcf':
        print("The entered image file is not part of the TechnoTeam image file format!\nValid formats are:\n\t*.pus"
              "\n\t*.pf\n\t*.pcf")
        return np.empty((0, 0)), {}

    with open(image_file, "rb") as imageInput:
        data = imageInput.read()

    r = data.find(b'\x00')  # Finding 0 byte which separates header and pixel data

    header = data[:r].decode('utf-8')  # utf-8 decode header to text
    header_fmt = header.replace('\r', '')  # header formatting to pack key/value pair to dict
    header_fmt = header_fmt.replace('|', '')
    while header_fmt[-1] == '\n':  # Remove line breaks at the end of the header for header to dict conversion
        header_fmt = header_fmt[:-1]
    header_dict = dict(entry.split('=') for entry in header_fmt.split('\n'))  # Pack dict

    pixel_data = data[r + 1:]

    image_array = np.empty((0, 0))
    try:
        #  Camera Image
        if header_dict['Typ'] == 'Pic98::TPlane<unsigned short>':
            image_array = np.frombuffer(pixel_data, dtype='<H').reshape(int(header_dict['Lines']),
                                                                        int(header_dict['Columns']))

        #  Luminance Image
        elif header_dict['Typ'] == "Pic98::TPlane<float>":
            image_array = np.frombuffer(pixel_data, dtype=np.float32).reshape(int(header_dict['Lines']),
                                                                              int(header_dict['Columns']))

        # Color Image
        elif header_dict['Typ'] == "Pic98::TPlane<Pic98::TRGBFloatPixel>":
            image_array = np.frombuffer(pixel_data, dtype=np.float32).reshape((int(header_dict['Lines']),
                                                                               int(header_dict['Columns']), 3))
    except KeyError:
        print("The file type is one of the expected TechnoTeam formats (*.pus, *.pf. *.pcf)"
              "\nBut the header seems to be corrupted"
              "\nThe image type is not defined in the header")
        image_array = np.empty((0, 0))
        return image_array, header_dict

    return image_array, header_dict


def convert_ascii_image_to_numpy_array(image_file: str):
    """
    Function that converts an image from ascii format to a numpy array

    Args:
        image_file: Path to the image in the ascii format (*.txt). Each new line represents a new image row.

    Returns:
        image_array: Numpy array containing the pixel information
    """

    if image_file[-3:] != 'txt':
        print("The entered image file has to be of type *.txt")
        return np.empty((0, 0)), {}

    with open(image_file, 'r') as file_object:
        pixel_values = file_object.read().replace(',', '.').split('\n')[2:-1]

    image_array = np.genfromtxt(pixel_values, dtype=np.float32, delimiter='\t')
    return image_array


class DUGRImage:
    def __init__(self, bbox=None):
        self._init(bbox)

    def _init(self, bbox=None):
        self.data = None
        self.tt_header = None
        self.lines = 0
        self.columns = 0
        self.first_line = 0
        self.first_column = 0
        self.errmsg = ""
        self.bbox = bbox

        if bbox != None:
            self.first_column, self.first_line = bbox.getFirstPoint()
            self.columns, self.lines = bbox.getSize()
            self.data = np.zeros((self.lines, self.columns))

    def isError(self):
        return len(self.errmsg) > 0

    def getErrorMessage(self):
        return self.errmsg

    def load(self, filename):
        self._init()
        if not exists(filename):
            self.errmsg = "File does not exist."
            return False

        if filename[-2:] != 'pf' and filename[-3:] != 'txt':
            self.errmsg = "File type not supported.\nMake sure to load a *.pf  or *.txt File."
            return False

        if filename[-2:] == 'pf':
            self.data, self.tt_header = convert_tt_image_to_numpy_array(filename)
            self.first_line = int(self.tt_header['FirstLine'])
            self.first_column = int(self.tt_header['FirstColumn'])
        else:
            self.data = convert_ascii_image_to_numpy_array(filename)

        self.lines = self.data.shape[0]
        self.columns = self.data.shape[1]

        bbox_a = [[self.first_column, self.first_line], [self.first_column+self.columns, self.first_line+self.lines]]
        self.bbox = Bbox(bbox_a)

        return True

    def getBBox(self):
        return self.bbox

    def getSubImage(self, bbox):
        new_image = DUGRImage(bbox)

        new_image.data = self.data[bbox.getSlice(self.bbox)]
        if self.tt_header:
            new_image.tt_header = self.tt_header.copy()
            new_image.tt_header['FirstLine'] = new_image.first_line
            new_image.tt_header['FirstColumn'] = new_image.first_column
            new_image.tt_header['Lines'] = new_image.lines
            new_image.tt_header['Columns'] = new_image.columns
        return new_image

    def getExtended(self, border):
        new_image = DUGRImage(self.bbox.getExtended(border))
        new_image.data[self.bbox.getSlice(new_image.bbox)] = self.data
        return new_image

    def getMax(self):
        return np.max(self.data)

    def getExtent(self):
        return [self.first_column, self.first_column+self.columns-1, self.first_line+self.lines-1, self.first_line]

    def saveToTar(self, tar, fname_prefix):
        if self.data is None:
            return

        # create a temporary param set for easy storage to tar
        params = ParamGroup('img', "")
        params.addParam(key='lines', name="", default=int(self.lines))
        params.addParam(key='columns', name="", default=int(self.columns))
        params.addParam(key='first_line', name="", default=int(self.first_line))
        params.addParam(key='first_column', name="", default=int(self.first_column))

        # save image parameter to tar
        params.writeToTar(tar, "%s_param" % (fname_prefix))

        # Save image numpy array
        stream = io.BytesIO()
        np.save(stream, self.data)
        stream.seek(0)
        info = tarfile.TarInfo("%s_data.npy" % (fname_prefix))
        info.size = len(stream.getvalue())
        tar.addfile(info, stream)

    def loadFromTar(self, tar, tar_content, fname_prefix):
        # create a temporary param set for easy reading from tar
        params = ParamGroup('img', "")
        params.addParam(key='lines', name="", default=0)
        params.addParam(key='columns', name="", default=0)
        params.addParam(key='first_line', name="", default=0)
        params.addParam(key='first_column', name="", default=0)

        # read image parameter when exist
        if not params.readFromTar(tar, tar_content, "%s_param" % (fname_prefix)):
            return False

        # read image numpy array
        # test if file exists
        img_fname = "%s_data.npy" % (fname_prefix)
        if img_fname not in tar_content:
            return False
        # Read numpy array. BytesIO workaround necessary,
        f = tar.extractfile(img_fname)
        stream = io.BytesIO()
        stream.write(f.read())
        stream.seek(0)
        data = np.load(stream)

        # init
        lines = params.getValue('lines')
        columns = params.getValue('columns')
        first_line = params.getValue('first_line')
        first_column = params.getValue('first_column')

        bbox_a = [[first_column, first_line], [first_column+columns, first_line+lines]]
        bbox = Bbox(bbox_a)
        self._init(bbox)
        self.data = data

        return True

