"""
Functions to handle common image formats for DUGR calculation
"""
import numpy as np
import re
from os.path import exists


def convert_tt_image_to_numpy_array(image_file: str):
    """
    Function that converts an image from the TechnoTeam image format to a numpy array

    Args:
        image_file: image file in the TechnoTeam image format
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
        image_array = np.frombuffer(pixel_data, dtype=np.float32).reshape(int(header_dict['Lines']),
                                                                          int(header_dict['Columns']), 3)

    return image_array, header_dict


def convert_ascii_image_to_numpy_array(image_file):
    """

    Function that converts an image from ascii format to a numpy array

    Args:
        image_file: Path to the image in the ascii format

    """

    with open(image_file, "r") as file_object:
        pixel_values = file_object.read()
        rows = len([pos for pos, char in enumerate(pixel_values) if char == "\n"])  # Find the number of rows
        pixel_values = re.split('[ \t \n]', pixel_values)  # Split string of file into pixel values
        pixel_values = pixel_values[:-1]  # Pop last element (empty string)
        columns = int(len(pixel_values) / rows)  # Find the number of columns

    image_array = (np.array(pixel_values, dtype=np.float32)).reshape((rows, columns))
    return image_array
