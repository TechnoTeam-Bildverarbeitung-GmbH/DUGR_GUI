"""
Functions to handle image processing steps of the DUGR calculation

- Gaussian filtering
- Projective rectification
- Projective rectification with borders in order to include the extension of luminaires due to blurring by filtering
- Conversion of an image in cartesian coordinate system to theta and phi coordinate systems
- Function that generates an array containing the cartesian distances to an optical axis for each pixel
- Function to calculate the x and y coordinate of a luminance images center of mass
- Function to calculate the positional index of a glare source
- Function to check where a point lies in relation to a line
- Function that executes the calculation of the DUGR value on projective distorted images
"""
import matplotlib.pyplot as plt
import scipy.ndimage
from matplotlib.colors import LogNorm
import numpy as np
from cv2 import getPerspectiveTransform, warpPerspective, filter2D
from scipy.ndimage import correlate, gaussian_filter
from math import sqrt, atan2, atan, cos, tan, radians, degrees, exp, log, ceil
from dugr_image_io import *
from params import *


def filter_image(image: DUGRImage, filter_width: float, sigma: float):
    filter_radius = int((filter_width - 1) / 2)

    filtered_data = gaussian_filter(image.data, sigma, radius=filter_radius)

    filtered_img = DUGRImage(image.bbox)
    filtered_img.data = filtered_data

    return filtered_img

def projective_rectification(image: np.ndarray, rect: np.ndarray, luminaire_width: int, luminaire_height: int):
    """
    Function that performs projective rectification and rescales the image to the resolution of 1[mm/px]

    Args:
        image: The image on which the projective rectification is going to be executed on as numpy array
        rect: The sorted coordinates of the corner points (Order: TopLeft, TopRight, BottomRight, BottomLeft)
        luminaire_width: The physical width of the luminaire (x dimension in the image)
        luminaire_height: The physical height of the luminaire (y dimension in the image)

    Returns:
    """
    dst = np.array([
        [0, 0],
        [luminaire_width - 1, 0],
        [luminaire_width - 1, luminaire_height - 1],
        [0, luminaire_height - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    transformation_matrix = getPerspectiveTransform(rect, dst)

    warped_image = warpPerspective(image, transformation_matrix, (luminaire_width, luminaire_height))
    # return the warped image
    return warped_image


def projective_rectification_with_borders(image, rect, border_size, luminaire_width, luminaire_height):
    """
    This function calculates new points for a projective rectification with borders.
    Those borders are needed to make sure that no luminance information is lost when filtering the image of the
    luminaire.

    Args:
        image: Source image to be rectified
        rect: Coordinates of the polygon points
        border_size: Size of the border in order to filter the image correctly
        luminaire_width: Width of the luminaire
        luminaire_height: Height of the luminaire

    Returns: warped_image_with_borders
    """
    (tl, tr, br, bl) = rect

    tw = tr[0] - tl[0]
    bw = br[0] - bl[0]
    rh = br[1] - tr[1]
    lh = bl[1] - tl[1]

    tw_factor = tw / luminaire_width
    bw_factor = bw / luminaire_width

    rh_factor = rh / luminaire_height
    lh_factor = lh / luminaire_height

    new_tw = (luminaire_width + 2 * border_size) * tw_factor
    new_bw = (luminaire_width + 2 * border_size) * bw_factor
    new_rh = (luminaire_height + 2 * border_size) * rh_factor
    new_lh = (luminaire_height + 2 * border_size) * lh_factor

    tl[0] = round(tl[0] - (new_tw - tw) / 2)
    tl[1] = round(tl[1] - (new_lh - lh) / 2)

    tr[0] = round(tr[0] + (new_tw - tw) / 2)
    tr[1] = round(tr[1] - (new_rh - rh) / 2)

    br[0] = round(br[0] + (new_bw - bw) / 2)
    br[1] = round(br[1] + (new_rh - rh) / 2)

    bl[0] = round(bl[0] - (new_bw - bw) / 2)
    bl[1] = round(bl[1] + (new_lh - lh) / 2)

    rect_with_border = np.array([tl, tr, br, bl])

    dst_border = np.array([
        [0, 0],
        [luminaire_width - 1 + 2*border_size, 0],
        [luminaire_width - 1 + 2*border_size, luminaire_height - 1 + 2*border_size],
        [0, luminaire_height - 1 + 2*border_size]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    transformation_matrix_border = getPerspectiveTransform(rect_with_border, dst_border)

    warped_image_with_borders = warpPerspective(image, transformation_matrix_border, (luminaire_width + 2*border_size,
                                                                                      luminaire_height + 2*border_size))
    # return the warped image
    return warped_image_with_borders






def get_center_of_mass(x_coordinates, y_coordinates, values):
    """
    Function to retrieve the center of mass
    Args:
        x_coordinates: x coordinates of the luminance values over the luminance threshold
        y_coordinates: y coordinates of the luminance values over the luminance threshold
        values: respective luminance values
    Returns:
        center_x: x coordinate of the center of mass
        center_y: y coordinate of the center of mass
    """
    weighted_sum_x = []
    weighted_sum_y = []
    m = np.sum(np.array(values))
    for i in range(len(values)):
        weighted_sum_x.append(values[i]*x_coordinates[i])
        weighted_sum_y.append(values[i]*y_coordinates[i])

    center_x = np.sum(np.array(weighted_sum_x)) / m
    center_y = np.sum(np.array(weighted_sum_y)) / m

    return center_x, center_y


def calculate_positional_index(phi_p, theta_p):
    """
    Function to retrieve the positional index of a luminaire
    Args:
        phi_p: Phi angle corresponding to the center of mass
        theta_p: Theta angle corresponding to the center of mass

    Returns:
        The positional index
    """
    t1 = (35.2 - (0.31889*phi_p) - 1.22 * exp(-2*(phi_p/9))) * theta_p * 10**-3
    t2 = (21 + (0.266667*phi_p) - (0.002963*phi_p**2)) * theta_p**2 * 10**-5
    return exp(t1+t2)


def check_point(x, y, p_x1, p_x2, p_y1, p_y2):
    """
    Function that checks where a point lies in relation to a line
    Args:
        x: x coordinate of point under test
        y: y coordinate of point under test
        p_x1: x coordinate of the first point of the line
        p_x2: x coordinate of the second point of the line
        p_y1: y coordinate of the first point of the line
        p_y2: y coordinate of the second point of the line

    Returns:
    """
    return np.sign((p_x2 - p_x1) * (y - p_y1) - (p_y2 - p_y1) * (x - p_x1))

"""
    Function that executes the calculation of the DUGR value on a projective distorted image

    Args:
        src_image: Source image as numpy array
        viewing_distance: Ideally the distance between the center of the luminaire under testing and the entrance pupil
                          of the optical system
        luminous_area_height: Physical height of the luminous area under testing (y dimension in the image)
        viewing_angle: Angle between camera plane and luminaire plane
        focal_length: Focal length of the optical system used for the measurement
        pixel_size: Size of a pixel of the sensor of the optical system used for the measurement (Pixel Pitch)
        rois: Regions of interest (Regions containing luminous areas)
        lum_th: Threshold for luminance
        filter_only_roi_flag: When this flag is set to true, only the "Regions of Interest" are gaussian filtered
        d: Minimal feature diameter (Default = 12mm)
        opt_x: x coordinate of the optical axis
        opt_y: y coordinate of the optical axis

    Returns:
        dugr: DUGR Value
        k_square: k^2 Value
        l_eff: Effective Luminance
        l_s: Mean Luminance
        omega_eff: Effective Solid Angle
        omega_l: Solid angle of the whole luminous area
        r_o: Measurement resolution
        rb_min: Distance to the furthest Edge of the luminaire in relation to the optical system
        ro_min: Minimal measurement resolution
        fwhm: Full width at half minimum of the gaussian filter
        sigma: Standard deviation of the gaussian filter
        filter_width: Width of the gaussian filter
        filtered_img: Filtered Image as Numpy array
        binarized_img: Binarized Image as Numpy array
"""

class DUGR_ProjectiveDistAlgorithm():
    def __init__(self, src_img:DUGRImage, cparams:ParamGroup, luminous_area_height: float,
                        viewing_angle: float, rois: list,
                        lum_th: int, use_only_roi: bool, d: int = 12, img_center: tuple[float, float] = [],
                        only_plausibility_test=False):
        self.luminous_area_height = luminous_area_height
        self.viewing_angle = viewing_angle
        self.rois = rois
        self.lum_th = lum_th
        self.use_only_roi = use_only_roi
        self.d = d
        self.img_center = img_center
        self.only_plausibility_test = only_plausibility_test

        ## Get parameters from common parameter data
        self.cparams = cparams
        self.viewing_distance = self.cparams.getValue("viewing_distance")
        self.focal_length = self.cparams.getValue("focal_length")
        self.pixel_size = self.cparams.getValue("pixel_size")



        # rb_min: Distance to the furthest edge of the luminaire in relation to the optical system.
        # luminous_area_height is the vertical dimension of the luminaire in this scene.
        self.rb_min = sqrt(self.viewing_distance**2 + (self.luminous_area_height/2)**2 + self.viewing_distance * self.luminous_area_height *
                  cos(radians(self.viewing_angle)))
        # ro_min: Minimal measurement resolution
        self.ro_min = degrees(atan(self.d/self.rb_min))/10

        # In general, the °/pixel ratio is not constant. Thus, the ratio is determined for a 5° angle. An improvement
        # of this method would be to take the half opening angle of the scene.
        r_5deg = (tan(radians(5.0)) * self.focal_length) / self.pixel_size
        self.r_o = 5.0 / r_5deg

        # fwhm: Full width at half minimum of the gaussian filter
        self.fwhm = self.ro_min / self.r_o * 10
        self.sigma = self.fwhm / 2.3548
        self.filter_radius = ceil(3 * self.sigma)
        self.filter_width = 2 * self.filter_radius + 1

        # Source image extended by border with the width of the filter radius
        self.src_img = src_img.getExtended(self.filter_radius)

        if len(self.img_center) == 0:
            self.img_center = self.src_img.bbox.getCenter()

    def cart2theta_phi(self):
        """
        Function to retrieve theta and phi angles from cartesian coordinates and optical axis

        Returns:
            theta: Image representing the theta angle of each pixel
            phi: Image representing the phi angle of each pixel
        """

        self.theta_img = DUGRImage(self.src_img.bbox)
        self.phi_img = DUGRImage(self.src_img.bbox)

        theta_data = self.theta_img.data
        phi_data = self.phi_img.data

        x0, y0 = self.img_center

        for yi, y in self.theta_img.bbox.enumerateY():
            for xi, x in self.theta_img.bbox.enumerateX():
                theta_data[yi][xi] = np.degrees(
                    atan2(sqrt(((x0 - x) * self.pixel_size) ** 2 +
                               ((y0 - y) * self.pixel_size) ** 2), self.focal_length))
                phi_data[yi][xi] = atan2(radians(y0 - y), radians(x0 - x))

    def img2cart_dist_img(self):
        """
        Function to retrieve an image with the cartesian distance to the optical axis.
        If no optical axis coordinates are given, the optical axis defaults to the image center

        Args:
            image: The image in the cartesian coordinate system
            opt_x: x coordinate of the optical axis
            opt_y: y coordinate of the optical axis

        Returns:
             cart_dist_image: Numpy array containing the cartesian distances of each pixel to the optical axis

        """

        self.cart_dist_img = DUGRImage(self.src_img.bbox)
        img_data = self.cart_dist_img.data
        x0, y0 = self.img_center

        for yi, y in self.cart_dist_img.bbox.enumerateY():
            for xi, x in self.cart_dist_img.bbox.enumerateX():
                img_data[yi, xi] = sqrt((x0 - x) ** 2 + (y0 - y) ** 2)

    def calcOmega(self):
        theta_data = self.theta_img.data

        #  Convert Theta to radians
        theta_rad = np.radians(theta_data)

        #  Calculate the sin theta image
        sin_theta_rad = np.sin(theta_rad)

        #  Define Filter kernels for horizontal and vertical filtering of an image
        kernel_h = np.array([[0, 0, 0],
                             [-1, 0, 1],
                             [0, 0, 0]])

        kernel_v = np.array([[0, -1, 0],
                             [0, 0, 0],
                             [0, 1, 0]])

        #  Steps for euclidian distance calculation:
        #  1. Calculate one horizontal and one vertical filtered image
        theta_filtered_h = (filter2D(theta_data, -1, kernel_h)) / 2
        theta_filtered_v = (filter2D(theta_data, -1, kernel_v)) / 2

        #  2. Square the filterd Images
        pow_theta_filtered_h = theta_filtered_h ** 2
        pow_theta_filtered_v = theta_filtered_v ** 2

        #  3. Sum of the squared images
        theta_add = pow_theta_filtered_h + pow_theta_filtered_v

        #  4. Square root of the sum
        theta_diff = np.sqrt(theta_add)

        #  Convert the euclidian distance image to radians
        theta_diff_arc = np.radians(theta_diff)

        #  Calculate the cartesian distance to the image optical axis
        self.img2cart_dist_img()

        #  Divide the euclidian theta distance in radians by the cartesian distance
        #  Ignore warnings for 0 division because we set our NAN element to 0 manually
        with np.errstate(divide='ignore', invalid='ignore'):
            theta_diff_arc_by_cart_dist = np.nan_to_num(theta_diff_arc / self.cart_dist_img.data)

        #  Calculate the omega image (Solid angle image)
        omega = theta_diff_arc_by_cart_dist * sin_theta_rad

        self.omega_img = DUGRImage(self.src_img.bbox)
        self.omega_img.data = omega

    def execute(self):
        #  Calculate Theta and Phi Image
        self.cart2theta_phi()
        self.calcOmega()

        #  Filter the image based on the filter parameters calculated
        self.filtered_img = filter_image(self.src_img, self.filter_width, self.sigma)

        # Generate mask image with threshold operator on filtered image
        mask_threshold_data = self.filtered_img.data >= self.lum_th
        # Generate mask image of ROI shapes
        mask_roi_img = DUGRImage(self.src_img.bbox)
        for roi in self.rois:
            mask_roi_img.data[roi.bbox.getSlice(mask_roi_img.bbox)] = roi.getMaskData()

        # When plausibility test is requested, set ROI of filtered image to 1000. Now, the solid angle of the
        # luminous area multiplied by the square of the radius should be equal to the projected area of the
        # given luminous area dimensions.
        if self.only_plausibility_test:
            self.filtered_img.data = mask_roi_img.data * 1000.0

        # When only calculation inside the ROIs is requested, multiply the threshold mask with ROI mask
        if self.use_only_roi:
            mask_threshold_data = mask_threshold_data * mask_roi_img.data

        # ------------- Statistics --------------
        # Values from ROI areas (not filtered) without threshold operator
        # luminance_roi_mask_data = self.filtered_img.data * mask_roi_img.data
        luminance_roi_mask_data = self.src_img.data * mask_roi_img.data
        omega_roi_mask_data = self.omega_img.data * mask_roi_img.data

        # omega_l: Solid angle of the whole luminous area (inside ROI shapes).
        self.omega_l = np.sum(omega_roi_mask_data)
        # l_s: Mean Luminance (inside ROI shapes)
        self.l_s = np.sum(luminance_roi_mask_data) / np.sum(mask_roi_img.data)
        #print("omega_l: %g, l_s = %g" % (self.omega_l, self.l_s))

        ## Effective values on basis of the threshold image. When calculation only inside ROIs is requested
        ## (use_only_roi - flag), mask_threshold_data excludes areas outside the ROIs.
        # Calculate threshold image and masked omega image
        self.threshold_img = DUGRImage(self.src_img.bbox)
        self.threshold_img.data = self.filtered_img.data * mask_threshold_data
        omega_threshold_mask_data = self.omega_img.data * mask_threshold_data
        # omega_eff: Effective Solid Angle
        self.omega_eff = np.sum(omega_threshold_mask_data)
        # l_eff: Effective Luminance
        self.l_eff = np.sum(self.threshold_img.data) / np.sum(mask_threshold_data)

        #print("omega_eff: %g, l_eff = %g" % (self.omega_eff, self.l_eff))


        return (self.l_eff, self.l_s, self.omega_eff, self.omega_l, self.r_o, self.rb_min,
                self.ro_min, self.fwhm, self.sigma, self.filter_width, self.filtered_img, self.threshold_img)

