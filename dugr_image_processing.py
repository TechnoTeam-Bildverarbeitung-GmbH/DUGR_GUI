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
import numpy as np
from cv2 import getPerspectiveTransform, warpPerspective, filter2D
from scipy.ndimage import correlate
from math import sqrt, atan2, atan, cos, tan, radians, degrees, exp, log, ceil


def filter_image(image: np.ndarray, filter_width: float, sigma: float):
    """
    This function uses a translation of the MATLAB fspecial('gaussian',[shape],[sigma]) function to calculate a 2D
    gaussian mask (See: https://de.mathworks.com/help/images/ref/fspecial.html#d123e101030).
    The results should be equal to the calculation with MATLAB (within the rounding error).
    The filter is then applied to the image by using the scipy.ndimage.correlate() function with the previously
    calculated 2D gaussian mask.

    Args:
        image: The image on which the filtering is performed on
        filter_width: The width of the gaussian filter
        sigma: The Standard Deviation of the gaussian function -> Higher sigma -> Wider blur radius
    Returns:
         The gaussian filtered image as numpy array

    """

    filter_shape = (filter_width, filter_width)
    m, n = [(ss-1.)/2. for ss in filter_shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return correlate(image, h, mode='nearest')


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


def cart2theta_phi(focal_length, pixel_size, image, opt_x=None, opt_y=None):
    """
    Function to retrieve theta and phi angles from cartesian coordinates and optical axis

    opt_x and opt_y define the coordinates of the optical axis in the image, default value is in the middle of the image

    Args:
        focal_length: Focal length of the camera
        pixel_size: Pixel size of the camera sensor
        image: Input image
        opt_x: x-Coordinates of the optical axis
        opt_y: y -Coordinates of the optical axis

    Returns:
        theta: Numpy array representing the theta angle of each pixel
        phi: Numpy array representing the phi angle of each pixel
    """

    if not opt_x:
        opt_x = image.shape[1] // 2
    if not opt_y:
        opt_y = image.shape[0] // 2

    theta = np.zeros(image.shape)
    phi = np.zeros(image.shape)
    for n in range(image.shape[0]):
        for m in range(image.shape[1]):
            theta[n][m] = np.degrees(atan2(sqrt(((opt_x-m)*pixel_size)**2 + ((opt_y-n)*pixel_size)**2), focal_length))
            phi[n][m] = atan2(radians(opt_y-n), radians(opt_x-m))

    return theta, phi


def img2cart_dist_img(image: np.ndarray, opt_x: int = None, opt_y: int = None):
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
    if not opt_x:
        opt_x = image.shape[1] // 2
    if not opt_y:
        opt_y = image.shape[0] // 2

    cart_dist_img = np.zeros(image.shape)
    for y in range(np.shape(cart_dist_img)[0]):
        for x in range(np.shape(cart_dist_img)[1]):
            cart_dist_img[y, x] = sqrt((opt_x - x) ** 2 + (opt_y - y) ** 2)
    return cart_dist_img


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


def execute_projective_dist_algorithm(src_image: np.ndarray, viewing_distance: float, luminous_area_height: float,
                                      viewing_angle: float, focal_length: float, pixel_size: float, rois: list,
                                      filter_flag: bool, d: int = 12, opt_x: int = None, opt_y: int = None):
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
        filter_flag: When this flag is set to true, only the "Regions of Interest" are gaussian filtered
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
    rb_min = sqrt(viewing_distance**2 + (luminous_area_height/2)**2 + viewing_distance * luminous_area_height *
                  cos(radians(viewing_angle)))

    ro_min = degrees(atan(d/rb_min))/10

    #  Calculate Theta and Phi Image
    theta, phi = cart2theta_phi(focal_length=focal_length, pixel_size=pixel_size, image=src_image, opt_x=opt_x,
                                opt_y=opt_y)

    # y_5deg, x_5deg = np.nonzero(theta <= 5)
    # r_h = (max(y_5deg) - min(y_5deg))/2
    # r_v = (max(x_5deg) - min(x_5deg))/2
    # r_mean = (r_v + r_h)/2
    # r_o_image = 5.0/r_mean

    r_5deg = (tan(radians(5.0)) * focal_length)/pixel_size
    r_o = 5.0 / r_5deg

    fwhm = ro_min/r_o
    sigma = fwhm / 2.3584
    filter_width = 2 * ceil(3 * sigma) + 1

    #  Convert Theta to radians
    theta_rad = np.radians(theta)

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
    theta_filtered_h = (filter2D(theta, -1, kernel_h)) / 2
    theta_filtered_v = (filter2D(theta, -1, kernel_v)) / 2

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
    cart_dist_img = img2cart_dist_img(image=src_image)

    #  Divide the euclidian theta distance in radians by the cartesian distance
    #  Ignore warnings for 0 division because we set our NAN element to 0 manually
    with np.errstate(divide='ignore', invalid='ignore'):
        theta_diff_arc_by_cart_dist = np.nan_to_num(theta_diff_arc / cart_dist_img)

    #  Calculate the omega image (Solid angle image)
    omega = theta_diff_arc_by_cart_dist * sin_theta_rad

    #  Filter the image based on the filter parameters calculated
    filtered_img = []
    if not filter_flag:
        filtered_img.append(filter_image(src_image, filter_width, sigma))
    elif filter_flag:
        for i in range(len(rois)):
            if type(rois[i]).__name__ == 'TrapezoidRoi':
                filtered_img_roi = filter_image(
                    src_image[int(min(rois[i].top_left[1], rois[i].top_right[1]) - ceil(filter_width/2)):int(
                        max(rois[i].bottom_left[1], rois[i].bottom_right[1]) + ceil(filter_width/2)),
                              int(min(rois[i].top_left[0], rois[i].bottom_left[0]) - ceil(filter_width/2)):int(
                                  max(rois[i].top_right[0], rois[i].bottom_right[0]) + ceil(filter_width/2))],
                    filter_width, sigma)
                filtered_img.append(filtered_img_roi)

            elif type(rois[i]).__name__ == 'RectangularRoi' or type(rois[i]).__name__ == 'CircularRoi':
                filtered_img_roi = filter_image(
                    src_image[
                              int(rois[i].top_left[1] - ceil(filter_width / 2)):int(rois[i].bottom_left[1] + ceil(
                                  filter_width / 2)),
                              int(rois[i].top_left[0] - ceil(filter_width / 2)):int(rois[i].top_right[0] + ceil(
                                  filter_width / 2))
                              ], filter_width, sigma)
                filtered_img.append(filtered_img_roi)

    #  Define lists to store parameters corresponding to the pixel values of the threshold
    binarized_img = []
    eff_solid_angle_values = []
    for count, filtered_image_roi in enumerate(filtered_img):
        binarized_img_roi = np.zeros(filtered_image_roi.shape)

        for i in range(filtered_image_roi.shape[0]):
            for j in range(filtered_image_roi.shape[1]):
                if filtered_image_roi[i][j] >= 500:
                    binarized_img_roi[i][j] = filtered_image_roi[i][j]
                    if filter_flag:
                        eff_solid_angle_values.append(omega[i + int(rois[count].top_left[1] - ceil(filter_width / 2))]
                                                      [j + int(rois[count].top_left[0] - ceil(filter_width / 2))])
                    else:
                        eff_solid_angle_values.append(omega[i][j])

        binarized_img.append(binarized_img_roi)

    #  Calculate the effective solid angel by calculating the sum of the pixel solid angles over the luminance threshold
    omega_eff = np.sum(np.array(eff_solid_angle_values))

    # Calculate the effective luminance by calculating the sum of the luminance values over the threshold
    l_eff = []
    for binarized_img_roi in binarized_img:
        l_eff.append(binarized_img_roi[binarized_img_roi != 0].mean())
    l_eff = np.array(l_eff).mean()

    # Calculate the mean luminance l_s of the whole luminaire
    # Calculate the solid angle omega_l of the whole luminaire

    l_values = []
    omega_values = []

    for i in range(len(rois)):

        if type(rois[i]).__name__ == 'TrapezoidRoi':
            # Check whether a point is inside the trapezoid:
            # https://math.stackexchange.com/questions/757591/how-to-determine-the-side-on-which-a-point-lies
            y_start = min(rois[i].roi_vertices[:, 1])
            y_end = max(rois[i].roi_vertices[:, 1])
            x_start = min(rois[i].roi_vertices[:, 0])
            x_end = max(rois[i].roi_vertices[:, 0])

            for y in range(int(y_start), int(y_end+1)):
                for x in range(int(x_start), int(x_end+1)):
                    if (check_point(x, y, rois[i].top_left[0], rois[i].bottom_left[0], rois[i].top_left[1],
                                    rois[i].bottom_left[1]) <= 0) and (check_point(x, y, rois[i].top_right[0],
                                                                                   rois[i].bottom_right[0],
                                                                                   rois[i].top_right[1],
                                                                                   rois[i].bottom_right[1]) >= 0):
                        l_values.append(src_image[y][x])
                        omega_values.append(omega[y][x])

        elif type(rois[i]).__name__ == 'CircularRoi':
            h = rois[i].middle_point_coordinates[0]
            k = rois[i].middle_point_coordinates[1]
            r_x = rois[i].width/2
            r_y = rois[i].height/2
            # Check whether a point is part of the Ellipsoid:
            # https://math.stackexchange.com/questions/76457/check-if-a-point-is-within-an-ellipse
            for y in range(min(rois[i].bounding_box_coordinates[:, 1])-1,
                           max(rois[i].bounding_box_coordinates[:, 1])+1):
                for x in range(min(rois[i].bounding_box_coordinates[:, 0])-1,
                               max(rois[i].bounding_box_coordinates[:, 0])+1):
                    if ((x - h)**2)/(r_x**2) + ((y - k)**2)/(r_y**2) <= 1:
                        l_values.append(src_image[y][x])
                        omega_values.append(omega[y][x])

        elif type(rois[i]).__name__ == 'RectangularRoi':
            for y in range(min(rois[i].roi_coordinates[:, 1]), max(rois[i].roi_coordinates[:, 1])):
                for x in range(min(rois[i].roi_coordinates[:, 0]),
                               max(rois[i].roi_coordinates[:, 0])):
                    l_values.append(src_image[y][x])
                    omega_values.append(omega[y][x])

    l_s = np.array(l_values).mean()
    omega_l = np.array(omega_values).sum()
    k_square = (l_eff**2 * omega_eff)/(l_s**2 * omega_l)

    # Calculate the DUGR value
    dugr = 8 * log(k_square, 10)

    return dugr, k_square, l_eff, l_s, omega_eff, omega_l, r_o, rb_min, ro_min, fwhm, sigma, filter_width,\
        filtered_img, binarized_img
