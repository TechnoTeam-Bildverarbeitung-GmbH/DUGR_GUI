"""
Script that contains Region of Interest classes for the GUI

- Rectangular ROI
- Circular ROI
- Trapezoid ROI
"""
import numpy as np


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

        self.area = np.pi * (self.width / 2) * (self.height / 2)

        shifted_middle_point_x = self.middle_point_coordinates[0] - self.bounding_box_coordinates[0][0]
        shifted_middle_point_y = self.middle_point_coordinates[1] - self.bounding_box_coordinates[0][1]

        # luminance_values = []
        # for y in range(self.bounding_box.shape[0]):
        #     for x in range(bounding_box.shape[1]):
        #         if (((x - shifted_middle_point_x) ** 2) / ((self.width / 2) ** 2) + (
        #                 (y - shifted_middle_point_y) ** 2) / (self.height / 2) ** 2) <= 1:
        #             luminance_values.append(bounding_box[y][x])

        self.luminance_values = np.zeros(self.bounding_box.shape)
        for y in range(self.bounding_box.shape[0]):
            for x in range(bounding_box.shape[1]):
                if (((x - shifted_middle_point_x) ** 2) / ((self.width / 2) ** 2) + (
                        (y - shifted_middle_point_y) ** 2) / (self.height / 2) ** 2) <= 1:
                    self.luminance_values[y][x] = self.bounding_box[y][x]

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

        self.x_anchor = min(self.top_left[0], self.top_right[0], self.bottom_right[0], self.bottom_left[0])
        self.y_anchor = min(self.top_left[1], self.top_right[1], self.bottom_right[1], self.bottom_left[1])

        self.d1_x = [self.top_left[0] - self.x_anchor, self.top_right[0] - self.x_anchor]
        self.d1_y = [self.top_left[1] - self.y_anchor, self.top_right[1] - self.y_anchor]
        self.d2_x = [self.top_right[0] - self.x_anchor, self.bottom_right[0] - self.x_anchor]
        self.d2_y = [self.top_right[1] - self.y_anchor, self.bottom_right[1] - self.y_anchor]
        self.d3_x = [self.bottom_right[0] - self.x_anchor, self.bottom_left[0] - self.x_anchor]
        self.d3_y = [self.bottom_right[1] - self.y_anchor, self.bottom_left[1] - self.y_anchor]
        self.d4_x = [self.bottom_left[0] - self.x_anchor, self.top_left[0] - self.x_anchor]
        self.d4_y = [self.bottom_left[1] - self.y_anchor, self.top_left[1] - self.y_anchor]

        self.bounding_box = src_image[int(min(self.top_left[1], self.top_right[1])):int(max(self.bottom_left[1],
                                                                                            self.bottom_right[1])),
                                      int(min(self.top_left[0], self.bottom_left[0])):int(max(self.top_right[0],
                                                                                              self.bottom_right[0]))]

        self.height_left = self.bottom_left[1] - self.top_left[1]
        self.height_right = self.bottom_right[1] - self.top_right[1]

        mean_height = (self.height_right + self.height_left)/2

        self.area = (mean_height * (self.width_top + self.width_bottom))/2