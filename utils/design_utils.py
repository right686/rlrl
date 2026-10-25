import numpy as np


def design_from_solution(x):
    x = np.asarray(x, dtype=np.float64)
    return [
        x[0], x[1], x[2], x[3], x[4],
        x[3], x[3], x[2],
        x[1] ** 2 * x[3],
        x[0] ** 2 * x[2],
        x[0] ** 2 * x[2],
        x[4], x[4], x[4], x[4], x[4], x[4],
        x[4], x[4], x[4], x[4], x[4], x[4],
    ]


def body_parameter_range():
    pole_r_range = [0.8, 1.2]
    pole_length_range = [0.8, 1.2]
    motion_platform_range = [0.8, 1.2]
    polar_range1 = [135, 165]
    azimuth_range1 = [29.3, 70]
    polar_range2 = [135, 165]
    azimuth_range2 = [-40, -29.3]
    return pole_r_range, pole_length_range, motion_platform_range, polar_range1, azimuth_range1, polar_range2, azimuth_range2


def denormalize_angle_variables(normalized_angles):
    _, _, _, polar_range1, azimuth_range1, polar_range2, azimuth_range2 = body_parameter_range()
    normalized_angles = np.asarray(normalized_angles, dtype=np.float64).reshape(4)
    ranges = np.array(
        [
            [polar_range1[0], polar_range1[1]],
            [azimuth_range1[0], azimuth_range1[1]],
            [polar_range2[0], polar_range2[1]],
            [azimuth_range2[0], azimuth_range2[1]],
        ],
        dtype=np.float64,
    )
    return ranges[:, 0] + normalized_angles * (ranges[:, 1] - ranges[:, 0])
