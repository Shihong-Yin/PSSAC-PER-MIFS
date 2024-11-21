import numpy as np
from math import sin, cos, pi, tan


def wraptopi(radian):
    # -pi to pi

    if radian > pi:
        radian2 = radian - 2 * pi
    elif radian < -pi:
        radian2 = radian + 2 * pi
    else:
        radian2 = radian

    return radian2


def generate_circle_points(center, radius, num_points, initial_angle):
    """ 在圆周上间隔相同位置取点 """
    h, k = center
    angles = np.linspace(initial_angle, initial_angle + 2 * np.pi, num_points, endpoint=False)
    points = [(h + radius * np.cos(angle), k + radius * np.sin(angle)) for angle in angles]
    return points


def calculate_angle(vector1, vector2):
    """ 计算两个向量之间的带符号的角度，顺时针为正角度，值域为[-180, 180] """
    epsilon = np.finfo(float).eps  # 一个非常小的数，以避免除0
    dot_product = np.dot(vector1, vector2)  # 计算点积
    norm1 = np.linalg.norm(vector1)  # 计算向量的模
    norm2 = np.linalg.norm(vector2)
    cos_theta = dot_product / (norm1 * norm2 + epsilon)  # 计算 cos(theta)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 裁剪 cos_theta 以防止浮点数精度误差
    angle = np.degrees(np.arccos(cos_theta))  # 计算角度
    # 计算叉积以确定角度的符号
    cross_product = np.cross(vector1, vector2)
    if cross_product > 0:
        angle = -angle
    return angle
