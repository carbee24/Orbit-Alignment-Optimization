import numpy as np
import pandas as pd
import math
import torch

# from variables import point_columns

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(vector1, vector2):
    cos_var = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    # print(cos_var)
    return np.arccos(cos_var) * 180 / np.pi

def angle_between_cuda(vector1, vector2):
    cos_var = torch.dot(vector1, vector2) / (torch.linalg.norm(vector1) * torch.linalg.norm(vector2))
    return torch.arccos(cos_var) * 180 / torch.pi

def angle_between_df(vector_df1, vector_df2):
    cos_var = np.dot(vector_df1, vector_df2) / (np.linalg.norm(vector_df1) * np.linalg.norm(vector_df2))
    # print(cos_var)
    return np.arccos(cos_var) * 180 / np.pi
def point2vector(point_group):
    print(point_group[0])
    return point_group - point_group[0]

def point2vector_df(point_group_df):
    print(type(point_group_df.loc[0]))
    return point_group_df - point_group_df.loc[0]

def rotate2X(vector_group):
    degree = angle_between(vector_group[-1], [1, 0])
    # print(degree)
    if degree != 0:
        if vector_group[-1][1] >= 0:
            # print(1)
            uniform_vector_group = Srotate(math.radians(degree), vector_group, 0, 0)
            # print(uniform_vector_group[-1])
            return uniform_vector_group
        else:
            # print(2)
            uniform_vector_group = Nrotate(math.radians(degree), vector_group, 0, 0)
            # print(uniform_vector_group[-1])
            return uniform_vector_group

def rotate2X_index(vector_group, index):
    degree = angle_between(vector_group[index], [1, 0])
    # print(degree)
    if degree != 0:
        if vector_group[-1][1] >= 0:
            # print(1)
            uniform_vector_group = Srotate(math.radians(degree), vector_group, 0, 0)
            # print(uniform_vector_group[-1])
            return uniform_vector_group
        else:
            # print(2)
            uniform_vector_group = Nrotate(math.radians(degree), vector_group, 0, 0)
            # print(uniform_vector_group[-1])
            return uniform_vector_group
    else:
        return vector_group

def rotate2X_index_cuda(vector_group, index):
    vector_group = vector_group.float()
    degree = angle_between_cuda(vector_group[index], torch.tensor([1., 0.], device='cuda'))
    # print(degree)
    if degree != 0:
        if vector_group[-1][1] >= 0:
            # print(1)
            uniform_vector_group = Srotate_cuda(math.radians(degree), vector_group, 0, 0)
            # print(uniform_vector_group[-1])
            return uniform_vector_group
        else:
            # print(2)
            uniform_vector_group = Nrotate_cuda(math.radians(degree), vector_group, 0, 0)
            # print(uniform_vector_group[-1])
            return uniform_vector_group
    else:
        return vector_group

def rotate2X_df(vector_group_df):
    degree = angle_between(vector_group_df[-1:], [1, 0])
    if degree != 0:
        if vector_group_df.iloc[-1]['y'] >= 0:
            # print(1)
            uniform_vector_group_df = Srotate_df(math.radians(degree), vector_group_df, 0, 0)
            # print(uniform_vector_group[-1])
            return uniform_vector_group_df
        else:
            # print(2)
            uniform_vector_group_df = Nrotate_df(math.radians(degree), vector_group_df, 0, 0)
            # print(uniform_vector_group[-1])
            return uniform_vector_group_df
def angle_diff(vector_group):
    angle_diff_group = []
    for i in range(len(vector_group)):
        if i == 0 or i == 1:
            continue
        else:
            angle_diff_group.append(angle_between(vector_group[i - 1], vector_group[i]))

    return np.array(angle_diff_group)

def Nrotate(angle,vector_group,pointx,pointy):
  valuex = vector_group[:, 0]
  valuey = vector_group[:, 1]
  nRotatex = (valuex-pointx)*math.cos(angle) - (valuey-pointy)*math.sin(angle) + pointx
  nRotatey = (valuex-pointx)*math.sin(angle) + (valuey-pointy)*math.cos(angle) + pointy
  return np.column_stack((nRotatex, nRotatey))

def Nrotate_cuda(angle,vector_group,pointx,pointy):
  valuex = vector_group[:, 0]
  valuey = vector_group[:, 1]
  nRotatex = (valuex-pointx)*math.cos(angle) - (valuey-pointy)*math.sin(angle) + pointx
  nRotatey = (valuex-pointx)*math.sin(angle) + (valuey-pointy)*math.cos(angle) + pointy
  return torch.column_stack((nRotatex, nRotatey))
# 绕pointx,pointy顺时针旋转

def Nrotate_df(angle, point_group_df, pointx, pointy):
    valuex = point_group_df['x']
    valuey = point_group_df['y']
    nRotatex = (valuex - pointx) * math.cos(angle) - (valuey - pointy) * math.sin(angle) + pointx
    nRotatey = (valuex - pointx) * math.sin(angle) + (valuey - pointy) * math.cos(angle) + pointy
    return pd.DataFrame(np.column_stack((nRotatex, nRotatey)), columns=point_columns)

def Srotate(angle,vector_group,pointx,pointy):
  valuex = vector_group[:, 0]
  valuey = vector_group[:, 1]
  sRotatex = (valuex-pointx)*math.cos(angle) + (valuey-pointy)*math.sin(angle) + pointx
  sRotatey = (valuey-pointy)*math.cos(angle) - (valuex-pointx)*math.sin(angle) + pointy
  return np.column_stack((sRotatex, sRotatey))

def Srotate_cuda(angle,vector_group,pointx,pointy):
  valuex = vector_group[:, 0]
  valuey = vector_group[:, 1]
  sRotatex = (valuex-pointx)*math.cos(angle) + (valuey-pointy)*math.sin(angle) + pointx
  sRotatey = (valuey-pointy)*math.cos(angle) - (valuex-pointx)*math.sin(angle) + pointy
  return torch.column_stack((sRotatex, sRotatey))

def Srotate_df(angle, point_group_df, pointx, pointy):
    valuex = point_group_df['x']
    valuey = point_group_df['y']
    sRotatex = (valuex - pointx) * math.cos(angle) + (valuey - pointy) * math.sin(angle) + pointx
    sRotatey = (valuey - pointy) * math.cos(angle) - (valuex - pointx) * math.sin(angle) + pointy
    return pd.DataFrame(np.column_stack((sRotatex, sRotatey)), columns=point_columns)

def line_angle(line_vector):
    if line_vector[1] == 0:
        if line_vector[0] >= 0:
            degree = 0
            return degree
        else:
            degree = 180
            return degree
    else:

        tan_var = line_vector[0]/line_vector[1]

        if line_vector[0] < 0:
            if line_vector[0] * line_vector[1] < 0:
                degree = np.degrees(np.arctan(tan_var)) + 180
                return degree
            else:
                degree = np.degrees(np.arctan(tan_var)) - 180
                return degree
        else:
            degree = np.degrees(np.arctan(tan_var))
            return degree

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    x_tem = np.linspace(0, 0.75 * np.pi, num=100)
    y_tem = np.sin(x_tem)
    plt.plot(x_tem, y_tem, c='r', linewidth=4)
    point_group = np.column_stack((x_tem, y_tem))
    rotate_point_group = Srotate(math.radians(45), point_group, 0, -1)
    plt.plot(rotate_point_group[:, 0], rotate_point_group[:, 1], c='g', linewidth=4)
    rotate_vector = point2vector(rotate_point_group)
    # print(rotate_vector)
    plt.plot(rotate_vector[:, 0], rotate_vector[:, 1], c='y', linewidth=4)
    uniform_vector = rotate2X(rotate_vector)
    plt.plot(uniform_vector[:, 0], uniform_vector[:, 1], linewidth=4)


    point_group_df = pd.DataFrame(point_group, columns=point_columns)
    plt.plot(point_group_df['x'], point_group_df['y'], c='b')
    rotate_point_group_df = Srotate_df(math.radians(45), point_group_df, 0, -1)
    plt.plot(rotate_point_group_df['x'], rotate_point_group_df['y'])
    rotate_vector_df = point2vector_df(rotate_point_group_df)
    # print(rotate_vector_df)
    plt.plot(rotate_vector_df['x'], rotate_vector_df['y'], c='b')
    uniform_vector_df = rotate2X_df(rotate_vector_df)
    plt.plot(uniform_vector_df['x'], uniform_vector_df['y'], c='black')
    plt.show()
