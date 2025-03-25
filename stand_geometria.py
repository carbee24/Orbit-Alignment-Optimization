from scipy import optimize as op
from scipy.linalg import solve
import numpy as np
import math

# def line_func(x, K, B):
#     return K * x + B


def twoPoint2Line(point1, point2):
    #### Ax + By + C = 0
    # x_y = list(zip(point1, point2))
    # K, B = op.curve_fit(line_func, x_y[0], x_y[1])[0]
    A = point2[1] - point1[1]
    B = point1[0] - point2[0]
    C = point2[0] * point1[1] - point1[0] * point2[1]
    return A, B, C

def twoLine2Point(ABC_tuple1, ABC_tuple2):
    a = np.array([list(ABC_tuple1)[:2], list(ABC_tuple2)[:2]])
    b = np.array([-1 * list(ABC_tuple1)[2], -1 * list(ABC_tuple2)[2]])
    point = solve(a, b)
    return point

def getGenenalExprVertical(ABC_tuple, point):
    C = ABC_tuple[1] * point[0] - ABC_tuple[0] * point[1]
    return -1 * ABC_tuple[1], ABC_tuple[0], C

def getGenenalExprDistance(ABC_tuple, distance):
    f = distance * math.sqrt(ABC_tuple[0] * ABC_tuple[0] + ABC_tuple[1] * ABC_tuple[1])
    c1 = ABC_tuple[2] + f
    c2 = ABC_tuple[2] - f
    return (ABC_tuple[0], ABC_tuple[1], c1), (ABC_tuple[0], ABC_tuple[1], c2)


if __name__ == "__main__":
    # kb_tuple1 = twoPoint2Line([1, 2.36], [3, 6.36])
    #
    # kb_tuple2 = twoPoint2Line([1.6, 0], [1.6, 0.5])
    # point = twoLine2Point(kb_tuple1, kb_tuple2)
    # print(point)
    point1 = [350, 275]
    point2 = [350, 325]
    line1_abc = twoPoint2Line(point1, point2)
    line1_c_abc = getGenenalExprVertical(line1_abc, point1)
    line1_distance_abc = getGenenalExprDistance(line1_abc, 5)
    print(line1_abc)
    print(line1_c_abc)
    print(line1_distance_abc)
    point3 = twoLine2Point(line1_c_abc, line1_distance_abc[0])
    point4 = twoLine2Point(line1_c_abc, line1_distance_abc[1])
    print(point3)
    print(point4)