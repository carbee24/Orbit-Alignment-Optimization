import pymunk
import numpy as np
import stand_geometria
from stand_action_bound import  *


# def getGenenalExpr(point1, point2):
#     A = point2[1] - point1[1]
#     B = point1[0] - point2[0]
#     C = point2[0] * point1[1] - point1[0] * point2[1]
#     return A, B, C
class Stand6():
    def __init__(self, space, adjust_point_group = None, manget_point_group = None, position = None):
        self.space = space
        self.adjust_point_group = adjust_point_group
        self.manget_point_group = manget_point_group
        self.target_point_group = manget_point_group
        self.ACTIONS = np.array(range(6))
        # self.STATE = np.array(range(10))
        self.body_1 = pymunk.Body()
        self.body_1.position = position
        # body_1.mass = 1
        self.line1 = pymunk.Segment(self.body_1, (-100, -50), (-100, 50), 0)
        self.line2 = pymunk.Segment(self.body_1, (-100, -50), (100, -50), 0)
        self.line3 = pymunk.Segment(self.body_1, (100, -50), (100, 50), 0)
        self.line4 = pymunk.Segment(self.body_1, (-100, 50), (100, 50), 0)
        self.line_list = [self.line1, self.line2, self.line3, self.line4]
        # for line in self.line_list:
        #     line.mass = 1
        # line1.mass = 1
        # line2.mass = 1
        # line3.mass = 1
        # line4.mass = 1

        self.manget_point1 = pymunk.Circle(self.body_1, 3, (-80, 10))  #### refer_point1 (220, 310)
        self.manget_point2 = pymunk.Circle(self.body_1, 3, (-40, -10))
        self.manget_point3 = pymunk.Circle(self.body_1, 3, (0, 0))
        self.manget_point4 = pymunk.Circle(self.body_1, 3, (40, 10))  #### refer_point2 (340, 310)
        self.manget_point5 = pymunk.Circle(self.body_1, 3, (80, -10))
        self.manget_point_list = [self.manget_point1, self.manget_point2, self.manget_point3, self.manget_point4, self.manget_point5]

        self.body_2 = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body_2.position = position
        ### line1 points ###

        self.c1 = pymunk.Circle(self.body_2, 2, (-105, -20))
        self.c2 = pymunk.Circle(self.body_2, 2, (-95, -20))
        self.c3 = pymunk.Circle(self.body_2, 2, (-95, 20))
        self.c4 = pymunk.Circle(self.body_2, 2, (-105, 20))

        ### line2 points ###
        self.c5 = pymunk.Circle(self.body_2, 2, (0, -55))
        self.c6 = pymunk.Circle(self.body_2, 2, (0, -45))
        ### line3 points ###
        self.c7 = pymunk.Circle(self.body_2, 2, (95, -20))
        self.c8 = pymunk.Circle(self.body_2, 2, (105, -20))
        self.c9 = pymunk.Circle(self.body_2, 2, (105, 20))
        self.c10 = pymunk.Circle(self.body_2, 2, (95, 20))

        ### line4 points ###
        self.c11 = pymunk.Circle(self.body_2, 2, (0, 45))
        self.c12 = pymunk.Circle(self.body_2, 2, (0, 55))
        # line3.mass = 1
        space.add(self.body_1)
        space.add(self.line1, self.line2, self.line3, self.line4)
        space.add(self.manget_point1, self.manget_point2, self.manget_point3, self.manget_point4, self.manget_point5)
        space.add(self.body_2)
        space.add(self.c1, self.c2, self.c3, self.c4)
        space.add(self.c5, self.c6)
        space.add(self.c7, self.c8, self.c9, self.c10)
        space.add(self.c11, self.c12)

    def body_move(self, ACTION, movement_tuple):
        DISPLACE_INC, ROTATE_INC = movement_tuple
        if ACTION == 0:
            self.body_1.position += DISPLACE_INC, 0
        elif ACTION == 1:
            self.body_1.position -= DISPLACE_INC, 0
        elif ACTION == 2:
            self.body_1.position += 0, DISPLACE_INC
        elif ACTION == 3:
            self.body_1.position -= 0, DISPLACE_INC
        elif ACTION == 4:
            self.body_1.angle += ROTATE_INC
        else:
            self.body_1.angle -= ROTATE_INC

    def continuous_body_move(self, movement):
        self.body_1.position += movement[:2]
        self.body_1.angle += movement[2]

    @property
    def body_state(self):
        return [self.body_1.position[0], self.body_1.position[1], self.body_1.angle]

    @property
    def manget_state(self):
        manget_points = np.zeros((len(self.manget_point_list), 2))
        for i in range(len(self.manget_point_list)):
            point = self.body_state()[:2] + self.manget_point_list[i].offset.rotated(self.body_state()[2])
            manget_points[i] = point

        return manget_points

    @property
    def state(self):
        manget_points = self.manget_state
        return manget_points.flatten()

    def action_sample(self):
        action = np.random.choice(self.ACTIONS)
        return np.array([np.where(self.ACTIONS == action)[0]])

    # def init_body_position(self, adjust_point_group):
    #     body_position = (adjust_point_group[0] + adjust_point_group[1]) / 2
    #     return body_position

class Stand6_Template():
    def __init__(self, space, adjust_point_group = None, manget_point_group = None, distance = 10):
        self.space = space
        self.adjust_point_group = adjust_point_group
        self.manget_point_group = manget_point_group
        self.target_point_group = manget_point_group
        self.distance = distance
        self.ACTIONS = np.array(range(6))

        position = (self.adjust_point_group[0] + self.adjust_point_group[1]) / 2
        self.body_1 = pymunk.Body()
        self.body_1.position = list(position)

        self.line1_ABC = stand_geometria.twoPoint2Line(self.adjust_point_group[2], self.adjust_point_group[3])
        self.line2_ABC = stand_geometria.getGenenalExprVertical(self.line1_ABC, self.adjust_point_group[0])
        # self.line2_ABC = -1 * self.line1_ABC[1], self.line1_ABC[0], line2_C
        self.line3_ABC = stand_geometria.twoPoint2Line(self.adjust_point_group[4], self.adjust_point_group[5])
        self.line4_ABC = stand_geometria.getGenenalExprVertical(self.line1_ABC, self.adjust_point_group[1])
        # self.line4_ABC = -1 * self.line1_ABC[1], self.line1_ABC[0], line4_C

        cross_point1 = stand_geometria.twoLine2Point(self.line1_ABC, self.line2_ABC)
        cross_point2 = stand_geometria.twoLine2Point(self.line2_ABC, self.line3_ABC)
        cross_point3 = stand_geometria.twoLine2Point(self.line3_ABC, self.line4_ABC)
        cross_point4 = stand_geometria.twoLine2Point(self.line4_ABC, self.line1_ABC)
        cross_points = [cross_point1, cross_point2, cross_point3, cross_point4]
        cross_point_offset = cross_points - position
        self.line1 = pymunk.Segment(self.body_1, list(cross_point_offset[3]), list(cross_point_offset[0]), 0)
        self.line2 = pymunk.Segment(self.body_1, list(cross_point_offset[0]), list(cross_point_offset[1]), 0)
        self.line3 = pymunk.Segment(self.body_1, list(cross_point_offset[1]), list(cross_point_offset[2]), 0)
        self.line4 = pymunk.Segment(self.body_1, list(cross_point_offset[2]), list(cross_point_offset[3]), 0)
        self.line_list = [self.line1, self.line2, self.line3, self.line4]
        for line in self.line_list:
            line.mass = 1


        magnet_point_offset = self.manget_point_group - position
        # self.manget_point1 = pymunk.Circle(self.body_1, 3, list(magnet_point_offset[0]))  #### refer_point1 (220, 310)
        # self.manget_point2 = pymunk.Circle(self.body_1, 3, list(magnet_point_offset[1]))
        # self.manget_point3 = pymunk.Circle(self.body_1, 3, list(magnet_point_offset[2]))
        # self.manget_point4 = pymunk.Circle(self.body_1, 3, list(magnet_point_offset[3]))  #### refer_point2 (340, 310)
        # self.manget_point5 = pymunk.Circle(self.body_1, 3, list(magnet_point_offset[4]))
        # self.manget_point_list = [self.manget_point1, self.manget_point2, self.manget_point3, self.manget_point4,
        #                           self.manget_point5]
        self.manget_point_list = []
        for offset in magnet_point_offset:
            self.manget_point_list.append(pymunk.Circle(self.body_1, 1, list(offset)))


        self.body_2 = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body_2.position = list(position)
        ### line1 points ###
        line1_V_abc1 = stand_geometria.getGenenalExprVertical(self.line1_ABC, self.adjust_point_group[2])
        line1_V_abc2 = stand_geometria.getGenenalExprVertical(self.line1_ABC, self.adjust_point_group[3])
        line1_dis_abc = stand_geometria.getGenenalExprDistance(self.line1_ABC, self.distance)
        fix_point1 = stand_geometria.twoLine2Point(line1_V_abc1, line1_dis_abc[0])
        fix_point2 = stand_geometria.twoLine2Point(line1_V_abc1, line1_dis_abc[1])
        fix_point3 = stand_geometria.twoLine2Point(line1_V_abc2, line1_dis_abc[0])
        fix_point4 = stand_geometria.twoLine2Point(line1_V_abc2, line1_dis_abc[1])
        self.c1 = pymunk.Circle(self.body_2, 0.5, list(fix_point1 - position))
        self.c2 = pymunk.Circle(self.body_2, 0.5, list(fix_point2 - position))
        self.c3 = pymunk.Circle(self.body_2, 0.5, list(fix_point3 - position))
        self.c4 = pymunk.Circle(self.body_2, 0.5, list(fix_point4 - position))

        ### line2 points ###
        line2_V_abc = stand_geometria.getGenenalExprVertical(self.line2_ABC, self.adjust_point_group[0])
        line2_dis_abc = stand_geometria.getGenenalExprDistance(self.line2_ABC, self.distance)
        fix_point5 = stand_geometria.twoLine2Point(line2_V_abc, line2_dis_abc[0])
        fix_point6 = stand_geometria.twoLine2Point(line2_V_abc, line2_dis_abc[1])
        self.c5 = pymunk.Circle(self.body_2, 0.5, list(fix_point5 - position))
        self.c6 = pymunk.Circle(self.body_2, 0.5, list(fix_point6 - position))

        ### line3 points ###
        line3_V_abc1 = stand_geometria.getGenenalExprVertical(self.line3_ABC, self.adjust_point_group[4])
        line3_V_abc2 = stand_geometria.getGenenalExprVertical(self.line3_ABC, self.adjust_point_group[5])
        line3_dis_abc = stand_geometria.getGenenalExprDistance(self.line3_ABC, self.distance)
        fix_point7 = stand_geometria.twoLine2Point(line3_V_abc1, line3_dis_abc[0])
        fix_point8 = stand_geometria.twoLine2Point(line3_V_abc1, line3_dis_abc[1])
        fix_point9 = stand_geometria.twoLine2Point(line3_V_abc2, line3_dis_abc[0])
        fix_point10 = stand_geometria.twoLine2Point(line3_V_abc2, line3_dis_abc[1])
        self.c7 = pymunk.Circle(self.body_2, 0.5, list(fix_point7 - position))
        self.c8 = pymunk.Circle(self.body_2, 0.5, list(fix_point8 - position))
        self.c9 = pymunk.Circle(self.body_2, 0.5, list(fix_point9 - position))
        self.c10 = pymunk.Circle(self.body_2, 0.5, list(fix_point10 - position))

        ### line4 points ###
        line4_V_abc = stand_geometria.getGenenalExprVertical(self.line4_ABC, self.adjust_point_group[1])
        line4_dis_abc = stand_geometria.getGenenalExprDistance(self.line4_ABC, self.distance)
        fix_point11 = stand_geometria.twoLine2Point(line4_V_abc, line4_dis_abc[0])
        fix_point12 = stand_geometria.twoLine2Point(line4_V_abc, line4_dis_abc[1])
        self.c11 = pymunk.Circle(self.body_2, 0.5, list(fix_point11 - position))
        self.c12 = pymunk.Circle(self.body_2, 0.5, list(fix_point12 - position))
        self.fix_point_list = [self.c1, self.c2, self.c3, self.c4,
                           self.c5, self.c6, self.c7, self.c8,
                           self.c9, self.c10, self.c11, self.c12]
        self.space.add(self.body_1)

        # self.space.add(self.line1, self.line2, self.line3, self.line4)
        for line in self.line_list:
            self.space.add(line)
        for manget_point in self.manget_point_list:
            self.space.add(manget_point)
        # self.space.add(self.manget_point1, self.manget_point2, self.manget_point3, self.manget_point4, self.manget_point5)
        self.space.add(self.body_2)
        for fix_point in self.fix_point_list:
            self.space.add(fix_point)
        # self.space.add(self.c1, self.c2, self.c3, self.c4)
        # self.space.add(self.c7, self.c8, self.c9, self.c10)
        # self.space.add(self.c5, self.c6)
        # self.space.add(self.c11, self.c12)

    def body_move(self, ACTION, movement_tuple):
        DISPLACE_INC, ROTATE_INC = movement_tuple
        if ACTION == 0:
            self.body_1.position += DISPLACE_INC, 0
        elif ACTION == 1:
            self.body_1.position -= DISPLACE_INC, 0
        elif ACTION == 2:
            self.body_1.position += 0, DISPLACE_INC
        elif ACTION == 3:
            self.body_1.position -= 0, DISPLACE_INC
        elif ACTION == 4:
            self.body_1.angle += ROTATE_INC
        else:
            self.body_1.angle -= ROTATE_INC

    # def continuous_body_move(self, movement):
    #     self.body_1.position += movement
    #     # self.body_1.angle += movement[2]

    def continuous_body_move(self, movement):
        position = self.body_1.position
        angle = self.body_1.angle
        next_position = position + movement[:2]
        next_angle = angle + movement[2]
        self.body_1.angle = next_angle
        self.body_1.position = next_position

    def body_state(self):
        return [self.body_1.position[0], self.body_1.position[1], self.body_1.angle]

    def body_state2(self):
        return [self.body_2.position[0], self.body_2.position[1], self.body_2.angle]

    def manget_state(self):
        manget_points = np.zeros((len(self.manget_point_list), 2))
        for i in range(len(self.manget_point_list)):
            point = self.body_state()[:2] + self.manget_point_list[i].offset.rotated(self.body_state()[2])
            manget_points[i] = point

        return manget_points

    def line_state(self):
        line_points = np.zeros((len(self.line_list), 2, 2))
        for i in range(len(self.line_list)):
            point_a = self.body_state()[:2] + self.line_list[i].a.rotated(self.body_state()[2])
            point_b = self.body_state()[:2] + self.line_list[i].b.rotated(self.body_state()[2])
            line_points[i, 0] = point_a
            line_points[i, 1] = point_b
        return line_points

    def fix_point_state(self):
        fix_points = np.zeros((len(self.fix_point_list), 2))
        for i in range(len(self.fix_point_list)):
            point = self.body_state2()[:2] + self.fix_point_list[i].offset.rotated(self.body_state2()[2])
            fix_points[i] = point
        return fix_points


    def state(self):
        manget_points = self.manget_state
        return manget_points.flatten()

    def action_sample(self):
        action = np.random.choice(self.ACTIONS)
        return np.array([np.where(self.ACTIONS == action)[0]])


class Stand8_1_Template():
    def __init__(self, space, adjust_point_group = None, manget_point_group = None, distance = 5):
        self.space = space
        self.adjust_point_group = adjust_point_group
        self.manget_point_group = manget_point_group
        self.target_point_group = manget_point_group
        self.distance = distance
        self.ACTIONS = np.array(range(6))
        position_1 = (self.adjust_point_group[0] + self.adjust_point_group[1]) / 2
        position_2 = (self.adjust_point_group[2] + self.adjust_point_group[3]) / 2
        position = (position_1 + position_2) / 2

        self.body_1 = pymunk.Body()
        self.body_1.position = list(position)
        # self.body_1.position.angle = 0
        line12_abc = stand_geometria.twoPoint2Line(self.adjust_point_group[0], self.adjust_point_group[1])
        line34_abc = stand_geometria.twoPoint2Line(self.adjust_point_group[2], self.adjust_point_group[3])
        line57_abc = stand_geometria.twoPoint2Line(self.adjust_point_group[4], self.adjust_point_group[6])
        line68_abc = stand_geometria.twoPoint2Line(self.adjust_point_group[5], self.adjust_point_group[7])
        cross_point1 = stand_geometria.twoLine2Point(line34_abc, line57_abc)
        cross_point2 = stand_geometria.twoLine2Point(line34_abc, line68_abc)
        cross_point3 = stand_geometria.twoLine2Point(line12_abc, line68_abc)
        cross_point4 = stand_geometria.twoLine2Point(line12_abc, line57_abc)
        cross_points = [cross_point1, cross_point2, cross_point3, cross_point4]
        adjust_points_offset = self.adjust_point_group - position
        cross_points_offset = cross_points - position

        self.line1 = pymunk.Segment(self.body_1, list(adjust_points_offset[2]), list(cross_points_offset[0]), 0)
        self.line2 = pymunk.Segment(self.body_1, list(adjust_points_offset[6]), list(cross_points_offset[0]), 0)
        self.line3 = pymunk.Segment(self.body_1, list(adjust_points_offset[7]), list(cross_points_offset[1]), 0)
        self.line4 = pymunk.Segment(self.body_1, list(adjust_points_offset[3]), list(cross_points_offset[1]), 0)
        self.line5 = pymunk.Segment(self.body_1, list(adjust_points_offset[1]), list(cross_points_offset[2]), 0)
        self.line6 = pymunk.Segment(self.body_1, list(adjust_points_offset[5]), list(cross_points_offset[2]), 0)
        self.line7 = pymunk.Segment(self.body_1, list(adjust_points_offset[4]), list(cross_points_offset[3]), 0)
        self.line8 = pymunk.Segment(self.body_1, list(adjust_points_offset[0]), list(cross_points_offset[3]), 0)

        self.line_list = [self.line1, self.line2, self.line3, self.line4,
                          self.line5, self.line6, self.line7, self.line8]
        for line in self.line_list:
            line.mass = 1

        magnet_point_offset = self.manget_point_group - position
        # self.manget_point1 = pymunk.Circle(self.body_1, 3, list(magnet_point_offset[0]))  #### refer_point1 (220, 310)
        # self.manget_point2 = pymunk.Circle(self.body_1, 3, list(magnet_point_offset[1]))
        # self.manget_point3 = pymunk.Circle(self.body_1, 3, list(magnet_point_offset[2]))
        # self.manget_point4 = pymunk.Circle(self.body_1, 3, list(magnet_point_offset[3]))  #### refer_point2 (340, 310)
        # self.manget_point5 = pymunk.Circle(self.body_1, 3, list(magnet_point_offset[4]))
        # self.manget_point_list = [self.manget_point1, self.manget_point2, self.manget_point3, self.manget_point4,
        #                           self.manget_point5]
        self.manget_point_list = []
        for offset in magnet_point_offset:
            self.manget_point_list.append(pymunk.Circle(self.body_1, 1, list(offset)))

        self.body_2 = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body_2.position = list(position)

        ### line1 points ###
        line1_V_abc = stand_geometria.getGenenalExprVertical(line34_abc, self.adjust_point_group[2])
        line1_dis_abc = stand_geometria.getGenenalExprDistance(line34_abc, self.distance)
        fix_point1 = stand_geometria.twoLine2Point(line1_V_abc, line1_dis_abc[0])
        fix_point2 = stand_geometria.twoLine2Point(line1_V_abc, line1_dis_abc[1])
        self.c1 = pymunk.Circle(self.body_2, 0.5, list(fix_point1 - position))
        self.c2 = pymunk.Circle(self.body_2, 0.5, list(fix_point2 - position))
        ### line2 points ###
        line2_V_abc = stand_geometria.getGenenalExprVertical(line57_abc, self.adjust_point_group[6])
        line2_dis_abc = stand_geometria.getGenenalExprDistance(line57_abc, self.distance)
        fix_point3 = stand_geometria.twoLine2Point(line2_V_abc, line2_dis_abc[0])
        fix_point4 = stand_geometria.twoLine2Point(line2_V_abc, line2_dis_abc[1])
        self.c3 = pymunk.Circle(self.body_2, 0.5, list(fix_point3 - position))
        self.c4 = pymunk.Circle(self.body_2, 0.5, list(fix_point4 - position))

        ### line3 points ###
        line3_V_abc = stand_geometria.getGenenalExprVertical(line68_abc, self.adjust_point_group[7])
        line3_dis_abc = stand_geometria.getGenenalExprDistance(line68_abc, self.distance)
        fix_point5 = stand_geometria.twoLine2Point(line3_V_abc, line3_dis_abc[0])
        fix_point6 = stand_geometria.twoLine2Point(line3_V_abc, line3_dis_abc[1])
        self.c5 = pymunk.Circle(self.body_2, 0.5, list(fix_point5 - position))
        self.c6 = pymunk.Circle(self.body_2, 0.5, list(fix_point6 - position))

        ### line4 points ###
        line4_V_abc = stand_geometria.getGenenalExprVertical(line34_abc, self.adjust_point_group[3])
        line4_dis_abc = stand_geometria.getGenenalExprDistance(line34_abc, self.distance)
        fix_point7 = stand_geometria.twoLine2Point(line4_V_abc, line4_dis_abc[0])
        fix_point8 = stand_geometria.twoLine2Point(line4_V_abc, line4_dis_abc[1])
        self.c7 = pymunk.Circle(self.body_2, 0.5, list(fix_point7 - position))
        self.c8 = pymunk.Circle(self.body_2, 0.5, list(fix_point8 - position))
        ### line5 points ###
        line5_V_abc = stand_geometria.getGenenalExprVertical(line12_abc, self.adjust_point_group[1])
        line5_dis_abc = stand_geometria.getGenenalExprDistance(line12_abc, self.distance)
        fix_point9 = stand_geometria.twoLine2Point(line5_V_abc, line5_dis_abc[0])
        fix_point10 = stand_geometria.twoLine2Point(line5_V_abc, line5_dis_abc[1])
        self.c9 = pymunk.Circle(self.body_2, 0.5, list(fix_point9 - position))
        self.c10 = pymunk.Circle(self.body_2, 0.5, list(fix_point10 - position))

        ### line6 points ###
        line6_V_abc = stand_geometria.getGenenalExprVertical(line68_abc, self.adjust_point_group[5])
        line6_dis_abc = stand_geometria.getGenenalExprDistance(line68_abc, self.distance)
        fix_point11 = stand_geometria.twoLine2Point(line6_V_abc, line6_dis_abc[0])
        fix_point12 = stand_geometria.twoLine2Point(line6_V_abc, line6_dis_abc[1])
        self.c11 = pymunk.Circle(self.body_2, 0.5, list(fix_point11 - position))
        self.c12 = pymunk.Circle(self.body_2, 0.5, list(fix_point12 - position))

        ### line7 points ###
        line7_V_abc = stand_geometria.getGenenalExprVertical(line57_abc, self.adjust_point_group[4])
        line7_dis_abc = stand_geometria.getGenenalExprDistance(line57_abc, self.distance)
        fix_point13 = stand_geometria.twoLine2Point(line7_V_abc, line7_dis_abc[0])
        fix_point14 = stand_geometria.twoLine2Point(line7_V_abc, line7_dis_abc[1])
        self.c13 = pymunk.Circle(self.body_2, 0.5, list(fix_point13 - position))
        self.c14 = pymunk.Circle(self.body_2, 0.5, list(fix_point14 - position))

        ### line8 points ###
        line8_V_abc = stand_geometria.getGenenalExprVertical(line12_abc, self.adjust_point_group[0])
        line8_dis_abc = stand_geometria.getGenenalExprDistance(line12_abc, self.distance)
        fix_point15 = stand_geometria.twoLine2Point(line8_V_abc, line8_dis_abc[0])
        fix_point16 = stand_geometria.twoLine2Point(line8_V_abc, line8_dis_abc[1])
        self.c15 = pymunk.Circle(self.body_2, 0.5, list(fix_point15 - position))
        self.c16 = pymunk.Circle(self.body_2, 0.5, list(fix_point16 - position))
        self.fix_point_list = [self.c1, self.c2, self.c3, self.c4,
                               self.c5, self.c6, self.c7, self.c8,
                               self.c9, self.c10, self.c11, self.c12,
                               self.c13, self.c14, self.c15, self.c16]
        ### redraw lines ###
        self.line_list[0] = self.regenerate_line(self.adjust_point_group[2], cross_point1, 20)
        self.line_list[1] = self.regenerate_line(self.adjust_point_group[6], cross_point1, 20)
        self.line_list[2] = self.regenerate_line(self.adjust_point_group[7], cross_point2, 20)
        self.line_list[3] = self.regenerate_line(self.adjust_point_group[3], cross_point2, 20)
        self.line_list[4] = self.regenerate_line(self.adjust_point_group[1], cross_point3, 20)
        self.line_list[5] = self.regenerate_line(self.adjust_point_group[5], cross_point3, 20)
        self.line_list[6] = self.regenerate_line(self.adjust_point_group[4], cross_point4, 20)
        self.line_list[7] = self.regenerate_line(self.adjust_point_group[0], cross_point4, 20)
        for line in self.line_list:
            line.mass = 1



        self.space.add(self.body_1)
        for line in self.line_list:
            self.space.add(line)
        # self.space.add(self.line1, self.line2, self.line3, self.line4)
        # self.space.add(self.line5, self.line6, self.line7, self.line8)
        for manget_point in self.manget_point_list:
            self.space.add(manget_point)

        self.space.add(self.body_2)
        for fix_point in self.fix_point_list:
            self.space.add(fix_point)
        # self.space.add(self.c1, self.c2, self.c3, self.c4)
        # self.space.add(self.c5, self.c6, self.c7, self.c8)
        # self.space.add(self.c9, self.c10, self.c11, self.c12)
        # self.space.add(self.c13, self.c14, self.c15, self.c16)

    def regenerate_line(self, manget_point, cross_point, distance):
        line_abc = stand_geometria.twoPoint2Line(manget_point, cross_point)
        line_v_abc = stand_geometria.getGenenalExprVertical(line_abc, manget_point)
        line_v_diatance_abc = stand_geometria.getGenenalExprDistance(line_v_abc, distance)
        point1 = stand_geometria.twoLine2Point(line_abc, line_v_diatance_abc[0])
        point2 = stand_geometria.twoLine2Point(line_abc, line_v_diatance_abc[1])
        vector = manget_point - cross_point
        for point in (point1, point2):
            point_vector = point - manget_point
            if sum(point_vector * vector) < 0:
                continue
            line = pymunk.Segment(self.body_1, list(point - self.body_state()[:2]), list(cross_point - self.body_state()[:2]), 0)

        return line
    def body_move(self, ACTION, movement_tuple):
        DISPLACE_INC, ROTATE_INC = movement_tuple
        if ACTION == 0:
            self.body_1.position += DISPLACE_INC, 0
        elif ACTION == 1:
            self.body_1.position -= DISPLACE_INC, 0
        elif ACTION == 2:
            self.body_1.position += 0, DISPLACE_INC
        elif ACTION == 3:
            self.body_1.position -= 0, DISPLACE_INC
        elif ACTION == 4:
            self.body_1.angle += ROTATE_INC
        else:
            self.body_1.angle -= ROTATE_INC

    # def continuous_body_move(self, movement):
    #     self.body_1.position += movement
    #     # self.body_1.angle += movement[2]

    def continuous_body_move(self, movement):
        position = self.body_1.position
        angle = self.body_1.angle
        next_position = position + movement[:2]
        next_angle = angle + movement[2]
        self.body_1.angle = next_angle
        self.body_1.position = next_position

    def body_state(self):
        return [self.body_1.position[0], self.body_1.position[1], self.body_1.angle]

    def body_state2(self):
        return [self.body_2.position[0], self.body_2.position[1], self.body_2.angle]

    def manget_state(self):
        manget_points = np.zeros((len(self.manget_point_list), 2))
        for i in range(len(self.manget_point_list)):
            point = self.body_state()[:2] + self.manget_point_list[i].offset.rotated(self.body_state()[2])
            manget_points[i] = point

        return manget_points

    def line_state(self):
        line_points = np.zeros((len(self.line_list), 2, 2))
        for i in range(len(self.line_list)):
            point_a = self.body_state()[:2] + self.line_list[i].a.rotated(self.body_state()[2])
            point_b = self.body_state()[:2] + self.line_list[i].b.rotated(self.body_state()[2])
            line_points[i, 0] = point_a
            line_points[i, 1] = point_b
        return line_points

    def fix_point_state(self):
        fix_points = np.zeros((len(self.fix_point_list), 2))
        for i in range(len(self.fix_point_list)):
            point = self.body_state2()[:2] + self.fix_point_list[i].offset.rotated(self.body_state2()[2])
            fix_points[i] = point
        return fix_points

    def state(self):
        manget_points = self.manget_state
        return manget_points.flatten()

    def action_sample(self):
        action = np.random.choice(self.ACTIONS)
        return np.array([np.where(self.ACTIONS == action)[0]])


class Stand8_2_Template():
    def __init__(self, space, adjust_point_group = None, manget_point_group = None, distance = 5):
        self.space = space
        self.adjust_point_group = adjust_point_group
        self.manget_point_group = manget_point_group
        self.target_point_group = manget_point_group
        self.distance = distance
        self.ACTIONS = np.array(range(6))
        position_1 = (self.adjust_point_group[0] + self.adjust_point_group[1]) / 2
        position_2 = (self.adjust_point_group[2] + self.adjust_point_group[3]) / 2
        position = (position_1 + position_2) / 2

        self.body_1 = pymunk.Body()
        self.body_1.position = list(position)
        line12_abc = stand_geometria.twoPoint2Line(self.adjust_point_group[0], self.adjust_point_group[1])
        line34_abc = stand_geometria.twoPoint2Line(self.adjust_point_group[2], self.adjust_point_group[3])
        line57_abc = stand_geometria.twoPoint2Line(self.adjust_point_group[4], self.adjust_point_group[6])
        line68_abc = stand_geometria.twoPoint2Line(self.adjust_point_group[5], self.adjust_point_group[7])
        cross_point1 = stand_geometria.twoLine2Point(line34_abc, line57_abc)
        cross_point2 = stand_geometria.twoLine2Point(line34_abc, line68_abc)
        cross_point3 = stand_geometria.twoLine2Point(line12_abc, line68_abc)
        cross_point4 = stand_geometria.twoLine2Point(line12_abc, line57_abc)
        cross_points = [cross_point1, cross_point2, cross_point3, cross_point4]
        adjust_points_offset = self.adjust_point_group - position
        cross_points_offset = cross_points - position

        self.line1 = pymunk.Segment(self.body_1, list(adjust_points_offset[2]), list(cross_points_offset[0]), 0)
        self.line2 = pymunk.Segment(self.body_1, list(adjust_points_offset[6]), list(cross_points_offset[0]), 0)
        self.line3 = pymunk.Segment(self.body_1, list(adjust_points_offset[7]), list(cross_points_offset[1]), 0)
        self.line4 = pymunk.Segment(self.body_1, list(adjust_points_offset[3]), list(cross_points_offset[1]), 0)
        self.line5 = pymunk.Segment(self.body_1, list(adjust_points_offset[1]), list(cross_points_offset[2]), 0)
        self.line6 = pymunk.Segment(self.body_1, list(adjust_points_offset[5]), list(cross_points_offset[2]), 0)
        self.line7 = pymunk.Segment(self.body_1, list(adjust_points_offset[4]), list(cross_points_offset[3]), 0)
        self.line8 = pymunk.Segment(self.body_1, list(adjust_points_offset[0]), list(cross_points_offset[3]), 0)

        self.line_list = [self.line1, self.line2, self.line3, self.line4,
                          self.line5, self.line6, self.line7, self.line8]
        for line in self.line_list:
            line.mass = 1

        magnet_point_offset = self.manget_point_group - position
        # self.manget_point1 = pymunk.Circle(self.body_1, 3, list(magnet_point_offset[0]))  #### refer_point1 (220, 310)
        # self.manget_point2 = pymunk.Circle(self.body_1, 3, list(magnet_point_offset[1]))
        # self.manget_point3 = pymunk.Circle(self.body_1, 3, list(magnet_point_offset[2]))
        # self.manget_point4 = pymunk.Circle(self.body_1, 3, list(magnet_point_offset[3]))  #### refer_point2 (340, 310)
        # self.manget_point5 = pymunk.Circle(self.body_1, 3, list(magnet_point_offset[4]))
        # self.manget_point_list = [self.manget_point1, self.manget_point2, self.manget_point3, self.manget_point4,
        #                           self.manget_point5]
        self.manget_point_list = []
        for offset in magnet_point_offset:
            self.manget_point_list.append(pymunk.Circle(self.body_1, 1, list(offset)))

        self.body_2 = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body_2.position = list(position)

        ### line1 points ###
        line1_V_abc = stand_geometria.getGenenalExprVertical(line34_abc, self.adjust_point_group[2])
        line1_dis_abc = stand_geometria.getGenenalExprDistance(line34_abc, self.distance)
        fix_point1 = stand_geometria.twoLine2Point(line1_V_abc, line1_dis_abc[0])
        fix_point2 = stand_geometria.twoLine2Point(line1_V_abc, line1_dis_abc[1])
        self.c1 = pymunk.Circle(self.body_2, 0.5, list(fix_point1 - position))
        self.c2 = pymunk.Circle(self.body_2, 0.5, list(fix_point2 - position))

        ### line2 points ###
        line2_V_abc = stand_geometria.getGenenalExprVertical(line57_abc, self.adjust_point_group[6])
        line2_dis_abc = stand_geometria.getGenenalExprDistance(line57_abc, self.distance)
        fix_point3 = stand_geometria.twoLine2Point(line2_V_abc, line2_dis_abc[0])
        fix_point4 = stand_geometria.twoLine2Point(line2_V_abc, line2_dis_abc[1])
        # self.c3 = pymunk.Circle(self.body_2, 2, list(fix_point3 - position))
        # self.c4 = pymunk.Circle(self.body_2, 2, list(fix_point4 - position))
        # min_x = min(self.adjust_point_group[6][0], self.adjust_point_group[7][0])
        # max_x = max(self.adjust_point_group[6][0], self.adjust_point_group[7][0])
        # min_y = min(self.adjust_point_group[6][1], self.adjust_point_group[7][1])
        # max_y = max(self.adjust_point_group[6][1], self.adjust_point_group[7][1])
        vector = self.adjust_point_group[6] - self.adjust_point_group[7]
        for fix_point in (fix_point3, fix_point4):
            fix_point_vector = fix_point - self.adjust_point_group[6]
            if sum(fix_point_vector * vector) > 0:
                self.c3 = pymunk.Circle(self.body_2, 0.5, list(fix_point - position))
            continue


        ### line3 points ###
        line3_V_abc = stand_geometria.getGenenalExprVertical(line68_abc, self.adjust_point_group[7])
        line3_dis_abc = stand_geometria.getGenenalExprDistance(line68_abc, self.distance)
        fix_point5 = stand_geometria.twoLine2Point(line3_V_abc, line3_dis_abc[0])
        fix_point6 = stand_geometria.twoLine2Point(line3_V_abc, line3_dis_abc[1])
        # self.c5 = pymunk.Circle(self.body_2, 2, list(fix_point5 - position))
        # self.c6 = pymunk.Circle(self.body_2, 2, list(fix_point6 - position))
        vector = self.adjust_point_group[7] - self.adjust_point_group[6]
        for fix_point in (fix_point5, fix_point6):
            fix_point_vector = fix_point - self.adjust_point_group[7]
            if sum(fix_point_vector * vector) > 0:
                self.c4 = pymunk.Circle(self.body_2, 0.5, list(fix_point - position))
            continue

        ### line4 points ###
        line4_V_abc = stand_geometria.getGenenalExprVertical(line34_abc, self.adjust_point_group[3])
        line4_dis_abc = stand_geometria.getGenenalExprDistance(line34_abc, self.distance)
        fix_point7 = stand_geometria.twoLine2Point(line4_V_abc, line4_dis_abc[0])
        fix_point8 = stand_geometria.twoLine2Point(line4_V_abc, line4_dis_abc[1])
        self.c5 = pymunk.Circle(self.body_2, 0.5, list(fix_point7 - position))
        self.c6 = pymunk.Circle(self.body_2, 0.5, list(fix_point8 - position))
        ### line5 points ###
        line5_V_abc = stand_geometria.getGenenalExprVertical(line12_abc, self.adjust_point_group[1])
        line5_dis_abc = stand_geometria.getGenenalExprDistance(line12_abc, self.distance)
        fix_point9 = stand_geometria.twoLine2Point(line5_V_abc, line5_dis_abc[0])
        fix_point10 = stand_geometria.twoLine2Point(line5_V_abc, line5_dis_abc[1])
        self.c7 = pymunk.Circle(self.body_2, 0.5, list(fix_point9 - position))
        self.c8 = pymunk.Circle(self.body_2, 0.5, list(fix_point10 - position))
        ### line6 points ###
        line6_V_abc = stand_geometria.getGenenalExprVertical(line68_abc, self.adjust_point_group[5])
        line6_dis_abc = stand_geometria.getGenenalExprDistance(line68_abc, self.distance)
        fix_point11 = stand_geometria.twoLine2Point(line6_V_abc, line6_dis_abc[0])
        fix_point12 = stand_geometria.twoLine2Point(line6_V_abc, line6_dis_abc[1])
        # self.c11 = pymunk.Circle(self.body_2, 2, list(fix_point11 - position))
        # self.c12 = pymunk.Circle(self.body_2, 2, list(fix_point12 - position))
        # min_x = min(self.adjust_point_group[5][0], self.adjust_point_group[4][0])
        # max_x = max(self.adjust_point_group[5][0], self.adjust_point_group[4][0])
        # min_y = min(self.adjust_point_group[5][1], self.adjust_point_group[4][1])
        # max_y = max(self.adjust_point_group[5][1], self.adjust_point_group[4][1])
        vector = self.adjust_point_group[5] - self.adjust_point_group[4]
        for fix_point in (fix_point11, fix_point12):
            fix_point_vector = fix_point - self.adjust_point_group[5]
            if sum(fix_point_vector * vector) > 0:
                self.c9 = pymunk.Circle(self.body_2, 0.5, list(fix_point - position))
            continue

        ### line7 points ###
        line7_V_abc = stand_geometria.getGenenalExprVertical(line57_abc, self.adjust_point_group[4])
        line7_dis_abc = stand_geometria.getGenenalExprDistance(line57_abc, self.distance)
        fix_point13 = stand_geometria.twoLine2Point(line7_V_abc, line7_dis_abc[0])
        fix_point14 = stand_geometria.twoLine2Point(line7_V_abc, line7_dis_abc[1])
        # self.c13 = pymunk.Circle(self.body_2, 2, list(fix_point13 - position))
        # self.c14 = pymunk.Circle(self.body_2, 2, list(fix_point14 - position))
        vector = self.adjust_point_group[4] - self.adjust_point_group[5]
        for fix_point in (fix_point13, fix_point14):
            fix_point_vector = fix_point - self.adjust_point_group[4]
            if sum(fix_point_vector * vector) > 0:
                self.c10 = pymunk.Circle(self.body_2, 0.5, list(fix_point - position))
            continue
        ### line8 points ###
        line8_V_abc = stand_geometria.getGenenalExprVertical(line12_abc, self.adjust_point_group[0])
        line8_dis_abc = stand_geometria.getGenenalExprDistance(line12_abc, self.distance)
        fix_point15 = stand_geometria.twoLine2Point(line8_V_abc, line8_dis_abc[0])
        fix_point16 = stand_geometria.twoLine2Point(line8_V_abc, line8_dis_abc[1])
        self.c11 = pymunk.Circle(self.body_2, 0.5, list(fix_point15 - position))
        self.c12 = pymunk.Circle(self.body_2, 0.5, list(fix_point16 - position))
        self.fix_point_list = [self.c1, self.c2, self.c3, self.c4,
                               self.c5, self.c6, self.c7, self.c8,
                               self.c9, self.c10, self.c11, self.c12]
        ### redraw lines ###
        self.line_list[0] = self.regenerate_line(self.adjust_point_group[2], cross_point1, 20)
        self.line_list[1] = self.regenerate_line(self.adjust_point_group[6], cross_point1, 20)
        self.line_list[2] = self.regenerate_line(self.adjust_point_group[7], cross_point2, 20)
        self.line_list[3] = self.regenerate_line(self.adjust_point_group[3], cross_point2, 20)
        self.line_list[4] = self.regenerate_line(self.adjust_point_group[1], cross_point3, 20)
        self.line_list[5] = self.regenerate_line(self.adjust_point_group[5], cross_point3, 20)
        self.line_list[6] = self.regenerate_line(self.adjust_point_group[4], cross_point4, 20)
        self.line_list[7] = self.regenerate_line(self.adjust_point_group[0], cross_point4, 20)
        for line in self.line_list:
            line.mass = 1

        self.space.add(self.body_1)
        for line in self.line_list:
            self.space.add(line)
        # self.space.add(self.line1, self.line2, self.line3, self.line4)
        # self.space.add(self.line5, self.line6, self.line7, self.line8)
        for manget_point in self.manget_point_list:
            self.space.add(manget_point)

        self.space.add(self.body_2)
        for fix_point in self.fix_point_list:
            self.space.add(fix_point)
        # self.space.add(self.c1, self.c2, self.c3, self.c4)
        # self.space.add(self.c5, self.c6, self.c7, self.c8)
        # self.space.add(self.c9, self.c10, self.c11, self.c12)
        # self.space.add(self.c13, self.c14, self.c15, self.c16)

    def body_move(self, ACTION, movement_tuple):
        DISPLACE_INC, ROTATE_INC = movement_tuple
        if ACTION == 0:
            self.body_1.position += DISPLACE_INC, 0
        elif ACTION == 1:
            self.body_1.position -= DISPLACE_INC, 0
        elif ACTION == 2:
            self.body_1.position += 0, DISPLACE_INC
        elif ACTION == 3:
            self.body_1.position -= 0, DISPLACE_INC
        elif ACTION == 4:
            self.body_1.angle += ROTATE_INC
        else:
            self.body_1.angle -= ROTATE_INC

    def single_step_move(self, ACTION):
        # position = self.body_1.position
        # angle = self.body_1.angle
        if ACTION == 0:
            self.body_1.position += 0.001, 0
        elif ACTION == 1:
            self.body_1.position -= 0.001, 0
        elif ACTION == 2:
            self.body_1.position += 0, 0.001
        elif ACTION == 3:
            self.body_1.position -= 0, 0.001
        elif ACTION == 4:
            self.body_1.angle += 0.00000002
        else:
            self.body_1.angle -= 0.00000002

    # def continuous_body_move(self, movement):
    #     self.body_1.position += movement
    #     # self.body_1.angle += movement[2]

    def continuous_body_move(self, movement):
        position = self.body_1.position
        angle = self.body_1.angle
        next_position = position + movement[:2]
        next_angle = angle + movement[2]
        self.body_1.angle = next_angle
        self.body_1.position = next_position

    def regenerate_line(self, manget_point, cross_point, distance):
        line_abc = stand_geometria.twoPoint2Line(manget_point, cross_point)
        line_v_abc = stand_geometria.getGenenalExprVertical(line_abc, manget_point)
        line_v_diatance_abc = stand_geometria.getGenenalExprDistance(line_v_abc, distance)
        point1 = stand_geometria.twoLine2Point(line_abc, line_v_diatance_abc[0])
        point2 = stand_geometria.twoLine2Point(line_abc, line_v_diatance_abc[1])
        vector = manget_point - cross_point
        for point in (point1, point2):
            point_vector = point - manget_point
            if sum(point_vector * vector) < 0:
                continue
            line = pymunk.Segment(self.body_1, list(point - self.body_state()[:2]), list(cross_point - self.body_state()[:2]), 0)

        return line

    def body_state(self):
        return [self.body_1.position[0], self.body_1.position[1], self.body_1.angle]

    def body_state2(self):
        return [self.body_2.position[0], self.body_2.position[1], self.body_2.angle]

    def manget_state(self):
        manget_points = np.zeros((len(self.manget_point_list), 2))
        for i in range(len(self.manget_point_list)):
            point = self.body_state()[:2] + self.manget_point_list[i].offset.rotated(self.body_state()[2])
            manget_points[i] = point

        return manget_points

    def line_state(self):
        line_points = np.zeros((len(self.line_list), 2, 2))
        for i in range(len(self.line_list)):
            point_a = self.body_state()[:2] + self.line_list[i].a.rotated(self.body_state()[2])
            point_b = self.body_state()[:2] + self.line_list[i].b.rotated(self.body_state()[2])
            line_points[i, 0] = point_a
            line_points[i, 1] = point_b
        return line_points

    def fix_point_state(self):
        fix_points = np.zeros((len(self.fix_point_list), 2))
        for i in range(len(self.fix_point_list)):
            point = self.body_state2()[:2] + self.fix_point_list[i].offset.rotated(self.body_state2()[2])
            fix_points[i] = point
        return fix_points

    def state(self):
        manget_points = self.manget_state
        return manget_points.flatten()

    def action_sample(self):
        action = np.random.choice(self.ACTIONS)
        return np.array([np.where(self.ACTIONS == action)[0]])


class newStand6_Template(Stand6_Template):
    def __init__(self,space, adjust_point_group = None, manget_point_group = None, target_points = None, distance = 10):
        super().__init__(space, adjust_point_group, manget_point_group, distance)
        self.target_points = target_points

        target_points_offset = self.target_points - self.body_1.position
        # self.manget_point1 = pymunk.Circle(self.body_1, 3, list(magnet_point_offset[0]))  #### refer_point1 (220, 310)
        # self.manget_point2 = pymunk.Circle(self.body_1, 3, list(magnet_point_offset[1]))
        # self.manget_point3 = pymunk.Circle(self.body_1, 3, list(magnet_point_offset[2]))
        # self.manget_point4 = pymunk.Circle(self.body_1, 3, list(magnet_point_offset[3]))  #### refer_point2 (340, 310)
        # self.manget_point5 = pymunk.Circle(self.body_1, 3, list(magnet_point_offset[4]))
        # self.manget_point_list = [self.manget_point1, self.manget_point2, self.manget_point3, self.manget_point4,
        #                           self.manget_point5]
        self.target_point_list = []
        for offset in target_points_offset:
            self.target_point_list.append(pymunk.Circle(self.body_1, 1, list(offset)))
        for point in self.target_point_list:
            self.space.add(point)

    def target_state(self):
        target_points = np.zeros((len(self.target_points), 2))
        for i in range(len(self.target_points)):
            point = self.body_state()[:2] + self.target_point_list[i].offset.rotated(self.body_state()[2])
            target_points[i] = point

        return target_points

class newStand8_1_Template(Stand8_1_Template):
    def __init__(self,space, adjust_point_group = None, manget_point_group = None, target_points = None, distance = 10):
        super().__init__(space, adjust_point_group, manget_point_group, distance)
        self.target_points = target_points

        target_points_offset = self.target_points - self.body_1.position
        # self.manget_point1 = pymunk.Circle(self.body_1, 3, list(magnet_point_offset[0]))  #### refer_point1 (220, 310)
        # self.manget_point2 = pymunk.Circle(self.body_1, 3, list(magnet_point_offset[1]))
        # self.manget_point3 = pymunk.Circle(self.body_1, 3, list(magnet_point_offset[2]))
        # self.manget_point4 = pymunk.Circle(self.body_1, 3, list(magnet_point_offset[3]))  #### refer_point2 (340, 310)
        # self.manget_point5 = pymunk.Circle(self.body_1, 3, list(magnet_point_offset[4]))
        # self.manget_point_list = [self.manget_point1, self.manget_point2, self.manget_point3, self.manget_point4,
        #                           self.manget_point5]
        self.target_point_list = []
        for offset in target_points_offset:
            self.target_point_list.append(pymunk.Circle(self.body_1, 1, list(offset)))
        for point in self.target_point_list:
            self.space.add(point)

    def target_state(self):
        target_points = np.zeros((len(self.target_points), 2))
        for i in range(len(self.target_points)):
            point = self.body_state()[:2] + self.target_point_list[i].offset.rotated(self.body_state()[2])
            target_points[i] = point

        return target_points

class newStand8_2_Template(Stand8_2_Template):
        def __init__(self, space, adjust_point_group=None, manget_point_group=None, target_points=None, distance=10):
            super().__init__(space, adjust_point_group, manget_point_group, distance)
            self.target_points = target_points

            target_points_offset = self.target_points - self.body_1.position
            # self.manget_point1 = pymunk.Circle(self.body_1, 3, list(magnet_point_offset[0]))  #### refer_point1 (220, 310)
            # self.manget_point2 = pymunk.Circle(self.body_1, 3, list(magnet_point_offset[1]))
            # self.manget_point3 = pymunk.Circle(self.body_1, 3, list(magnet_point_offset[2]))
            # self.manget_point4 = pymunk.Circle(self.body_1, 3, list(magnet_point_offset[3]))  #### refer_point2 (340, 310)
            # self.manget_point5 = pymunk.Circle(self.body_1, 3, list(magnet_point_offset[4]))
            # self.manget_point_list = [self.manget_point1, self.manget_point2, self.manget_point3, self.manget_point4,
            #                           self.manget_point5]
            self.target_point_list = []
            for offset in target_points_offset:
                self.target_point_list.append(pymunk.Circle(self.body_1, 1, list(offset)))
            for point in self.target_point_list:
                self.space.add(point)

        def target_state(self):
            target_points = np.zeros((len(self.target_points), 2))
            for i in range(len(self.target_points)):
                point = self.body_state()[:2] + self.target_point_list[i].offset.rotated(self.body_state()[2])
                target_points[i] = point

            return target_points








if __name__ == "__main__":
    # adjust_points1 = np.array([
    #     [3720.00, 28329.79],
    #     [3720.00, 27794.61],
    #     [3346.00, 28292.20],
    #     [3346.00, 27837.20],
    #     [4094.00, 28232.20],
    #     [4094.00, 27872.20]
    # ])
    #
    # manget_points1 = np.array([
    #     [3500.00, 28062.20],
    #     [3975.00, 28062.20]
    # ])
    #
    # adjust_points2 = np.array([
    #     [4225.95, 27620.16],
    #     [5227.83, 27541.04],
    #     [4283.16, 28408.10],
    #     [5302.99, 28327.56],
    #     [4390.51, 27531.93],
    #     [5026.53, 27481.70],
    #     [4464.52, 28469.01],
    #     [5100.54, 28418.78]
    # ])
    #
    # manget_points2 = np.array([
    #     [4269.994, 28060.987],
    #     [4509.816, 28051.664],
    #     [4749.273, 28035.500],
    #     [4988.326, 28014.196],
    #     [5226.975, 27988.769]
    # ])
    #
    # adjust_points3 = np.array([
    #     [5555.30, 27635.55],
    #     [7274.67, 27426.79],
    #     [5627.25, 28228.20],
    #     [7346.63, 28019.44],
    #     [6379.34, 27420.17],
    #     [6419.05, 27415.35],
    #     [6483.41, 28235.80],
    #     [6523.12, 28230.98]
    # ])
    # # 5.519877 - 27.953640
    # # 5.792873 - 27.920494
    # # 6.289227 - 27.860229
    # # 6.587085 - 27.824446
    # # 7.084344 - 27.772153
    # # 7.357858 - 27.743598
    #
    # manget_points3 = np.array([
    #     [5519.877, 27953.640],
    #     [5792.873, 27920.494],
    #     [6289.227, 27860.229],
    #     [6587.085, 27824.446],
    #     [7084.344, 27772.153],
    #     [7357.858, 27743.598]
    # ])


    # import pygame
    # import pymunk.pygame_util
    # pygame.init()
    # screen = pygame.display.set_mode((300, 300))
    # pygame.display.set_caption("stand Q-learning simulation")
    # clock = pygame.time.Clock()

    space = pymunk.Space()  # 2
    # space.gravity = (-1, 0.5)
    # draw_options = pymunk.pygame_util.DrawOptions(screen)

    # stand1 = Stand6(space, position=(150, 300))
    # stand2 = Stand6_Template(space, adjust_point_group=adjust_points, manget_point_group=manget_points)
    stand1 = newStand6_Template(space, adjust_point_group=new_adjust_points1, manget_point_group=new_magnet_points1,  target_points=new_target_points1, distance=10)
    stand2 = newStand8_1_Template(space, adjust_point_group=new_adjust_points2, manget_point_group=new_magnet_points2, target_points=new_target_points2,distance=10)
    stand3 = newStand8_2_Template(space, adjust_point_group=new_adjust_points3, manget_point_group=new_magnet_points3, target_points=new_target_points3,distance=10)
    stand4 = newStand8_1_Template(space, adjust_point_group=new_adjust_points4, manget_point_group=new_magnet_points4, target_points=new_target_points4,distance=10)
    stand5 = newStand8_2_Template(space, adjust_point_group=new_adjust_points5, manget_point_group=new_magnet_points5, target_points=new_target_points5,distance=10)
    # stand6 = Stand8_1_Template(space, adjust_point_group=adjust_points6, manget_point_group=manget_points6, distance=10)
    stand1_init_state = stand1.body_state()
    stand2_init_state = stand2.body_state()
    stand3_init_state = stand3.body_state()
    stand4_init_state = stand4.body_state()
    stand5_init_state = stand5.body_state()
    # stand6_init_state = stand6.body_state

    print("Stand1")

    print(stand1_init_state)
    print(stand1.manget_state())
    print(stand1.target_state())
    print(stand1.fix_point_state())

    print("#########################################")
    print("Stand2")
    print(stand2_init_state)
    print(stand2.manget_state())
    print(stand2.target_state())
    print(stand2.fix_point_state())

    print("#########################################")
    print("Stand3")
    print(stand3_init_state)
    print(stand3.manget_state())
    print(stand3.target_state())
    print(stand3.fix_point_state())


    print("#########################################")

    print("Stand4")

    print(stand4.body_state())
    print(stand4.manget_state())
    print(stand4.target_state())
    print(stand4.fix_point_state())
    print("#########################################")

    print("Stand5")
    print(stand5.body_state())
    print(stand5.manget_state())
    print(stand5.target_state())
    print(stand5.fix_point_state())
    print("#########################################")

    # print("Stand6")
    # print(stand6_init_state)
    # # print(stand3.manget_state)
    # # print(stand3.adjust_point_group)
    # # print(stand3.line_state)
    # # print(stand3.fix_point_state)
    # stand6.continuous_body_move([0.1, 0.1, 0.0002])
    # # stand3.continuous_body_move([0, 0, 0.005])
    # space.step(.1)
    # print(stand6.body_state)
    # print("#########################################")
