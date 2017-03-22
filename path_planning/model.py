import numpy as np
import time
import cv2
from robot import *
from geo import *
from drawing import *
from dataprovider import *


class PidModel:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

        self.orientation = 0.0
        self.taup = 0.001
        self.taud = 0.01
        self.taui = 0.0
        self.int_cte = 0.0
        self.sequence = -1

        self.steering_angle = 0
        self.target_speed = 10.0

        self.prev_point = [0,0]
        self.cte = 0.0

        filename = 'data/000/driving_log.csv'
        self.position_data = self.load_simulator_data(filename)

        self.keypoints = []
        self.pathimg = np.ones((1500, 1800, 3), np.uint8)

        for idx, p in enumerate(self.position_data[280:]):
            pos = (int(float(p[1][4])), int(float(p[1][5])))
            cv2.circle(self.pathimg, pos, 1, (0, 0, 255))
            self.keypoints.append(pos)

    def load_simulator_data(self, csvfname):
        """
        Load dataset from csv file
        """
        data = []
        with open(csvfname, 'r') as csvfile:
            data_tmp = list(csv.reader(csvfile, delimiter=','))
            for row in data_tmp:
                x7 = [float(x) for x in row[7].split(':')]
                x8 = [float(x) for x in row[8].split(':')]

                data.append(((row[0], row[1], row[2]),
                             np.array([float(row[3]), float(row[4]), float(row[5]), float(row[6])] + x7 + x8)))

        return data

    def set(self, x, y, orientation):
        self.x = float(x)
        self.y = float(y)
        self.orientation = float(orientation)

    def steer_turn_by(self, p):
        l1, l2 = self.prev_point, (self.x, self.y)

        return ((l2[0] - l1[0]) * (p[1] - l1[1]) - (l2[1] - l1[1]) * (p[0] - l1[0]))

    def get_cte(self):
        cp = (self.x, self.y)
        mind = 10000.
        min_idx = 0
        for idx, kp in enumerate(self.keypoints):
            d = line_dist(kp, cp)
            if mind > d:
                min_idx = idx
                mind = d

        return mind, min_idx

    def adjust_throttle(self, speed):
        if speed < self.target_speed:
            self.throttle += 0.025
        elif speed > self.target_speed:
            self.throttle -= 0.025

    def start_sequence(self, sequence):
        if sequence > 50:
            self.sequence = 0
            self.int_cte = 0.0
        else:
            self.sequence = sequence

    def _update(self):
        cte, cte_idx = self.get_cte()
        direction = self.steer_turn_by(self.keypoints[cte_idx])
        dcte = cte - self.cte
        steer = -cte * self.taup - dcte * self.taud - self.int_cte * self.taui

        if direction < 0.0:
            steer *= -1

        """
        cv2.circle(img, (int(px), int(py)), 2, (255, 255, 255))
        cv2.circle(img, keypoints[tpidx], 2, (255, 255, 0))
        cv2.line(img, cp, keypoints[tpidx], (255, 0, 0))
        cv2.imwrite("image.png", img)
        """
        self.int_cte += cte
        self.steering_angle = steer
        self.cte = cte

    def predict(self, sequence, steering_angle, throttle, speed, imgString, position, rotation):
        self.steering_angle = steering_angle
        self.throttle = throttle
        self.speed = speed

        if self.target_speed < speed:
            self.throttle -= 0.01
        elif self.target_speed > speed:
            self.throttle += 0.01

        self.x, self.y, self.z = [float(x) for x in position.split(":")]
        rx, ry, rz = [float(x) for x in rotation.split(":")]
        self.orientation = deg2rad(ry - 175.0 - 84.924)

        self.start_sequence(sequence)
        self._update()

        self.update_prev_point()

        return self.steering_angle, self.throttle

    def update_prev_point(self):
        if self.sequence % 3 == 0:
            self.prev_point = (self.x, self.y)