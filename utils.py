from Stream import VideoStream
import numpy as np
from collections import namedtuple
from math import sqrt
import random
from PIL import Image
import cv2
import time

from PIL.ImageColor import getcolor


class GetUtil(object):

    @staticmethod
    def drawPred(frame, classes, scores, boxes, class_names, COLOR_LIST):
        for (classid, score, box) in zip(classes, scores, boxes):
            color = COLOR_LIST[int(classid)].tolist()
            label = "%s : %f" % (class_names[classid[0]], score)
            cv2.rectangle(frame, box, color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    @staticmethod
    def getColorList(class_names):
        return list([np.random.choice(range(256), size=3) for x in class_names])

    @staticmethod
    def getNames(classes_file):
        with open(classes_file, "r") as f:
            class_names = [cname.strip() for cname in f.readlines()]
        return class_names

    @staticmethod
    def getStream(src=0):
        stream = VideoStream(src).start()
        if stream.grabbed == 0:
            print('No connect')
        else:
            print('Connection...')
        time.sleep(1)
        return stream


class ControlColor:

    def __init__(self):
        super().__init__()
        self.Pr, self.Pg, self.Pb = .299, .587, .114

        self.Point = namedtuple('Point', ('coords', 'n', 'ct'))
        self.Cluster = namedtuple('Cluster', ('points', 'center', 'n'))

        self.detail = ["Detail 1", "Detail 2", "Detail 3"]

    def get_points(self, img):
        points = []
        w, h = img.size
        for count, color in img.getcolors(w * h):
            points.append(self.Point(color, 3, count))
        return points

    def colorize(self, frame, n=16):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        im_pil.thumbnail((200, 200))
        w, h = im_pil.size
        points = self.get_points(im_pil)
        clusters = self.kmeans(points, n, 1)
        rgbs = [list(map(int, c.center.coords)) for c in clusters]
        rtoh = lambda rgb: '#%s' % ''.join(('%02x' % p for p in rgb))
        rgb_to_hsp = lambda rgb: sqrt(
            sum([rgb[0] ** 2 * self.Pr, rgb[1] ** 2 * self.Pg, rgb[2] ** 2 * self.Pb]))
        return list(map(rtoh, sorted(rgbs, key=rgb_to_hsp)))

    def euclidean(self, p1, p2):
        return sqrt(sum([
            (p1.coords[i] - p2.coords[i]) ** 2 for i in range(p1.n)
        ]))

    def calculate_center(self, points, n):
        vals = [0.0 for i in range(n)]
        plen = 0
        for p in points:
            plen += p.ct
            for i in range(n):
                vals[i] += (p.coords[i] * p.ct)
        return self.Point([(v / plen) for v in vals], n, 1)

    def kmeans(self, points, k, min_diff):
        clusters = [self.Cluster([p], p, p.n) for p in random.sample(points, k)]
        while True:
            plists = [[] for i in range(k)]
            for p in points:
                smallest_distance = float('Inf')
                for i in range(k):
                    distance = self.euclidean(p, clusters[i].center)
                    if distance < smallest_distance:
                        smallest_distance = distance
                        idx = i
                plists[idx].append(p)
            diff = 0
            for i in range(k):
                old = clusters[i]
                center = self.calculate_center(plists[i], old.n)
                new = self.Cluster(plists[i], center, old.n)
                clusters[i] = new
                diff = max(diff, self.euclidean(old.center, new.center))

            if diff < min_diff:
                break
        return clusters

    # def start(self, frame):
    #     colorsHex = self.colorize(frame, 2)
    #     rgbB = getcolor(colorsHex[0], "RGB")[::-1]
    #     rgbW = getcolor(colorsHex[1], "RGB")[::-1]
    #     if rgbW < (110, 140, 120):
    #         if rgbB < (80, 80, 80):
    #             return 1  # норм
    #         else:
    #             return 2  # ниочем
    #     else:
    #         return 3  # ниочем с засветом

    def start(self, frame):
        colorsHex = self.colorize(frame, 2)
        rgbB = getcolor(colorsHex[0], "RGB")[::-1]
        rgbW = getcolor(colorsHex[1], "RGB")[::-1]
        if rgbW > (130, 130, 130):
            if rgbB < (95, 85, 85):
                return 1  # Норм
            else:
                return 2  # Хуита
        else:
            if rgbB < (95, 85, 85):
                return 3  # Норм но с хуевым освещением
            else:
                return 3  # Хуита с хуевым освещением
