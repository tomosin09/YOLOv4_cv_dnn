import cv2 as cv
from threading import Thread


# Class was taken from imutils by Adrian Rosebrock
# https://github.com/jrosebr1/imutils/

class VideoStream:
    def __init__(self, src=0, name='VideoStream'):
        # Video stream initialization
        self.stream = cv.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()

        self.name = name

        self.stopped = False

    def start(self):
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while 1:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stream.release()
        self.stopped = True

    def getSize(self):
        width = int(self.stream.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(self.stream.get(cv.CAP_PROP_FRAME_HEIGHT))
        return width, height
