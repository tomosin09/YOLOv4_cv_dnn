import cv2
import time
from utils import GetUtil
from utils import ControlColor


class YOLOv4(object):

    def __init__(self, path_to_weights, path_to_cfg, class_names):
        self.path_to_weights = path_to_weights
        self.path_to_cfg = path_to_cfg
        self.class_names = class_names
        self.color_list = GetUtil.getColorList(self.class_names)

    def getModel(self, input_size):
        """
        :param input_size: network input frame resolution, the more - the more accurate, and vice versa
        in order to use cuda, you can use the following link:
        https://medium.com/analytics-vidhya/build-opencv-from-source-with-cuda-for-gpu-access-on-windows-5cd0ce2b9b37
        if you need to make any changes to the network parameters, change the parameters in the function .setInputParams
        https://docs.opencv.org/master/d3/df0/classcv_1_1dnn_1_1Model.html#a3af51e4292f8cbce196df311254826c2
        :return: prediction model
        """
        net = cv2.dnn.readNet(self.path_to_weights, self.path_to_cfg)
        if cv2.cuda.getCudaEnabledDeviceCount() == 1:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print('Target device is CUDA')
        else:
            print('Target device is CPU,')
        model = cv2.dnn_DetectionModel(net)
        model.setInputParams(size=(input_size, input_size), scale=1 / 255, swapRB=True)
        return model

    def putFpsLabel(self, frame, time_inf, time_draw):
        fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (
            1 / time_inf, time_draw * 1000)
        cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    def getFramePred(self, frame, model, conf=0.9, nms=0.1, drawing=True, checkColor=False, writer_screen=False):

        if drawing:
            if checkColor:
                colorRun = ControlColor()

        start = time.time()
        classes, scores, boxes = model.detect(frame, conf, nms)
        end = time.time()
        if drawing and not checkColor:

            start_drawing = time.time()
            GetUtil.drawPred(frame, classes, scores, boxes, self.class_names, self.color_list)
            end_drawing = time.time()
            time_inf = end - start
            time_draw = end_drawing - start_drawing
            self.putFpsLabel(frame, time_inf, time_draw)

        if checkColor:

            start_drawing = time.time()
            for (classid, score, box) in zip(classes, scores, boxes):
                ret = colorRun.start(frame[box[1]: box[1] + box[3], box[0]: box[0] + box[2]])
                if ret == 1:
                    label = "%s : %f" % (self.class_names[classid[0]], score)
                    cv2.rectangle(frame, box, (0, 255, 0), 2)
                if ret == 2:
                    label = "%s : %f" % (self.class_names[classid[0]], score)
                    cv2.rectangle(frame, box, (0, 0, 255), 2)
                if ret == 3:
                    label = "%s : %f" % (self.class_names[classid[0]], score)
                    cv2.rectangle(frame, box, (255, 255, 255), 2)
                cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            end_drawing = time.time()
            time_inf = end - start
            time_draw = end_drawing - start_drawing
            self.putFpsLabel(frame, time_inf, time_draw)

        if writer_screen:
            return frame, boxes
        else:
            return frame


