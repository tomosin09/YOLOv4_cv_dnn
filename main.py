from model import YOLOv4
from utils import GetUtil
import argparse
import configparser
import cv2


def main(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    writer = config['parameters'].getboolean('writer')
    writer_screen = config['parameters'].getboolean('writer_screen')

    if writer_screen:
        print('Mode of saving screenshots is selected')
        num_screen = int(config['parameters']['number_screen'])
        screen_path = config['paths']['path_to_screen']
        start_num = int(config['parameters']['start_number'])
        num = num_screen+start_num

    show = config['parameters'].getboolean('show')
    classes_file = GetUtil.getNames(config['paths']["path_to_classes"])

    stream = GetUtil.getStream(config['paths']['path_to_source'])
    if writer:
        print('Video recording mode selected')
        # output_path = config['paths']['path_to_out']
        # fps = int(config['parameters']['fps_writer'])
        width, height = stream.getSize()
        # codec = cv2.VideoWriter_fourcc(*'XVID')
        writing = cv2.VideoWriter(config['paths']['path_to_out'],
                                  cv2.VideoWriter_fourcc(*'XVID'),
                                  int(config['parameters']['fps_writer']), (width, height))

    yolo_model = YOLOv4(config['paths']['path_to_weights'], config['paths']['path_to_cfg'], classes_file)

    model = yolo_model.getModel(input_size=416)

    print('Stream is starting\nIf you want to turn off stream - press escape')
    if show: cv2.namedWindow("Stream", cv2.WINDOW_NORMAL)
    count = 0

    while 1:
        frame = stream.read()
        if frame is None:
            print('Frame is empty')
            break
        if cv2.waitKey(10) != cv2.waitKey(27): break
        if writer_screen:
            out, boxes = yolo_model.getFramePred(frame, model,
                                                 conf=float(config['parameters']['confidence']),
                                                 nms=float(config['parameters']['nms_threshold']),
                                                 drawing=(config['parameters'].getboolean('drawing')),
                                                 checkColor=config['parameters'].getboolean('checkColor'),
                                                 writer_screen=True)
            for box in boxes:
                y1 = box[1]
                y2 = box[1] + box[3]
                x1 = box[0]
                x2 = box[0] + box[2]
                cv2.imwrite(f'{screen_path}/{start_num:03}.jpg', frame[y1:y2, x1:x2])
                start_num += 1
            if start_num >= num:
                print(f"{num_screen} screenshots were saved in {screen_path}")
                writer_screen = False
        else:
            out = yolo_model.getFramePred(frame, model,
                                          conf=float(config['parameters']['confidence']),
                                          nms=float(config['parameters']['nms_threshold']),
                                          drawing=(config['parameters'].getboolean('drawing')),
                                          checkColor=config['parameters'].getboolean('checkColor'),
                                          writer_screen=False)

        if writer: writing.write(out)
        if show: cv2.imshow('Stream', out)

    if show: cv2.destroyAllWindows()
    stream.stop()
    print('Stream is over')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv4 inference to check the quality of painting parts')
    parser.add_argument('path_to_cfg',
                        type=str,
                        help='enter config path')
    args = parser.parse_args()
    main(args.path_to_cfg)
