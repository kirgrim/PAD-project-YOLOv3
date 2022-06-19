import logging
import os
import urllib.request
import requests
import subprocess
import sys

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class YOLOEvaluator:

    # COCO dataset default trained sets
    __config_url__ = 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg'
    __coco_weights_url__ = 'https://pjreddie.com/media/files/yolov3.weights'
    __coco_classes_url__ = 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'

    def __init__(self, version: str = '3'):
        self.version = version

    @staticmethod
    def __extract_from_url(url) -> str:
        res = requests.get(url)
        if res.ok:
            return res.text

    @staticmethod
    def __download_from_url(url, dest: str = None):
        logger.info(f'Downloading data from url: {url} to {dest}')
        if not dest:
            dest = url.split('/')[-1]
        urllib.request.urlretrieve(url, dest)

    @staticmethod
    def prepare_environment():

        # TODO: run this to prepare darknet environment locally,
        # it is also possible to use Docker image: https://hub.docker.com/r/daisukekobayashi/darknet

        # clonning AlexeyAB darknet
        os.system('git clone https://github.com/AlexeyAB/darknet')

        p = subprocess.Popen(["powershell.exe",
                              'runas',
                              '/noprofile',
                              '/user:Administrator',
                              'NeedsAdminPrivilege.exe',
                              "cd darknet",
                              "./build.ps1 -UseVCPKG -EnableOPENCV -EnableCUDA -EnableCUDNN"],
                             stdout=sys.stdout)
        p.communicate()

    @classmethod
    def run_training(cls, data_file, config_file):
        """ Use this method in case you want to start the new training """
        #
        # e.g. data_file=data/obj.data, config_file=yolov3.cfg, network_config=yolov3.conv.137
        network_config = 'darknet53.conv.74'
        if not os.path.exists(network_config):
            cls.__download_from_url(url='https://pjreddie.com/media/files/darknet53.conv.74')
        # .weights will be stored in /darknet/backup by default
        command = f"cd darknet && darknet.exe detector train ../{data_file} ../{config_file} ../darknet53.conv.74 -gpus 0,1,2,3"
        try:
            output = subprocess.check_output(command, shell=True)
            logger.info(output)
        except Exception as ex:
            logger.error(ex)

    @staticmethod
    def run_training_from_checkpoint(data_file, config_file, weights_file):
        """ Use this method in case you want to improve the model from the previous training (.weights file) """
        # e.g. data_file=data/obj.data, config_file=yolo-obj.cfg, weights_file=yolov3.weights.1000
        p = subprocess.Popen([f"./darknet/darknet.exe detector train {data_file} {config_file} {weights_file}"],
                             stdout=sys.stdout)
        p.communicate()

    def run_detection(self, img_path,
                      weights_path: str = "yolov3.weights",
                      config_path='yolov3.cfg',
                      detection_classes_path: str = 'coco.names',
                      run_coco: bool = True):
        """
            Runs detection over the provided image based on provided configs

            :param img_path: path to the image to run detection on
            :param weights_path: path to the YOLOv3 model weights
            :param config_path: path to the YOLOv3 configuration
            :param detection_classes_path: path to the list of classes to consider
            :param run_coco: to run detection over COCO dataset (default option)
        """
        if run_coco:
            if not os.path.exists(config_path):
                self.__download_from_url(self.__config_url__)
            if not os.path.exists(weights_path):
                self.__download_from_url(self.__coco_weights_url__)
            if not os.path.exists(detection_classes_path):
                self.__download_from_url(self.__coco_classes_url__)

        with open(detection_classes_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        def load_yolo():
            _net = cv2.dnn.readNet(weights_path, config_path)
            layers_names = _net.getLayerNames()
            _output_layers = [layers_names[i[0] - 1] for i in _net.getUnconnectedOutLayers()]
            _colors = np.random.uniform(0, 255, size=(len(classes), 3))
            return _net, _colors, _output_layers

        net, colors, output_layers = load_yolo()

        def load_image():
            # image loading
            _image = cv2.imread(img_path)
            _image = cv2.resize(_image, None, fx=0.4, fy=0.4)
            _height, _width, _channels = _image.shape
            return _image, _height, _width, _channels

        image, height, width, channels = load_image()

        def detect_objects():
            _blob = cv2.dnn.blobFromImage(image, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
            net.setInput(_blob)
            _outputs = net.forward(output_layers)
            return _blob, _outputs

        blob, outputs = detect_objects()

        def get_box_dimensions():
            _boxes = []
            _confs = []
            _class_ids = []
            for output in outputs:
                for detect in output:
                    scores = detect[5:]
                    class_id = np.argmax(scores)
                    conf = scores[class_id]
                    if conf > 0.3:
                        center_x = int(detect[0] * width)
                        center_y = int(detect[1] * height)
                        w = int(detect[2] * width)
                        h = int(detect[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        _boxes.append([x, y, w, h])
                        _confs.append(float(conf))
                        _class_ids.append(class_id)
            return _boxes, _confs, _class_ids

        boxes, confs, class_ids = get_box_dimensions()

        def draw_labels():
            indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
            font = cv2.FONT_HERSHEY_PLAIN
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = colors[i]
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(image, label, (x, y - 5), font, 1, color, 1)

        draw_labels()

        def show_image():
            cv2.imshow('Image with detected objects', image)
            key = cv2.waitKey(0)
            if key == 27:
                cv2.destroyAllWindows()

        show_image()


if __name__ == '__main__':
    evaluator = YOLOEvaluator(version='3')
    evaluator.run_detection(img_path='img.png')
    # evaluator.run_training(data_file='coco.data', config_file='yolov3.cfg')
