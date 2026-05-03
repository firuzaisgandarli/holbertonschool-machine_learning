#!/usr/bin/env python3
'''initialize'''
import tensorflow
import keras
import numpy as np


class Yolo:
    '''class'''
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        '''instance attr'''
        self.model = tensorflow.keras.models.load_model(model_path)
        self.class_names = [line.rstrip('\n') for line in
                            open(classes_path, "r")]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
