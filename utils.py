import logging
from collections import defaultdict, Counter
import re
import numpy as np
from typing import Tuple

import tensorflow.compat.v1 as tf
from object_detection.inputs import train_input
from object_detection.protos import input_reader_pb2
from object_detection.builders.dataset_builder import build as build_dataset
from object_detection.utils.config_util import get_configs_from_pipeline_file
from waymo_open_dataset import dataset_pb2 as open_dataset

def get_object_distribution(batch:tf.data.Dataset) -> Tuple[Counter]:
    """Determines labeled object counts within dataset.

    Returns:
        Labels and counts.
    """
    object_density_dict = defaultdict(dict)
    object_count = Counter()
    previous_object_count= Counter()
    source_id_object_count = Counter()
    source_id_frame_count = Counter()
    filename_pattern = '(\S+)_(\w+).tfrecord'
    for sample in batch:
        source_id = sample['source_id'].numpy().decode('utf-8')
        pattern_match = re.search(filename_pattern, source_id)
        parent_filename = pattern_match.group(1)+'.tfrecord'
        source_id_frame_count[parent_filename] += 1
        for idx in range(len(sample['groundtruth_classes'])):
            object_count[
                sample['groundtruth_classes'][idx].numpy()
            ] += 1
        for key in object_count.keys():
            if key not in object_density_dict.keys():
                object_density_dict[key] = Counter()
            if key in previous_object_count.keys():
                delta_count = object_count[key] - previous_object_count[key]
                if delta_count > 0:
                    object_density_dict[key][delta_count] += 1
                    source_id_object_count[parent_filename] += delta_count
            else:
                object_density_dict[key][object_count[key]] += 1
                source_id_object_count[parent_filename] += object_count[key]
        previous_object_count = object_count.copy()
    retval = (
        object_density_dict, 
        object_count, 
        source_id_object_count,
        source_id_frame_count,
    )
    return retval

def get_dataset(tfrecord_path, label_map='label_map.pbtxt'):
    """
    Opens a tf record file and create tf dataset
    args:
      - tfrecord_path [str]: path to a tf record file
      - label_map [str]: path the label_map file
    returns:
      - dataset [tf.Dataset]: tensorflow dataset
    """
    input_config = input_reader_pb2.InputReader()
    input_config.label_map_path = label_map
    input_config.tf_record_input_reader.input_path[:] = [tfrecord_path]
    
    dataset = build_dataset(input_config)
    return dataset


def get_module_logger(mod_name):
    """ simple logger """
    logger = logging.getLogger(mod_name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


def get_train_input(config_path):
  """
  Get the tf dataset that inputs training batches
  args:
    - config_path [str]: path to the edited config file
  returns:
    - dataset [tf.Dataset]: data outputting augmented batches
  """
  # parse config
  configs = get_configs_from_pipeline_file(config_path)
  train_config = configs['train_config']
  train_input_config = configs['train_input_config']

  # get the dataset
  dataset = train_input(train_config, train_input_config, configs['model'])
  return dataset

def parse_frame(frame, camera_name='FRONT'):
    """ 
    take a frame, output the bboxes and the image

    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
      for data in dataset:
      frame = open_dataset.Frame()
      frame.ParseFromString(bytearray(data.numpy()))
    
    args:
      - frame [waymo_open_dataset.dataset_pb2.Frame]: a waymo frame, contains images and annotations
      - camera_name [str]: one frame contains images and annotations for multiple cameras
    
    returns:
      - encoded_jpeg [bytes]: jpeg encoded image
      - annotations [protobuf object]: bboxes and classes
    """
    # get image
    images = frame.images
    for im in images:
        if open_dataset.CameraName.Name.Name(im.name) != camera_name:
            continue
        encoded_jpeg = im.image
    
    # get bboxes
    labels = frame.camera_labels
    for lab in labels:
        if open_dataset.CameraName.Name.Name(lab.name) != camera_name:
            continue
        annotations = lab.labels
    return encoded_jpeg, annotations


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))