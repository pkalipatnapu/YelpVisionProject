# imports
import json
import time
import pickle
import scipy.misc
import skimage.io
import caffe
import csv

import numpy as np
import os.path as osp

from xml.dom import minidom
from random import shuffle
from threading import Thread
from PIL import Image

from tools import SimpleTransformer


"""
This file is taken from caffe example for PASCAL multilabel classification.
"""



class YelpMultilabelSync(caffe.Layer):

    """
    This is a simple syncronous datalayer for training a multilabel model on
    Yelp.
    """

    def setup(self, bottom, top):

        self.top_names = ['data', 'label']

        # === Read input parameters ===

        # params is a python dictionary with layer parameters.
        params = eval(self.param_str)

        # Check the paramameters for validity.
        check_params(params)

        # store input as class variables
        self.batch_size = params['batch_size']

        # Create a batch loader to load the images.
        self.batch_loader = BatchLoader(params, None)

        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape(
            self.batch_size, 3, params['im_shape'][0], params['im_shape'][1])
        # Note the 20 channels (because Yelp has 20 classes.)
        top[1].reshape(self.batch_size, 9)

        print_info("YelpMultilabelSync", params)

    def forward(self, bottom, top):
        """
        Load data.
        """
        for itt in range(self.batch_size):
            # Use the batch loader to load the next image.
            im, multilabel = self.batch_loader.load_next_image()

            # Add directly to the caffe data layer
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = multilabel

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
		# TODO(prad): Add code to reshape the image to 227x227 randomly.
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass


class BatchLoader(object):

    def __init__(self, params, result):
        self.result = result
        self.batch_size = params['batch_size']
        self.yelp_root = params['yelp_root']
        self.im_shape = params['im_shape']
        # get list of image indexes.
        list_csv = self.yelp_root + params['split'] + '_photo_to_biz_ids.csv'
        image_key = []
        with open(list_csv) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                image_key.append(row)
		
        self._cur = 0  # current image
        # this class does some simple data-manipulations
        self.transformer = SimpleTransformer()

        print "BatchLoader initialized with {} images".format(
            len(self.image_key))

    	"""classes = ('good_for_lunch', 'good_for_dinner', 'takes_reservations', 'outdoor_seating',
				   'restaurant_is_expensive', 'has_alcohol', 'has_table_service',
				   'ambience_is_classy', 'good_for_kids')"""

        attributes_csv = osp.join(yelp_root, 'train.csv')
        self.attributes_dict = {}

        with open(attributes_csv) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                attr_string = row["labels"]
                self.attributes_dict[row[business_id]] = [int(label) for label in row[labels].split()]
		

    def load_next_image(self):
        """
        Load the next image in a batch.
        """
        # Did we finish an epoch?
        if self._cur == len(self.image_key):
            self._cur = 0
            shuffle(self.image_key)

        # Load an image
        photo_id = self.image_key[self._cur]["photo_id"]  # Get the image index
        business_id = self.image_key[self._cur]["business_id"]
        image_file_name = photo_id + '.jpg'
        im = np.asarray(Image.open(osp.join(self.pascal_root, 'JPEGImages', image_file_name)))
        im = scipy.misc.imresize(im, self.im_shape)  # resize

        # do a simple horizontal flip as data augmentation
        flip = np.random.choice(2)*2-1
        im = im[:, ::flip, :]

        # Load and prepare ground truth
        multilabel = np.zeros(9).astype(np.float32)
        if split in ["train", "val"]:
            anns = load_yelp_attributes(business_id)
            for label in anns:
                # convert label information to a 1/0 array.
                multilabel[label] = 1

        self._cur += 1
        return self.transformer.preprocess(im), multilabel


    def load_yelp_attributes(business_id):
        return self.attributes_dict[business_id]


def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """
    assert 'split' in params.keys(
    ), 'Params must include split (train, val, or test).'

    required = ['batch_size', 'yelp_root', 'im_shape']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)


def print_info(name, params):
    """
    Ouput some info regarding the class
    """
    print "{} initialized for split: {}, with bs: {}, im_shape: {}.".format(
        name,
        params['split'],
        params['batch_size'],
        params['im_shape'])

