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
import sys
import PIL.Image
from StringIO import StringIO

import caffe
import lmdb
import os
import caffe.proto.caffe_pb2
from caffe.io import datum_to_array

"""
This file is taken from caffe example for PASCAL multilabel classification.
"""



class YelpMultilabelSync(caffe.Layer):

    """
    This is a simple syncronous datalayer for training a multilabel model on
    Yelp.
    """

    def setup(self, bottom, top):
	self.top_names = ['data', 'label', 'business_id', 'image_id']

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
        top[0].reshape(
            self.batch_size, 3, params['im_shape'][0], params['im_shape'][1])

        # Note the 9 channels (because Yelp has 9 classes.)
        top[1].reshape(self.batch_size, 9)
        top[2].reshape(self.batch_size, 1)
        top[3].reshape(self.batch_size, 1)

        print_info("YelpMultilabelSync", params)

    def forward(self, bottom, top):
        """
        Load data.
        """
        for itt in range(self.batch_size):
            # Use the batch loader to load the next image.
            data, multilabel, photo_id, business_id = self.batch_loader.load_next_image()

            # Add directly to the caffe data layer
            top[0].data[itt, ...] = data
            top[1].data[itt, ...] = multilabel
            top[2].data[itt, ...] = photo_id
            top[3].data[itt, ...] = business_id

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
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
        self.yelp_picture_root = params['yelp_picture_root']
        self.yelp_csv_root = params['yelp_csv_root']
        self.im_shape = params['im_shape']
	self.split=params['split']

        # get list of image indexes.
        if self.split in ["train", "validation"]:
       	    list_csv = self.yelp_csv_root + self.split + '_photo_to_biz_ids2.csv'
	else:
       	    list_csv = self.yelp_csv_root + self.split + '_photo_to_biz.csv'
        self.image_key = []
        with open(list_csv) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                self.image_key.append(row)
		
        self._cur = 0  # current image
        # this class does some simple data-manipulations
        self.transformer = SimpleTransformer()

        print "BatchLoader initialized with {} images".format(
            len(self.image_key))

	if self.split in ["train", "validation", "poster"]:
	    attributes_csv = osp.join(self.yelp_csv_root, self.split + '2.csv')
            self.attributes_dict = {}

	    if self.split == "poster":
		attributes_csv = osp.join(self.yelp_picture_root, self.split + '.csv')

            with open(attributes_csv) as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    attr_string = row["labels"]
                    self.attributes_dict[row["business_id"]] = [int(label) for label in row["labels"].split()]

	if self.split == 'validation':
	    lmdb_dir = self.yelp_picture_root + 'val_lmdb'
	else:
	    lmdb_dir = self.yelp_picture_root + self.split + '_lmdb'
	lmdb_env = lmdb.open(lmdb_dir)
	self.lmdb_txn = lmdb_env.begin()
	self.lmdb_cursor = self.lmdb_txn.cursor()
	

    def load_next_image(self):
        """
        Load the next image in a batch.
	"""
        # Did we finish an epoch?
        if self._cur == len(self.image_key):
            self._cur = 0
	if not self.lmdb_cursor.next():
	    self.lmdb_cursor = self.lmdb_txn.cursor()
	    self.lmdb_cursor.next()

	business_id = self.image_key[self._cur]["business_id"]
        photo_id = self.image_key[self._cur]["photo_id"]  # Get the image index

        # Load and prepare ground truth
        multilabel = np.zeros(9).astype(np.float32)
        if self.split in ["train", "validation","poster"]:
            anns = self.load_yelp_attributes(business_id)
            for label in anns:
                # convert label information to a 1/0 array.
                multilabel[label] = 1

        self._cur += 1
	datum = caffe.proto.caffe_pb2.Datum()
	key, value = self.lmdb_cursor.item()

    	datum.ParseFromString(value)
        label = datum.label
	# TODO(prad): Add a check for test as well.
	if self.split != 'test' and self.split != 'poster' and str(label) != business_id:
	    print "Houston, we have a problem." + str(label) + ":" + str(business_id)

	data = caffe.io.datum_to_array(datum)
	im = scipy.misc.imresize(data, self.im_shape)  # resize

        # do a simple horizontal flip as data augmentation
        flip = np.random.choice(2)*2-1
        im = im[:, ::flip, :]
	# The datum above already does BGR transformation. But we are redoing it using the transformer below.
	im = im[:, :, ::-1]
	
	return self.transformer.preprocess(im), multilabel, photo_id, label


    def load_yelp_attributes(self, business_id, mapping=None):
        return self.attributes_dict[business_id]


def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """
    assert 'split' in params.keys(
    ), 'Params must include split (train, validation, poster or test).'

    required = ['batch_size', 'yelp_picture_root', 'yelp_csv_root', 'im_shape']
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

