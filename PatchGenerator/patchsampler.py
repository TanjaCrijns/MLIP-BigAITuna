import numpy as np
import threading

class PatchSampler(object):
	
	def __init__(self, patch_source, patch_shape, label_shape, label_list, augmentation_ratio, augmentation_list, sampling_mode):
		'''
		'''
		self._label_list = []
		self._patch_shape = (0,0)
		self._label_shape = None
		self._augmentation_ratio = 0.0
		self._augmentation_list = []
		self._sampling_mode = 'bounding_box'
		
		self._set_labels(label_list)
		self._set_shapes(
		
		
		