import numpy as np
import threading

class PatchSampler(object):
	
	def __init__(self, patch_source, label_list, augmentation_list, sampling_mode):
		"""
		"""
		self._label_list = label_list
		self._augmentation_list = []
		self._sampling_mode = sampling_mode
		
	def _configure_source(self, patch_source, sampling_mode):
		"""
		"""
		pass
	
	def _create_augmenters(self, augmentation_list):
		"""
		"""
		for aug in augmentation_list:
			if aug = 'flip':
				pass
				
	
	def sample(self, count, patch_shape, label_shape, augmentation_ratio):
		"""
		"""
		pass
		
		
class BoundingBoxSampler(object):
	def __init__(self, patch_source, label_list, augmentation_list, sampling_mode = 'bounding_box'):
		"""
		"""
		
	