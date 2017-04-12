from . import patchbuffer as buffer
from . import batchsampler as bsampler

import threading
import numpy as np
import Queue


class PatchGenerator(object):

	def __init__(self, patch_source, label_list, main_cache_size, read_size, patch_shape, label_shape, buffer_mode, augmentation_ratio, augmentation_list, sampling_mode, multi_threaded): 
		"""
		"""
		
		super(PatchGenerator, self).__init__()
		
		self._cache_size = 0
		self._read_size = 0
		self._patch_shape = (0,0)
		self._label_shape = None
		self._label_list = []
		self._multi_threaded = False
		self._augmentation_ratio = 0.
		self._patch_buffer = None
		self._read_buffer = None
		self._batch_sampler = None

		
		self._init_buffers(main_cache_size, read_size, patch_shape, label_shape, buffer_mode)
		self._init_sampler(patch_source, label_list, augmentation_ratio, augmentation_list, sampling_mode, multi_threaded)
		
	
	def _init_buffers(self, main_cache_size, read_size, patch_shape, label_shape, buffer_mode):
		"""
		Initiate main buffer and read buffer if double buffering.
		"""
		
		self._cache_size = main_cache_size
		self._patch_shape = patch_shape
		self._label_shape = label_shape

		self._patch_buffer = buffer.PatchBuffer(self._patch_shape, self._label_shape, self._cache_size, buffer_mode)

		if read_size is not None:
			self._read_size = read_size
			self._read_buffer = buffer.PatchBuffer(self._patch_shape, self._label_shape, self._read_size, buffer_mode)

	def _init_sampler(self, patch_source, label_list, augmentation_ratio, augmentation_list, sampling_mode, multi_threaded):
		'''
		'''
		self._multi_threaded = multi_threaded
		self._label_list = label_list
		self._augmentation_ratio = augmentation_ratio
		
		if self._multi_threaded:
			raise NotImplementedError('multi-threading mode not implemented yet')
		else:
			self._batch_sampler = bsampler.BatchSampler(patch_source, self._patch_shape, self._label_shape, self._label_list, self._augmentation_ratio, augmentation_list, sampling_mode)
		
		
		
	def _load(self, batch_size, buffer):
		"""
		Fill the target buffer with a batch of patches.
		"""
		
		patches, labels = self._batch_sampler.gen_batch(self._augmentation_ratio)
		buffer.push(patches, labels)
		
	def load(self,batch_size):
		"""
		"""
		
		if self._read_buffer:
			self._load(batch_size, self._read_buffer)
		else:
			self._load(batch_size, self._patch_buffer)
			
	def transfer(self):
		"""
		"""
		
		patches, labels = self._read_buffer.get_batch(self._read_size)
		self._patch_buffer.push_patches(patches, labels)
		
	def batch(self, batch_size):
		"""
		"""
		
		patches, labels, indices = self._patch_buffer.get_batch(batch_size)
		
		patches = patches.astype(float32) / 255.0
		
		return patches, labels, indices
		
		
		
		
		