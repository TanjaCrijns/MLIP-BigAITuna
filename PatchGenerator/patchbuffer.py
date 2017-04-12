import numpy as np
import threading

class PatchBuffer(object):
	def __init__(self, patch_shape, label_shape, cache_size, mode = 'ring'):
		"""
		"""
		
		super(PatchBuffer, self).__init__()

		self._mode = mode
		self._read_index = 0
		self._write_index = 0
		self._buffer_size = cache_size
		self._patch_buffer = None
		self._label_buffer = None

		
		self._allocate_space(patch_shape, label_shape, cache_size)
		self._apply_mode()
		self._buffer_lock = threading.Lock()
		
	def _allocate_space(self, patch_shape, label_shape, cache_size):
		"""
		"""
		if patch_shape[0] <= 0 or patch_shape[1] <= 0:
			raise ValueError('Invalid patch shape ({},{})'.format(patch_shape[0], patch_shape[1]))
		
		self._patch_buffer = np.zeros((cache_size, 3, patch_shape[0], patch_shape[1]), dtype=np.uint8)
		if label_shape is not None:
			self._label_buffer = np.zeros((cache_size, 1, label_shape[0], label_shape[1]), dtype=np.uint8)
		else:
			self._label_buffer = np.zeros((cache_size,), dtype=np.uint8)
			
	def _apply_mode(self):
		"""
		"""
		if self._mode == 'random':
			raise NotImplementedError('random mode not implemented yet')
		
		
	def get_batch(self, patch_count):
		"""
		"""
		self._buffer_lock.acquire()
		
		if self._mode == 'random':
			raise NotImplementedError('random mode not implemented yet')
		elif self._mode == 'ring':
			indices = np.arange(self._read_index, self._read_index + patches.shape[0])
			indices %= self._buffer_size
			self._read_index = (self._read_index + patches.shape[0]) % self._buffer_size
			
		labels = self._label_buffer[indices]
		patches = self._patch_buffer[indices]
		
		self._buffer_lock.release()
		
		return patches, labels, indices
		
	def push_patches(self, patches, labels):
		"""
		"""
		self._buffer_lock.acquire()
		
		if self._mode == 'random':
			raise NotImplementedError('random mode not implemented yet')
		elif self._mode == 'ring':
			indices = np.arange(self._write_index, self._write_index + patches.shape[0])
			indices %= self._buffer_size
			self._write_index = (self._write_index + patches.shape[0]) % self._buffer_size
			
		self._label_buffer[indices] = labels
		self._patch_buffer[indices] = patches
		
		self._buffer_lock.release()
			
		