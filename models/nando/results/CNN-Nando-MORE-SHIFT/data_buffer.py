import queue


class DataBuffer():

	SIZE = 2

	def __init__(self):
		self.queue = queue.Queue(maxsize=self.SIZE)

	def add_item(self, item):
		if self.queue.full():
			self.queue.get(False) # remove fist item from queue
			self.queue.put(item, False)
		else:
			self.queue.put(item, False)

	def get_item_for_processing(self):
		try:
			return self.queue.get(False)
		except queue.Empty:
			return None
		