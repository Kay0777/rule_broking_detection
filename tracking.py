from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():
	def __init__(self, maxDisappeared=5, maxDistance=100):
		self.highestIDUsed = -1
		self.nextObjectID = 0
		self.usedIDs = set()  # Set to track all used IDs
		self.objects = OrderedDict()
		self.bboxes = OrderedDict()
		self.disappeared = OrderedDict()
		self.maxDisappeared = maxDisappeared
		self.maxDistance = maxDistance

	def register(self, centroid, bbox):
        # Always use one more than the highest ID ever used
		newID = self.highestIDUsed + 1
		self.objects[newID] = centroid
		self.bboxes[newID] = bbox
		self.disappeared[newID] = 0
		self.highestIDUsed = newID

	def deregister(self, objectID):
		# Deregister the object but do not remove the ID from usedIDs
		del self.objects[objectID]
		del self.bboxes[objectID]
		del self.disappeared[objectID]

	def update(self, rects):
		if len(rects) == 0:
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)
			return self.objects

		inputCentroids = np.zeros((len(rects), 2), dtype="int")
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)

		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i], rects[i])
		else:
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			D = dist.cdist(np.array(objectCentroids), inputCentroids)
			rows = D.min(axis=1).argsort()
			cols = D.argmin(axis=1)[rows]
			
			usedRows = set()
			usedCols = set()
			for (row, col) in zip(rows, cols):
				if row in usedRows or col in usedCols:
					continue

				if D[row, col] > self.maxDistance:  # Check if distance exceeds threshold
					continue
				objectID = objectIDs[row]
				
				self.objects[objectID] = inputCentroids[col]
				self.bboxes[objectID] = rects[col] 
				self.disappeared[objectID] = 0

				usedRows.add(row)
				usedCols.add(col)

			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)
			if D.shape[0] >= D.shape[1]:
				for row in unusedRows:
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)
			else:
				for col in unusedCols:
					self.register(inputCentroids[col], rects[col])
					
		output = []
		for objectID, centroid in self.objects.items():
			if self.disappeared[objectID] == 0:
				bbox = self.bboxes[objectID]
				output.append((objectID, centroid, bbox))

		return output
