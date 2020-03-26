'''
The code is based on Adrian Rosebrock's tutorial
Simple object tracking with OpenCV
https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
'''

from collections import OrderedDict
import numpy as np
from scipy.spatial import distance as dist
import math

class ObjectTracker:

    def __init__(self, maxDisappeared=12):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.disappeared = OrderedDict()

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

        self.framebuf = []


    def register(self, object):
        self.objects[self.nextObjectID] = object
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, input_objects):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(input_objects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # return early as there are no centroids or tracking info
            # to update
            return self.objects

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(input_objects), 2), dtype="int")

        # loop over the bounding boxes
        for i in range(len(input_objects)):
            x1 = input_objects[i].bbox[0]
            y1 = input_objects[i].bbox[1]
            x2 = input_objects[i].bbox[2]
            y2 = input_objects[i].bbox[3]
            w = (x2 - x1)
            h = (y2 - y1)
            inputCentroids[i] = (int( (x1+x2)/2 ), int( (y1 + y2) / 2))
            input_objects[i].center = inputCentroids[i]
        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(input_objects[i])

        # otherwise, there are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objects_stored = list(self.objects.values())
            objectCentroids = np.zeros((len(objects_stored), 2), dtype="int")
            for i in range(len(objects_stored)):
                objectCentroids[i] = objects_stored[i].center

                # min_dist_idx = -1
                # min_dist = 1e8
                # for j in range(len(input_objects)):
                #     dx = objectCentroids[i][0] - inputCentroids[j][0]
                #     dy = objectCentroids[i][1] - inputCentroids[j][1]
                #     dist = math.sqrt(dx*dx + dy*dy)
                #     if dist < min_dist:
                #         min_dist = dist
                #         min_dist_idx = j



            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value as at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                # self.objects[objectID] = input_objects[col]
                self.objects[objectID].bbox = input_objects[col].bbox
                self.objects[objectID].center = input_objects[col].center
                self.disappeared[objectID] = 0
                self.objects[objectID].isdissapeared = False
                self.objects[objectID].trackingStatus = False
                self.objects[objectID].score = input_objects[col].score
                self.objects[objectID].lable = input_objects[col].label
                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    self.objects[objectID].isdissapeared = True
                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(input_objects[col])

        # return the set of trackable objects
        return self.objects

    def update_particles(self, input_objects):

        return


class Object:

    def __init__(self):
        self.frameStart = 0
        self.frameEnd = 0
        self.frameCurrent = 0
        self.id = 0
        self.trackingStatus = False
        self.isdissapeared = False
        self.label = 0
        self.center = 0
        self.bboxList = []
        self.scoreList = []
        self.maskList = []
        self.state = None
        self.bbox = None
        self.location = None
        self.particles = []
        self.score = 0

















