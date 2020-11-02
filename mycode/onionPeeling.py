import numpy as np
from typing import List, Tuple
from scipy.spatial import distance
from mycode.utils import Point, mahalanobis, toVector, list_splice, giftWrapping

class OnionPeeling:
    def __init__(self, pts: List[Point]):
        self.data = pts
    
    def getDistance(self, p: Point, distance_metric: str='euclidean'):
        dst = None
        if distance_metric=='euclidean':
            mean = np.mean(self.data, axis=0)
            dst = distance.euclidean(p, mean)
        elif distance_metric=='mahalanobis':
            dst = mahalanobis(x=p, data=self.data)
        else:
            print("ERRO: Distance metric not found: ", distance_metric)
        return dst
    
    def convexHull(self, pts: List[Point], algorithm='gift-wrapping'):
        if algorithm=='gift-wrapping':
            return giftWrapping(pts)
        else:
            return None

    def outlierDetection(self, k: int, distance_metric='euclidean', convex_hull='gift-wrapping'):
        L = [] # List of points representing outliers
        S_ = self.data[:] # make copy points
        if k<=0:
            print('outliers(k) must be greater than 0.')
            return L, self.convexHull(S_, algorithm=convex_hull)
        while True:
            # Run Convex-Hull algorithm
            hull = self.convexHull(S_, algorithm=convex_hull)
            
            # check the size of points
            if len(S_)<k:
                print('Size of points must be greater than outliers(k).')
                return L, hull
            
            # Start removing points one at a time
            # Calculate distance with requested distance metric
            dst = []
            for pt in hull:
                dst.append(self.getDistance(pt, distance_metric))
            # Find the point with largest distance to all the points in the data
            idx_pt = np.argmax(dst)
            idx_ = S_.index(hull[idx_pt])
            # We remove the furthest point found in Hull
            remove = list_splice(S_, idx_, 1)
            # Store the point and the corresponding index in the memory.
            L.append(remove[0])
            if len(L)==k:
                return L, self.convexHull(S_, algorithm=convex_hull)
            # Remove the current hull.
            hull = None