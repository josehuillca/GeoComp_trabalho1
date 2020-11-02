import numpy as np
import scipy as sp
from scipy.spatial import distance
import skgeom as sg
from typing import List, Tuple, Any

# =============================================================================================
#                                   BASIC ALGORITHMS
# =============================================================================================

# Nós definimos a estrutura de um ponto em 2-dimensões
Point = Tuple[float, float]

# Usamos esta função apenas para desenhar os pontos usando a função DRAW da scikit-geometry
def toPoint2(p: Point) -> sg.Point2:
    return sg.Point2(p[0], p[1])

# Return a vector with homogeneos coordenate
def toVector(p1: Point, p2: Point) -> np.ndarray:
    v = np.array(p1) - np.array(p2)
    return np.array([v[0], v[1], 1])

# Nós geramos pontos aleatórios seguindo uma distribuição normal
def makePoints(nPoints: int, outlierPercent: int=15) -> List[Point]:
    # generate data
    n_features = 2
    gen_cov = np.eye(n_features)
    n_outliers = int((outlierPercent * nPoints) /100.)
    gen_cov[0, 0] = 2.
    X = np.dot(np.random.randn(nPoints, n_features), gen_cov)
    # add some outliers
    outliers_cov = np.eye(n_features)
    outliers_cov[np.arange(1, n_features), np.arange(1, n_features)] = 7.
    X[-n_outliers:] = np.dot(np.random.randn(n_outliers, n_features), outliers_cov)
    return X.tolist()
    #xSample = np.random.normal(mean, std, nPoints)
    #ySample = np.random.normal(mean, std, nPoints)

    #return list(zip(xSample, ySample))

def list_splice(target: List[Any], start: int, delete_count: int=None, *items):
    """Remove existing elements and/or add new elements to a list.

    target        the target list (will be changed)
    start         index of starting position
    delete_count  number of items to remove (default: len(target) - start)
    *items        items to insert at start index

    Returns a new list of removed items (or an empty list)
    """
    if delete_count == None:
        delete_count = len(target) - start

    # store removed range in a separate list and replace with *items
    total = start + delete_count
    removed = target[start:total]
    target[start:total] = items

    return removed

# Uma vez que usamos apenas dados bidimensionais, então será p=2 colunas, representado por Point
# Mas você pode realmente trabalhar com n-dimensões, onde n = p
# Algoritmo baseado em: https://www.machinelearningplus.com/statistics/mahalanobis-distance/
def mahalanobis(x:Point, data:List[Point], cov: np.ndarray=None):
    """Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = np.array([x]) - np.mean(np.array(data), axis=0)
    if not cov:
        cov = np.cov(np.array(data).T)
    inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()[0]

def sortPoints(points:List[Point]) -> List[Point]:
    """Return point_array sorted by leftmost first, then by slope, ascending."""

    def slope(y:Point):
        """returns the slope of the 2 points."""
        x = points[0]
        return (x[1] - y[1]) / (x[0] - y[0])

    points.sort()  # put leftmost first
    points = points[:1] + sorted(points[1:], key=slope)
    return points


# =============================================================================================
#                                   CONVEX-HULL ALGORITHMS
# =============================================================================================
def giftWrapping(data: List[Point]):
    pts = data[:]
    pts.sort(key=lambda x: x[0]) # ordenamos por el eje x

    hull = []
    index = 2
    nextIndex = -1
    
    leftMost = pts[0]
    currentVertex = leftMost
    hull.append(currentVertex)
    nextVertex = pts[1]
    try:
        while True:
            checkingPoint = pts[index]
            u = toVector(nextVertex, currentVertex)
            v = toVector(checkingPoint, currentVertex)
            cross = np.cross(u, v)
            if cross[2]>0:
                nextVertex = checkingPoint
                nextIndex = index
            index += 1
            if index == len(pts):
                if nextVertex == leftMost:
                    return hull
                else:
                    hull.append(nextVertex)
                    currentVertex = nextVertex
                    index = 0
                    _ = list_splice(pts, nextIndex, 1)
                    nextVertex = leftMost
    except Exception as e:
        print('Exception:',repr(e))
        return hull

def grahamAlgorithm(data: List[Point]):
    """Takes an array of points to be scanned.
    Returns an array of points that make up the convex hull surrounding the points passed in in point_array.
    """
    def crossProdOrientation(a: Point, b: Point, c: Point):
        """Returns the orientation of the set of points.
        >0 if x,y,z are clockwise, <0 if counterclockwise, 0 if co-linear.
        """
        return (b[1] - a[1]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[1] - a[1])
    pts_cp = data[:]
    # convex_hull is a stack of points beginning with the leftmost point.
    convex_hull = []
    sorted_points = sortPoints(pts_cp)
    for p in sorted_points:
        # if we turn clockwise to reach this point, pop the last point from the stack, else, append this point to it.
        while len(convex_hull) > 1 and crossProdOrientation(convex_hull[-2], convex_hull[-1], p) >= 0:
            convex_hull.pop()
        convex_hull.append(p)
    # the stack is now a representation of the convex hull, return it.
    return convex_hull

# https://github.com/samfoy/GrahamScan/blob/master/graham_scan.py

# =============================================================================================
#                                   DRAW(JUPYTER) ALGORITHMS
# =============================================================================================
from skgeom.draw import draw

def drawPoints(pts: List[Point], cor: str='blue'):
    points_sg = []
    for i in range(len(pts)):
        x, y = pts[i]
        points_sg.append(sg.Point2(x, y))
    draw(points_sg, color=cor) # draw all points

def drawHull(hull: List[Point], cor: str='green'):
    for i in range(len(hull)-1):
        draw(sg.Segment2(toPoint2(hull[i]),toPoint2(hull[i+1])), color=cor)
    draw(sg.Segment2(toPoint2(hull[-1]),toPoint2(hull[0])), color=cor)