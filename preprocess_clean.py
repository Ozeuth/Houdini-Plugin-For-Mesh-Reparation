import hou
import math
import numpy as np
import time

node = hou.pwd()
geo = node.geometry()
inputs = node.inputs()
boundaries = geo.pointGroups()
edge_boundaries = geo.edgeGroups()
# NOTE: points ordered, but ordering breaks after deletion
for i in range(1, len(boundaries)):
  points = boundaries[i].points()
  edges = edge_boundaries[i].edges()
  '''
  2. Fill small holes with centroid-based method
                       | 1 if h has <= 6 points in it
    is_small_hole(h) = | 0 otherwise
  '''
  # TODO: Change small hole definition to n-gon?
  if len(points) <= 6:
    center = np.zeros(3)
    normal = np.zeros(3)
    for point in points:
      center += point.position()
      normal += point.attribValue("N")
    center /= len(points)
    normal /= len(points)
    centroid = geo.createPoint()
    centroid.setPosition(center)
    centroid.setAttribValue("N", normal)
    for edge in edges:
      new_poly = geo.createPolygon()
      new_poly.addVertex(centroid)
      edge_points = edge.points()
      for edge_point in edge_points:
        new_poly.addVertex(edge_point)
  elif len(points) <= 50:
    '''
    3. Fill Medium hole with projection-based method
    3A. Initialize with minimum area triangulation
    '''
    class MinTriangulation():
      # Works due to Houdini point ordering
      def __init__(self, geo, points):
        cache_costs = {}
        for i in range(len(points)):
          for j in range(i+1, len(points)):
            p1_pos = points[i].position()
            p2_pos = points[j].position()
            cache_costs[(i, j)] = math.sqrt(math.pow(p2_pos[0] - p1_pos[0], 2) + math.pow(p2_pos[1] - p1_pos[1], 2) + math.pow(p2_pos[2] - p1_pos[2], 2))
        self.geo = geo
        self.points = points
        self.cache_costs = cache_costs

      def tri_cost(self, i, j, k, is_mwt=True):
        eik_len = self.cache_costs[(i, k)]
        ekj_len = self.cache_costs[(k, j)]
        if is_mwt:
          return eik_len + ekj_len
        else:
          eij_len = self.cache_costs[(i, j)]
          s = eij_len + eik_len + ekj_len / 2
          return math.sqrt(s*(s-eij_len)*(s-eik_len)*(s-ekj_len))

      def tri_min(self, i, j, optimized=False):
        if j <= i+1:
          return (0, [])
        else:
          min_cost = float('inf')
          potential_polys = {}
          for k in range(i+1, j):
            if optimized:
              curr_cost = self.tri_cost(i, j, k)
              if k - i >= j - k:
                elem1 = i
                elem2 = k
                elem3 = k
                elem4 = j
              else:
                elem1 = k
                elem2 = j
                elem3 = i
                elem4 = k
              min_cost_1, min_polys_1 = self.tri_min(elem1, elem2)
              curr_cost += min_cost_1
              if curr_cost < min_cost:
                min_cost_2, min_polys_2 = self.tri_min(elem3, elem4)
                curr_cost += min_cost_2
                if curr_cost < min_cost:
                  min_cost = curr_cost
                  curr_polys = [(i, j, k)] + min_polys_1 + min_polys_2
                  potential_polys[curr_cost] = curr_polys
            else:
              cost_center = self.tri_cost(i, j, k)
              min_cost_left, min_polys_left = self.tri_min(i, k)
              min_cost_right, min_polys_right = self.tri_min(k, j)
              curr_cost = cost_center + min_cost_left + min_cost_right
              curr_polys = [(i, j, k)] + min_polys_left + min_polys_right
              if curr_cost < min_cost:
                min_cost = curr_cost
                potential_polys[curr_cost] = curr_polys

          min_polys = potential_polys[min_cost]
        return min_cost, min_polys

      def min_triangulation(self):
        start = time.time()
        _, min_polys = self.tri_min(0, len(self.points)-1)
        print(str(time.time() - start))
        for min_poly in min_polys:
          new_poly = self.geo.createPolygon()
          new_poly.addVertex(self.points[min_poly[0]])
          new_poly.addVertex(self.points[min_poly[1]])
          new_poly.addVertex(self.points[min_poly[2]])
    if i == 3:
      MinTriangulation(geo, points).min_triangulation()

  else:
    '''
    4. Fill large hole with advancing front method
    '''
    continue
  
node.bypass(True)