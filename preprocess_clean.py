import hou
import math
import numpy as np

node = hou.pwd()
geo = node.geometry()
boundaries = geo.pointGroups()
edge_boundaries = geo.edgeGroups()
boundaries_neighbors = node.inputs()[1].geometry().edgeGroups()

def unord_hash(a, b):
  if a < b:
    return a * (b - 1) + math.trunc(math.pow(b - a - 1, 2)/ 4)
  elif a > b:
    return (a - 1) * b + math.trunc(math.pow(a - b - 1, 2)/ 4)
  else:
    return a * b + math.trunc(math.pow(abs(a - b) - 1, 2)/ 4)

# NOTE: points ordered, but ordering breaks after deletion.
#       Min triangulation relies on ordering
for i in range(1, len(boundaries)):
  points = boundaries[i].points()
  edges = edge_boundaries[i].edges()
  edges_neighbors = boundaries_neighbors[i].edges()
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
  elif len(points) <= 10:
    '''
    3. Fill Medium hole with projection-based method
    3A. Initialize with minimum area triangulation
    '''
    cache_lengths = {}
    for i in range(len(points)):
      for j in range(i+1, len(points)):
        p_i, p_j = points[i], points[j]
        pi_pos = p_i.position()
        pj_pos = p_j.position()
        cache_lengths[unord_hash(p_i.number(), p_j.number())] = (pi_pos - pj_pos).length()
    
    class MinTriangulation():
      def __init__(self, geo, points, cache_costs=None):
        if cache_costs is None:
          cache_costs = {}
          for i in range(len(points)):
            for j in range(i+1, len(points)):
              p_i, p_j = points[i], points[j]
              pi_pos = p_i.position()
              pj_pos = p_j.position()
              cache_costs[unord_hash(p_i.number(), p_j.number())] = (pi_pos - pj_pos).length()
        self.geo = geo
        self.points = points
        self.cache_costs = cache_costs

      def tri_cost(self, i, j, k, is_mwt=True):
        eik_len = self.cache_costs[unord_hash(points[i].number(), points[k].number())]
        ekj_len = self.cache_costs[unord_hash(points[k].number(), points[j].number())]
        if is_mwt:
          return eik_len + ekj_len
        else:
          eij_len = self.cache_costs[unord_hash(i, j)]
          s = eij_len + eik_len + ekj_len / 2
          return math.sqrt(s*(s-eij_len)*(s-eik_len)*(s-ekj_len))

      def tri_min(self, i, j):
        if j <= i+1:
          return (0, [])
        else:
          min_cost = float('inf')
          potential_polys = {}
          for k in range(i+1, j):
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

      def min_triangulation(self, generate=True):
        _, min_polys = self.tri_min(0, len(self.points)-1)
        if generate:
          for min_poly in min_polys:
            new_poly = self.geo.createPolygon()
            new_poly.addVertex(self.points[min_poly[0]])
            new_poly.addVertex(self.points[min_poly[1]])
            new_poly.addVertex(self.points[min_poly[2]])
        return min_polys
    
    min_polys = MinTriangulation(geo, points, cache_costs=cache_lengths).min_triangulation(generate=False)
    '''
    3B. Conduct Triangle Splitting
      We split the minimum polygons with centroid-based method if:
        ALL_t, t elem(i, j, m), sqrt(2) * ||vc-vt|| > s(vc) and sqrt(2) * ||vc-vt|| > s(vt)
        where
          i, j, m = points of minimum polygon
          c = center point of minimum polygon
          s = scale factor function
    '''
    for edges_neighbor in (set(edges_neighbors)-set(edges)):
      p_1, p_2 = edges_neighbor.points()
      p1_pos = p_1.position()
      p2_pos = p_2.position()
      cache_lengths[unord_hash(p_1.number(), p_2.number())] = (p1_pos - p2_pos).length()

    edge_points_hashed = []

    for edge in edges:
      edge_points = edge.points()
      edge_points_hashed.append(unord_hash(edge_points[0].number(), edge_points[1].number()))

    interior_edges = []
    for min_poly in min_polys:
      i, j, k = min_poly
      p_i, p_j, p_k = points[i], points[j], points[k]
      ts = [p_i, p_j, p_k]
      # i and j are neighbour points, definitely not interior edge
      if unord_hash(p_i.number(), p_k.number()) in edge_points_hashed:
        interior_edges.append((min_poly[0], min_poly[2]))
      if unord_hash(p_k.number(), p_j.number()) in edge_points_hashed:
        interior_edges.append((min_poly[2], min_poly[1]))
      
      for t in ts:
        print(t.prims())
  else:
    '''
    4. Fill large hole with advancing front method
    '''
    continue
  
node.bypass(True)