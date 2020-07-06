import hou
import math
import numpy as np
from collections import defaultdict

node = hou.pwd()
geo = node.geometry()
boundaries = geo.pointGroups()
edge_boundaries = []
neighbor_boundaries = []
for edge_group in geo.edgeGroups():
  if "neighbor" in edge_group.name():
    neighbor_boundaries.append(edge_group)
  else:
    edge_boundaries.append(edge_group)

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
  edges_neighbors = neighbor_boundaries[i].edges()
  '''
  2. Fill small holes with centroid-based method
                       | 1 if h has <= 6 points in it
    is_small_hole(h) = | 0 otherwise
  '''
  # TODO: Change small hole definition to n-gon?
  if len(points) <= 8:
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
  elif len(points) <= 12:
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
            curr_polys = [(self.points[i], self.points[j], self.points[k])] + min_polys_left + min_polys_right
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
            new_poly.addVertex(min_poly[0])
            new_poly.addVertex(min_poly[1])
            new_poly.addVertex(min_poly[2])
        return min_polys
    
    min_polys = MinTriangulation(geo, points, cache_costs=cache_lengths).min_triangulation(generate=False)
    '''
    3B. Conduct Triangle Splitting
      We split the minimum polygons with centroid-based method if:
        ALL_t, t elem(p_i, p_j, p_k), sqrt(2) * ||p_c-t|| > s(p_c) and sqrt(2) * ||p_c-t|| > s(t)
        where
          p_i, p_j, p_k = points of minimum polygon
          p_c = center point of minimum polygon
          s = scale factor function, average length of edges connected to point except
              for hole boundary edges
    '''
    points_neighbors = defaultdict(list)
    for edges_neighbor in list(set(edges_neighbors) - set(edges)):
      p_1, p_2 = edges_neighbor.points()
      if p_1 in points:
        points_neighbors[p_1.number()].append(p_2)
      elif p_2 in points:
        points_neighbors[p_2.number()].append(p_1)
      p1_pos = p_1.position()
      p2_pos = p_2.position()
      cache_lengths[unord_hash(p_1.number(), p_2.number())] = (p1_pos - p2_pos).length()
    
    exterior_edges_hashed = []
    for edge in edges:
      p_1, p_2 = edge.points()
      exterior_edges_hashed.append(unord_hash(p_1.number(), p_2.number()))
    interior_edges = []
    for min_poly in min_polys:
      for p_1 in min_poly:
        for p_2 in filter(lambda x: x != p_1, min_poly):
          if not unord_hash(p_1.number(), p_2.number()) in exterior_edges_hashed:
            interior_edges.append((p_1, p_2))
            points_neighbors[p_2.number()].append(p_1)
            points_neighbors[p_1.number()].append(p_2)

    new_min_polys = min_polys
    min_polys_created = True
    while min_polys_created:
      min_polys_created = False
      fixed_new_min_polys = new_min_polys
      for min_poly in fixed_new_min_polys:
        p_i, p_j, p_k = min_poly
        ts = [p_i, p_j, p_k]
        center = (p_i.position() + p_j.position() + p_k.position()) / 3
        eic_len, ejc_len, ekc_len = (center - p_i.position()).length(), (center - p_j.position()).length(), (center - p_k.position()).length()
        c_scale = eic_len + ejc_len + ekc_len
        c_normal = np.zeros(3)

        split = True
        for t in ts:
          c_normal += t.attribValue("N")
          t_scale = 0
          t_neighbors = points_neighbors[t.number()]
          for t_neighbor in t_neighbors:
            if not unord_hash(t.number(), t_neighbor.number()) in exterior_edges_hashed:
              t_scale += cache_lengths[unord_hash(t.number(), t_neighbor.number())]
          if math.sqrt(2) * (center - t.position()).length() <= min(t_scale, c_scale):
            split = False
        c_normal /= 3

        if split:
          p_c = geo.createPoint()
          p_c.setPosition(center)
          p_c.setAttribValue("N", c_normal)
          new_min_polys.remove((p_i, p_j, p_k))
          new_min_polys.extend([(p_i, p_c, p_j), (p_i, p_c, p_k), (p_k, p_c, p_j)])
          cache_lengths[unord_hash(p_i.number(), p_c.number())] = eic_len
          cache_lengths[unord_hash(p_k.number(), p_c.number())] = ekc_len
          cache_lengths[unord_hash(p_j.number(), p_c.number())] = ejc_len
          points_neighbors[p_i.number()].append(p_c)
          points_neighbors[p_j.number()].append(p_c)
          points_neighbors[p_k.number()].append(p_c)
          points_neighbors[p_c.number()] = [p_i, p_j, p_k]
          min_polys_created = True
    
    for min_poly in new_min_polys:
      p_i, p_j, p_k = min_poly
      new_poly = geo.createPolygon()
      new_poly.addVertex(p_i)
      new_poly.addVertex(p_j)
      new_poly.addVertex(p_k)
    '''
    3C. Conduct Edge-Swapping
    '''

  else:
    '''
    4. Fill large hole with advancing front method
    '''
    continue
  
node.bypass(True)