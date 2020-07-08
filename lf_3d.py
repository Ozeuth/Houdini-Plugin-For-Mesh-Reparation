import hou
import itertools
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
    e_lens_hashed = {}
    for i in range(len(points)):
      for j in range(i+1, len(points)):
        p_i, p_j = points[i], points[j]
        pi_pos = p_i.position()
        pj_pos = p_j.position()
        e_lens_hashed[unord_hash(p_i.number(), p_j.number())] = (pi_pos - pj_pos).length()

    class VirtualPolygon():
      # class to avoid generation of Houdini Polygons during intermediary phases
      def __init__(self, p1, p2, p3):
        self.p1, self.p2, self.p3 = p1, p2, p3
      
      def __eq__(self, other):
        ps = [other.p1, other.p2, other.p3]
        return (self.p1 in ps and self.p2 in ps and self.p3 in ps)

      def __str__(self):
        ps = [str(self.p1.number()), str(self.p2.number()), str(self.p3.number())]
        ps.sort()
        return "<" + ', '.join(ps) + ">"
      
      def __repr__(self):
        return str(self)

      def get_edges(self):
        return list(itertools.combinations([self.p1, self.p2, self.p3], 2))

      def get_common_edge(self, other):
        edge_points = []
        self_ps = [self.p1, self.p2, self.p3]
        ps = [other.p1, other.p2, other.p3]
        for self_p in self_ps:
          if self_p in ps:
            edge_points.append(self_p)
        return edge_points
        
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
            curr_polys = [VirtualPolygon(self.points[i], self.points[j], self.points[k])] + min_polys_left + min_polys_right
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
            new_poly.addVertex(min_poly.p1)
            new_poly.addVertex(min_poly.p2)
            new_poly.addVertex(min_poly.p3)
        return min_polys
    
    min_polys = MinTriangulation(geo, points, cache_costs=e_lens_hashed).min_triangulation(generate=False)
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
      e_lens_hashed[unord_hash(p_1.number(), p_2.number())] = (p1_pos - p2_pos).length()
    
    exterior_edges_hashed = []
    for edge in edges:
      p_1, p_2 = edge.points()
      exterior_edges_hashed.append(unord_hash(p_1.number(), p_2.number()))
    interior_edges_neighbors = defaultdict(list)
    for min_poly in min_polys:
      for p_1, p_2 in min_poly.get_edges():
        if not unord_hash(p_1.number(), p_2.number()) in exterior_edges_hashed:
          interior_edges_neighbors[unord_hash(p_1.number(), p_2.number())].append(min_poly)
          points_neighbors[p_2.number()].append(p_1)
          points_neighbors[p_1.number()].append(p_2)

    new_min_polys = min_polys
    min_polys_created = True
    while min_polys_created:
      min_polys_created = False
      fixed_new_min_polys = new_min_polys
      for min_poly in fixed_new_min_polys:
        p_i, p_j, p_k = min_poly.p1, min_poly.p2, min_poly.p3
        ts = [p_i, p_j, p_k]
        center = (p_i.position() + p_j.position() + p_k.position()) / 3
        e_lens = [(center - p_i.position()).length(), (center - p_j.position()).length(), (center - p_k.position()).length()]
        c_scale = sum(e_lens)
        c_normal = np.zeros(3)

        split = True
        for t in ts:
          c_normal += t.attribValue("N")
          t_scale = 0
          t_neighbors = points_neighbors[t.number()]
          for t_neighbor in t_neighbors:
            t_scale += e_lens_hashed[unord_hash(t.number(), t_neighbor.number())]
          if math.sqrt(2) * (center - t.position()).length() <= min(t_scale, c_scale):
            split = False
        c_normal /= 3

        if split:
          p_c = geo.createPoint()
          p_c.setPosition(center)
          p_c.setAttribValue("N", c_normal)
          new_min_polys.remove(min_poly)
          new_min_polys.extend([VirtualPolygon(p_i, p_c, p_j), VirtualPolygon(p_i, p_c, p_k), VirtualPolygon(p_k, p_c, p_j)])
          for t in ts:
            e_lens_hashed[unord_hash(t.number(), p_c.number())] = e_lens.pop()
            points_neighbors[t.number()].append(p_c)
            others = list(filter(lambda x: x != t, ts))
            interior_edges_neighbors[unord_hash(t.number(), p_c.number())] = [VirtualPolygon(t, p_c, others[0]), VirtualPolygon(t, p_c, others[1])]
          points_neighbors[p_c.number()] = ts
          for t_1, t_2 in min_poly.get_edges():
            if not unord_hash(t_1.number(), t_2.number()) in exterior_edges_hashed:
              interior_edges_neighbors[unord_hash(t_1.number(), t_2.number())].remove(min_poly)
              interior_edges_neighbors[unord_hash(t_1.number(), t_2.number())].append(VirtualPolygon(t_1, p_c, t_2))
          min_polys_created = True
    '''
    3C. Conduct Edge-Swapping
      For two new polygons (p_i, p_j, p_k), (p_i, p_j, p_m) adjacent to interior edge eij,
                                | 1 p_k lies outside circumsphere created with p_i, p_j, p_m
                                |   OR p_m lies outside cirumsphere created with p_i, p_j, p_k
      is_locally_delaunay(ts) = | 0 otherwise
      if two polygons are not locally delaunay, swap edges. Replace eij with ekm.
    '''
    marked_for_update = {}
    marked_for_deletion = []
    for interior_edge in interior_edges_neighbors:
      poly_1, poly_2 = interior_edges_neighbors[interior_edge]
      common_edge = poly_1.get_common_edge(poly_2)
      poly_1_p = list(set([poly_1.p1, poly_1.p2, poly_1.p3]) - set(common_edge))[0]
      poly_2_p = list(set([poly_2.p1, poly_2.p2, poly_2.p3]) - set(common_edge))[0]
      
      poly_1_c_1 = (common_edge[0].position() + poly_1_p.position()) / 2
      poly_1_c_2 = (common_edge[1].position() + poly_1_p.position()) / 2
      poly_1_e_1 = common_edge[0].position() - poly_1_p.position()
      poly_1_e_2 = common_edge[1].position() - poly_1_p.position()
      poly_1_v_1 = hou.Vector3(1, 1, -1 * (poly_1_e_1[0] + poly_1_e_1[1]) / poly_1_e_1[2])
      poly_1_v_2 = hou.Vector3(1, 1, -1 * (poly_1_e_2[0] + poly_1_e_2[1]) / poly_1_e_2[2])

      poly_1_alpha = (poly_1_c_2[2] - poly_1_c_1[2] + (poly_1_c_1[1] - poly_1_c_2[1]) * poly_1_v_2[2]
              / (poly_1_v_1[2] - poly_1_v_2[2]))
      poly_1_circumsphere_c = poly_1_c_1 + poly_1_alpha * poly_1_v_1
      poly_1_circumsphere_r = (poly_1_circumsphere_c - poly_1_p.position()).length()
      
      if (poly_1_circumsphere_c - poly_2_p.position()).length() < poly_1_circumsphere_r:
        new_poly_1 = VirtualPolygon(poly_1_p, common_edge[0], poly_2_p)
        new_poly_2 = VirtualPolygon(poly_1_p, common_edge[1], poly_2_p)
        new_min_polys.remove(poly_1)
        new_min_polys.remove(poly_2)
        new_min_polys.extend([new_poly_1, new_poly_2])
        marked_for_deletion.append(interior_edge)
        marked_for_update[unord_hash(poly_1_p.number(), poly_2_p.number())] = [new_poly_1, new_poly_2]
    for marked in marked_for_deletion:
      del interior_edges_neighbors[marked]
    interior_edges_neighbors.update(marked_for_update)

    for min_poly in new_min_polys:
      new_poly = geo.createPolygon()
      new_poly.addVertex(min_poly.p1)
      new_poly.addVertex(min_poly.p2)
      new_poly.addVertex(min_poly.p3)
  else:
    '''
    4. Fill large hole with advancing front method
    '''
    continue
  
node.bypass(True)