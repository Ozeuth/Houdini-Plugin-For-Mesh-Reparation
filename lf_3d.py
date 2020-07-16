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
    return a * (b - 1) + math.trunc(math.pow(b - a - 2, 2)/ 4)
  elif a > b:
    return (a - 1) * b + math.trunc(math.pow(a - b - 2, 2)/ 4)
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
      # works for triangles and quads
      def __init__(self, ps):
        self.ps = ps
        self.tri = (len(ps) == 3)

      def __eq__(self, other):
        same = True
        for p in self.ps:
          same = same and (p in other.ps)
        return same

      def __str__(self):
        string = []
        for p in self.ps:
          string.append(str(p.number()))
        string.sort()
        return "<" + ', '.join(string) + ">"

      def __repr__(self):
        return str(self)

      def get_edges(self):
        combinations = list(itertools.combinations(self.ps, 2))
        if self.tri:
          return combinations
        combinations_lengths = []
        for combination in combinations:
          combinations_lengths.append((combination[0].position() - combination[1].position()).length())
        return [edge for _, edge in sorted(zip(combinations_lengths, combinations))][2:]

      def get_common_edge(self, other):
        edge_points = []
        for p in self.ps:
          if p in other.ps:
            edge_points.append(p)
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
            curr_polys = [VirtualPolygon([self.points[i], self.points[j], self.points[k]])] + min_polys_left + min_polys_right
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
            for p in min_poly.ps:
              new_poly.addVertex(p)
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
    exterior_edges_neighbors = defaultdict(list)
    for edges_neighbor in list(set(edges_neighbors) - set(edges)):
      p_1, p_2 = edges_neighbor.points()
      points_neighbors[p_1].append(p_2)
      points_neighbors[p_2].append(p_1)
      p1_pos = p_1.position()
      p2_pos = p_2.position()
      e_lens_hashed[unord_hash(p_1.number(), p_2.number())] = (p1_pos - p2_pos).length()
      for prim in edges_neighbor.prims():
        if prim.type() == hou.primType.Polygon:
          exterior_edges_neighbors[unord_hash(p_1.number(), p_2.number())].append(VirtualPolygon(prim.points()))
    
    exterior_edges_hashed = []
    for edge in edges:
      p_1, p_2 = edge.points()
      exterior_edges_hashed.append(unord_hash(p_1.number(), p_2.number()))
    interior_edges_neighbors = defaultdict(list)
    for min_poly in min_polys:
      for p_1, p_2 in min_poly.get_edges():
        if not unord_hash(p_1.number(), p_2.number()) in exterior_edges_hashed:
          interior_edges_neighbors[unord_hash(p_1.number(), p_2.number())].append(min_poly)
          points_neighbors[p_2].append(p_1)
          points_neighbors[p_1].append(p_2)

    new_min_polys = min_polys
    min_polys_created = True
    while min_polys_created:
      min_polys_created = False
      fixed_new_min_polys = new_min_polys
      for min_poly in fixed_new_min_polys:
        p_i, p_j, p_k = min_poly.ps
        ts = [p_i, p_j, p_k]
        center = (p_i.position() + p_j.position() + p_k.position()) / 3
        e_lens = [(center - p_i.position()).length(), (center - p_j.position()).length(), (center - p_k.position()).length()]
        c_scale = sum(e_lens)
        c_normal = np.zeros(3)

        split = True
        for t in ts:
          c_normal += t.attribValue("N")
          t_scale = 0
          t_neighbors = points_neighbors[t]
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
          new_min_polys.extend([VirtualPolygon([p_i, p_c, p_j]), VirtualPolygon([p_i, p_c, p_k]), VirtualPolygon([p_k, p_c, p_j])])
          for t in ts:
            e_lens_hashed[unord_hash(t.number(), p_c.number())] = e_lens.pop()
            points_neighbors[t].append(p_c)
            others = list(filter(lambda x: x != t, ts))
            interior_edges_neighbors[unord_hash(t.number(), p_c.number())] = [VirtualPolygon([t, p_c, others[0]]), VirtualPolygon([t, p_c, others[1]])]
          points_neighbors[p_c] = ts
          for t_1, t_2 in min_poly.get_edges():
            if not unord_hash(t_1.number(), t_2.number()) in exterior_edges_hashed:
              interior_edges_neighbors[unord_hash(t_1.number(), t_2.number())].remove(min_poly)
              interior_edges_neighbors[unord_hash(t_1.number(), t_2.number())].append(VirtualPolygon([t_1, p_c, t_2]))
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
      poly_1_p = list(set(poly_1.ps) - set(common_edge))[0]
      poly_2_p = list(set(poly_2.ps) - set(common_edge))[0]
      
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
        new_poly_1 = VirtualPolygon([poly_1_p, common_edge[0], poly_2_p])
        new_poly_2 = VirtualPolygon([poly_1_p, common_edge[1], poly_2_p])
        new_min_polys.remove(poly_1)
        new_min_polys.remove(poly_2)
        new_min_polys.extend([new_poly_1, new_poly_2])

        marked_for_deletion.append(interior_edge)
        marked_for_update[unord_hash(poly_1_p.number(), poly_2_p.number())] = [new_poly_1, new_poly_2]
        # 4 interior_edges_neighbors could potentially need updating
        e1, e2, e3, e4 = (unord_hash(poly_1_p.number(), common_edge[0].number()),
                          unord_hash(poly_1_p.number(), common_edge[1].number()),
                          unord_hash(poly_2_p.number(), common_edge[0].number()),
                          unord_hash(poly_2_p.number(), common_edge[1].number()))
        old_ps = [poly_1, poly_1, poly_2, poly_2]
        new_ps = [new_poly_1, new_poly_2, new_poly_1, new_poly_2]
        for e_i, e in enumerate([e1, e2, e3, e4]):
          if ((e in interior_edges_neighbors and not e in marked_for_deletion) or (e in marked_for_update)):
            update = interior_edges_neighbors if (e in interior_edges_neighbors and not e in marked_for_deletion) else marked_for_update
            update[e].remove(old_ps[e_i])
            update[e].append(new_ps[e_i])
        
        points_neighbors[common_edge[0]].remove(common_edge[1])
        points_neighbors[common_edge[1]].remove(common_edge[0])
        points_neighbors[poly_1_p].append(poly_2_p)
        points_neighbors[poly_2_p].append(poly_1_p)
    for marked in marked_for_deletion:
      del interior_edges_neighbors[marked]
    interior_edges_neighbors.update(marked_for_update)

    for min_poly in new_min_polys:
      new_poly = geo.createPolygon()
      for p in min_poly.ps:
        new_poly.addVertex(p)
    '''
    3D. Conduct Patch Fairing.
    Compute the Laplace Beltrami Matrix,
         pi  pj  pk
    pi | Li  Wij Wik |
    pj | Wji Lj  Wjk |
    pk | Wki Wkj Lk  |
    where
      Li = Wij + Wik + ...
      Wij = 0.5*(cot(alpha) + cot(beta))
      where for two polygons (p_i, p_j, p_k), (p_i, p_j, p_m) adjacent to interior edge eij,
        alpha = angle_ikj
        beta = angle_imj
    '''
    laplace_beltrami = np.zeros((len(points_neighbors), len(points_neighbors)))
    ref_keys = list(points_neighbors.keys())
    for p_i in points_neighbors:
      ref_i = ref_keys.index(p_i)
      for p_j in points_neighbors[p_i]:
        ref_j = ref_keys.index(p_j)
        if unord_hash(p_i.number(), p_j.number()) in interior_edges_neighbors:
          poly_1, poly_2 = interior_edges_neighbors[unord_hash(p_i.number(), p_j.number())]
        elif unord_hash(p_i.number(), p_j.number()) in exterior_edges_neighbors:
          poly_1, poly_2 = exterior_edges_neighbors[unord_hash(p_i.number(), p_j.number())]
        laplace_beltrami[ref_i, ref_j] = 0
      laplace_beltrami[ref_i, ref_i] = sum(laplace_beltrami[ref_i])

  else:
    '''
    4. Fill large hole with advancing front method
    '''
    continue
  
node.bypass(True)