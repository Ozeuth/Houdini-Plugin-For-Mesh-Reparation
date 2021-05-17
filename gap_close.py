import hou
from collections import defaultdict
from queue import PriorityQueue

node = hou.pwd()
geo = node.geometry()

points = geo.pointGroups()[0].points()
edges = geo.edgeGroups()[0].edges()

dist_to_pairs = PriorityQueue()
shared_groups, shared_groups_to_edges = [], defaultdict(list)
points_to_elem, elems_to_points = {}, defaultdict(list)

class Pair():
  def __init__(self, point, elem, inter):
    self.point = point
    self.elem = elem
    self.inter = inter

  def __gt__(self, other):
    return self.point.number() > other.point.number()
  
  def __eq__(self, other):
    return self.point.number() == other.point.number()

  def __repr__(self):
    return (repr((point, elem)))

def sort_points(points):
  points_copy = list(points)
  points_copy.sort(key=lambda x: x.number())
  return tuple(points_copy)

def get_clockwise_neighbors(p, p_a_b):
  # p_1 = left of p, p_2 = right of p
  p_a, p_b = p_a_b
  p_1, p_2 = None, None
  for prim in p.prims():
    if prim.type() == hou.primType.Polygon:
      ps = []
      for v in prim.vertices():
        ps.append(v.point())
        
      if p_a in ps or p_b in ps:
        p_i = ps.index(p)
        ps =  ps[p_i:] + ps[:p_i]
        assert(ps[0] == p)
        p_2 = ps[1] if (ps[1] == p_a or ps[1] == p_b) else p_2
        p_1 = ps[len(ps) - 1] if (ps[len(ps) - 1] == p_a or ps[len(ps) - 1] == p_b) else p_1
  assert(p_1 != None and p_2 != None)
  return p_1, p_2

def min_dist_and_elem(point, neighbors_edges_pairs, epsilon=0.1):
  min_dist, min_elem, min_inter = float('inf'), None, None
  virtual_edges = []
  for points_neighbor_, virtual_edges_ in neighbors_edges_pairs:
    if point in points_neighbor_:
      virtual_edges += virtual_edges_

  for virtual_edge in virtual_edges:
    '''
          p1
           | proj
    p_inter|_____pi
           |
          p2
    '''
    p1, p2 = virtual_edge
    if p1 != point and p2 != point:
      e_1i = point.position() - p1.position()
      e_12 = p2.position() - p1.position()
      e_1inter = (e_1i.dot(e_12) / e_12.dot(e_12)) * e_12
      mu = 0
      for i in range(3):
        mu += e_1inter[i] / e_12[i] if e_12[i] != 0 else 0
      mu /= 3
      is_self_merge = False
      for prim in point.prims():
        if prim.type() == hou.primType.Polygon:
          if p1 in prim.points() and p2 in prim.points():
            is_self_merge = True
      if epsilon < mu and mu < (1 - epsilon) and not is_self_merge: # Viable Edge
        proj = e_1i - e_1inter
        proj_dist = proj.length()
        p_inter = p1.position() + e_1inter
        if proj_dist < min_dist:
          min_dist, min_elem, min_inter = proj_dist, virtual_edge, p_inter
      else: # Viable Point
        p_1i_dist, p_2i_dist = point.position().distanceTo(p1.position()), point.position().distanceTo(p2.position())
        if p_1i_dist < min_dist:
          min_dist, min_elem = p_1i_dist, p1
        if p_2i_dist < min_dist:
          min_dist, min_elem = p_2i_dist, p2
  return (min_dist, min_elem, min_inter)

points_neighbors = defaultdict(list)
virtual_edges = []
for edge in edges:
  p1, p2 = edge.points()
  points_neighbors[p1].append(p2)
  points_neighbors[p2].append(p1)
  if len(points_neighbors[p1]) == 2:
    points_neighbors[p1] = list(get_clockwise_neighbors(p1, tuple(points_neighbors[p1])))
  if len(points_neighbors[p2]) == 2:
    points_neighbors[p2] = list(get_clockwise_neighbors(p2, tuple(points_neighbors[p2])))
  virtual_edges.append(sort_points(edge.points()))

neighbors_edges_pairs = [(points_neighbors, virtual_edges)]

for point in points:
  min_dist, min_elem, min_inter = min_dist_and_elem(point, neighbors_edges_pairs)
  if min_elem != None:
    elems_to_points[min_elem].append(point)
    dist_to_pairs.put((min_dist, Pair(point, min_elem, min_inter)))
    points_to_elem[point] = min_elem

print("START")
break_point  = 0
marked_for_delete_points, marked_for_delete_polys = [], []
while not dist_to_pairs.empty() and break_point < 50:
  dist, pair = dist_to_pairs.get()
  point, elem, inter = pair.point, pair.elem, pair.inter
  if point not in marked_for_delete_points and points_to_elem[point] == elem:
    point_other, point_others = None, []
    if type(elem) != hou.Point:
      # Point to Edge Contraction
      print("P-E: " + str((point.number(), (elem[0].number(), elem[1].number()))))
      for points_neighbor_, virtual_edges_ in neighbors_edges_pairs:
        if point in points_neighbor_ and elem[0] in points_neighbor_ and elem[1] in points_neighbor_:
          points_neighbors = points_neighbor_
          virtual_edges = virtual_edges_
      neighbors_edges_pairs.remove((points_neighbors, virtual_edges))
      '''
          Typical Point-Edge Contraction
                elem
          er____________el       er____     _____el
                             =>        \   /
                                     point
          pl___point____pr       pl____/   \_____pr

          Edge case Point-Edge Contraction
              
              el                              el
             /                               /
            / elem           =>  other----point
           /                                  \ 
          other___point___pr                   pr

      '''
      point.setPosition((inter + point.position()) / 2)
      elem_l, elem_r = elem[0] if points_neighbors[elem[0]][1] == elem[1] else elem[1], elem[0] if points_neighbors[elem[0]][0] == elem[1] else elem[1]

      elem_poly = set(elem_l.prims()).intersection(set(elem_r.prims())).pop()
      marked_for_delete_polys.append(elem_poly)
      poly_points = elem_poly.points()
      poly_1, poly_2 = poly_points.copy(), poly_points.copy()
      elem_r_index, elem_l_index = poly_1.index(elem_r), poly_2.index(elem_l)
      poly_1[elem_r_index], poly_2[elem_l_index] = point, point
      geo.createPolygons((tuple(poly_1), tuple(poly_2)))

      old_elem_edges = ([sort_points((elem_l, p)) for p in points_neighbors[elem_l] if p != elem_r] 
                      + [sort_points((elem_r, p)) for p in points_neighbors[elem_r] if p != elem_l])
      old_point_edges = [sort_points((point, p)) for p in points_neighbors[point]]
      duplicate_edges = set(old_point_edges).intersection(set(old_elem_edges))

      '''virtual_edges = ((set(virtual_edges)).union(set([sort_points((elem_l, point)), sort_points((elem_r, point))])) 
                      - duplicate_edges - set([elem]))'''
      affected_elems = [point, elem] + old_point_edges

      points_neighbors_l, points_neighbors_r = defaultdict(list), defaultdict(list)
      virtual_edges_l, virtual_edges_r = [], []
      point_l, point_r = points_neighbors[point]
      points_to_elem[point] = None

      points_neighbors[point] = [point_l, elem_r]
      points_neighbors[elem_r][0] = point
      curr, is_loop = point, False
      while not is_loop:
        points_neighbors_l[curr] = points_neighbors[curr]
        virtual_edges_l.append(sort_points((curr, points_neighbors[curr][0])))
        curr = points_neighbors[curr][0]
        if curr == point:
          is_loop = True
      
      points_neighbors[point] = [elem_l, point_r]
      points_neighbors[elem_l][1] = point
      curr, is_loop = point, False
      while not is_loop:
        points_neighbors_r[curr] = points_neighbors[curr]
        virtual_edges_r.append(sort_points((curr, points_neighbors[curr][0])))
        curr = points_neighbors[curr][0]
        if curr == point:
          is_loop = True

      marked_for_delete_groups = []
      new_neighbors_edges_pairs = [(points_neighbors_l, virtual_edges_l), (points_neighbors_r, virtual_edges_r)]
      while len(duplicate_edges) > 0:
        pa, pb = duplicate_edges.pop()
        point_other = pa if pb == point else pb
        points_to_elem[point_other] = None
        if point_other in points_neighbors_l: marked_for_delete_groups.append((points_neighbors_l, virtual_edges_l))
        if point_other in points_neighbors_r: marked_for_delete_groups.append((points_neighbors_r, virtual_edges_r))
        affected_elems += [point_other]
        point_others.append(point_other)

      for marked_for_delete_group in marked_for_delete_groups:
        new_neighbors_edges_pairs.remove(marked_for_delete_group)
        del marked_for_delete_group
      if not new_neighbors_edges_pairs:
        points_to_elem[point] = None
      neighbors_edges_pairs += new_neighbors_edges_pairs

    if type(elem) == hou.Point:
      # Point to Point Contraction
      for points_neighbor_, virtual_edges_ in neighbors_edges_pairs:
        if point in points_neighbor_ and elem in points_neighbor_:
          points_neighbors = points_neighbor_
          virtual_edges = virtual_edges_
      neighbors_edges_pairs.remove((points_neighbors, virtual_edges))
      is_ee_contraction = sort_points((point, elem)) in virtual_edges
      print("P-P " + ("EE" if is_ee_contraction else "NE") + ":" + str((point.number(), elem.number())))
      ''' 
          Typical Point-Point Contraction
          p1____elem____p2     p1 _____    _____p2
                            =>        elem               Non-Edge Contraction
          p3____point____p4     p3____/    \____p4
      OR
          p1___point___elem___p2  =>   p1____elem____p2  Edge Contraction

          Edge case Point-Point Contraction
                / elem        
               /         =>   other------point           Non-Edge Contraction
              /            
          other____point
      OR
                 elem
                /   |
               /    |    =>   other------point           Edge Contraction
              /     |
          other____point
      '''
      elem.setPosition((elem.position() + point.position()) / 2)
      for prim in point.prims():
        if prim.type() == hou.primType.Polygon:
          if elem not in prim.points():
            poly_points = prim.points()
            point_index = poly_points.index(point)
            poly_points[point_index] = elem
            new_poly = geo.createPolygon()
            for poly_point in poly_points:
              new_poly.addVertex(poly_point)
          marked_for_delete_polys.append(prim)
      marked_for_delete_points.append(point)

      old_elem_edges = [sort_points((elem, p)) for p in points_neighbors[elem]]
      old_point_edges = [sort_points((point, p)) for p in points_neighbors[point]]
      new_point_edges = [sort_points((elem, p)) for p in points_neighbors[point] if p != elem]
      duplicate_edges = set(new_point_edges).intersection(set(old_elem_edges))

      affected_elems = old_point_edges + old_elem_edges + [point, elem]

      point_l, point_r = points_neighbors[point]
      elem_l, elem_r = points_neighbors[elem]
      points_to_elem[point] = None
      if is_ee_contraction:
        # Edge Contraction
        virtual_edges = ((set(virtual_edges) - set(old_point_edges)).union(set(new_point_edges))
                - duplicate_edges - set([sort_points((point, elem))]))
        temp_list = []
        del points_neighbors[point]
        points_neighbors[point_l][1] = elem if point_l != elem else point_r
        points_neighbors[point_r][0] = elem if point_r != elem else point_l

        while len(duplicate_edges) > 0:
          p1, p2 = duplicate_edges.pop()
          point_other = p1 if p2 == elem else p2
          points_to_elem[p1] = None
          del points_neighbors[p1]
          points_to_elem[p2] = None
          del points_neighbors[p2]
          affected_elems += [point_other]
          point_others.append(point_other)
        neighbors_edges_pairs.append((points_neighbors, virtual_edges))

      else:
        # Non-Edge Contraction
        points_neighbors_l, points_neighbors_r = defaultdict(list), defaultdict(list)
        virtual_edges_l, virtual_edges_r = [], []
        points_neighbors[elem] = [elem_l, point_r]
        points_neighbors[point_r][0] = elem
        curr, is_loop = elem, False
        while not is_loop:
          points_neighbors_l[curr] = points_neighbors[curr]
          virtual_edges_l.append(sort_points((curr, points_neighbors[curr][0])))
          curr = points_neighbors[curr][0]
          if curr == elem:
            is_loop = True

        points_neighbors[elem] = [point_l, elem_r]
        points_neighbors[point_l][1] = elem
        curr, is_loop = elem, False
        while not is_loop:
          points_neighbors_r[curr] = points_neighbors[curr]
          virtual_edges_r.append(sort_points((curr, points_neighbors[curr][0])))
          curr = points_neighbors[curr][0]
          if curr == elem:
            is_loop = True

        marked_for_delete_groups = []
        new_neighbors_edges_pairs = [(points_neighbors_l, virtual_edges_l), (points_neighbors_r, virtual_edges_r)]
        while len(duplicate_edges) > 0:
          pa, pb = duplicate_edges.pop()
          point_other = pa if pb == elem else pb
          points_to_elem[point_other] = None
          if point_other in points_neighbors_l: marked_for_delete_groups.append((points_neighbors_l, virtual_edges_l))
          if point_other in points_neighbors_r: marked_for_delete_groups.append((points_neighbors_r, virtual_edges_r))
          affected_elems += [point_other]
          point_others.append(point_other)

        for marked_for_delete_group in marked_for_delete_groups:
          new_neighbors_edges_pairs.remove(marked_for_delete_group)
          del marked_for_delete_group
        if not new_neighbors_edges_pairs:
          points_to_elem[elem] = None
        neighbors_edges_pairs += new_neighbors_edges_pairs
    
    affected_elems = list(set(affected_elems))
    affected_elems_points = []
    for affected_elem in affected_elems:
      for affected_elem_point in elems_to_points[affected_elem]:
        if affected_elem_point not in point_others:
          affected_elems_points.append(affected_elem_point)
      del elems_to_points[affected_elem]
    for affected_elems_point in affected_elems_points:
      points_to_elem[affected_elems_point] = None
      if affected_elems_point not in marked_for_delete_points and affected_elems_point not in point_others:
        min_dist, min_elem, min_inter = min_dist_and_elem(affected_elems_point, neighbors_edges_pairs)
        if min_elem != None:
          elems_to_points[min_elem].append(affected_elems_point)
          points_to_elem[affected_elems_point] = min_elem
          dist_to_pairs.put((min_dist, Pair(affected_elems_point, min_elem, min_inter)))

    '''for points_neighbor_, virtual_edges_ in neighbors_edges_pairs:
      temp_list = []
      print("PN, VE")
      for key, value in points_neighbor_.items():
        vs = []
        for v in value:
          vs.append(v.number())
        temp = [key.number(), vs]
        temp_list.append(temp)
      print(temp_list)
      temp_list = []
      for virtual_edge in virtual_edges_:
        temp_list.append((virtual_edge[0].number(), virtual_edge[1].number()))
      print(temp_list)
        
    temp_list = []
    print("points_to_elem")
    for key, value in points_to_elem.items():
      if type(value) == hou.Point:
        elem = value.number()
      elif value == None:
        elem = None
      else:
        elem = (value[0].number(), value[1].number())
      temp_list.append([key.number(), elem])
    print(temp_list)'''

geo.deletePrims(marked_for_delete_polys, keep_points=True)
geo.deletePoints(marked_for_delete_points)