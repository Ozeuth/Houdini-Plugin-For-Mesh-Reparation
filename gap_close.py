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

def unord_hash(a, b):
  if a < b:
    return a * (b - 1) + math.trunc(math.pow(b - a - 2, 2)/ 4)
  elif a > b:
    return (a - 1) * b + math.trunc(math.pow(a - b - 2, 2)/ 4)
  else:
    return a * b + math.trunc(math.pow(abs(a - b) - 1, 2)/ 4)

class Pair():
  def __init__(self, point, elem):
    self.point = point
    self.elem = elem

  def __gt__(self, other):
    return self.point.number() > other.point.number()
  
  def __eq__(self, other):
    return self.point.number() == other.point.number()

  def __repr__(self):
    return (repr((point, elem)))

def elem_is_point(elem):
  return type(elem) == hou.Point

def sort_points(points):
  points_copy = list(points)
  points_copy.sort(key=lambda x: x.number())
  return tuple(points_copy)

def min_dist_and_elem(point, virtual_edges, init=False):
  point_neighbors = []
  min_dist, min_elem = float('inf'), None
  for virtual_edge in virtual_edges:
    '''
            p1
            | proj
      p_inter|------pi
            |
            p2
    '''
    p1, p2 = virtual_edge
    if init:
      if p1 == point: point_neighbors.append(p2)
      if p2 == point: point_neighbors.append(p1)
    if p1 != point and p2 != point:
      e_1i = point.position() - p1.position()
      e_12 = p2.position() - p1.position()
      e_1inter = (e_1i.dot(e_12) / e_12.dot(e_12)) * e_12
      mu = 0
      for i in range(3):
        mu += e_1inter[i] / e_12[i] if e_12[i] != 0 else 0
      mu /= 3
      # NOTE: CHANGE
      mu = -1
      if 0 < mu and mu < 1: # Viable Edge
        proj = e_1i - e_1inter
        proj_dist = proj.length()
        if proj_dist < min_dist:
          min_dist, min_elem = proj_dist, virtual_edge
      else: # Viable Point
        p_1i_dist, p_2i_dist = point.position().distanceTo(p1.position()), point.position().distanceTo(p2.position())
        if p_1i_dist < min_dist:
          min_dist, min_elem = p_1i_dist, p1
        if p_2i_dist < min_dist:
          min_dist, min_elem = p_2i_dist, p2
  return (min_dist, min_elem) if not init else (point_neighbors, min_dist, min_elem)
  
virtual_edges = []
for edge in edges:
  virtual_edges.append(sort_points(edge.points()))

points_neighbors = defaultdict(list)
for point in points:
  point_neighbors, min_dist, min_elem = min_dist_and_elem(point, virtual_edges, init=True)
  points_neighbors[point] = point_neighbors
  '''if not elem_is_point(min_elem):
    edges_to_points[min_elem].append(point)''' # edges_to_points = points for which the closest elem is that edge
  elems_to_points[min_elem].append(point)
  dist_to_pairs.put((min_dist, Pair(point, min_elem)))
  points_to_elem[point] = min_elem

break_point  = 0
marked_for_delete = []
while not dist_to_pairs.empty() and break_point < 4:
  dist, pair = dist_to_pairs.get()
  point, elem = pair.point, pair.elem
  #if point not in marked_for_delete and elem not in marked_for_delete and points_to_elem[point] == elem: 
  if point not in marked_for_delete and points_to_elem[point] == elem: 
    invalid_edge = False
    if type(elem) == hou.Edge: # Point to Edge Contraction
      continue
    if type(elem) == hou.Point or invalid_edge: # Point to Point Contraction
      if sort_points((point, elem)) in virtual_edges: # Edge Contraction
        continue
      else: # Non-Edge Contraction
        print((point, elem))
        elem.setPosition((elem.position() + point.position()) / 2)
        # Update edge set
        old_elem_edges = []
        for p in points_neighbors[elem]:
          old_elem_edges.append(sort_points((elem, p)))

        old_point_edges = []
        new_point_edges = []
        for p in points_neighbors[point]:
          old_point_edges.append(sort_points((point, p)))
          new_point_edges.append(sort_points((elem, p)))
        '''
        Duplicate edges indicate a seam has been closed. These must be:
          1. removed from virtual edges
          2. have the other point removed from points_to_elem
          3. have the other point removed from points_neighbors
          4. Update affected elems of other points
        '''
        point_other = None
        affected_elems = []          
        duplicate_edges = (set(virtual_edges) - set(old_point_edges)).intersection(set(new_point_edges))
        virtual_edges = (set(virtual_edges) - set(old_point_edges)).union(set(new_point_edges)) - duplicate_edges

        if len(duplicate_edges) != 0:
          p1, p2 = list(duplicate_edges)[0]
          point_other = p1 if p2 == elem else p2
          points_to_elem[point_other] = None
          del points_neighbors[point_other]
          points_neighbors[point].remove(point_other)
          points_neighbors[elem].remove(point_other)
          affected_elems = [point_other]

        # Update point neighbors set
        for neighbor in points_neighbors[point]:
          points_neighbors[neighbor].remove(point)
          points_neighbors[neighbor].append(elem)
        points_neighbors[elem] += points_neighbors[point]

        del points_neighbors[point]

        # Update polygons
        for prim in point.prims():
          if prim.type() == hou.primType.Polygon:
            prim.addVertex(elem)
        #geo.deletePoints([point])
        marked_for_delete.append(point)

        affected_elems += old_point_edges + old_elem_edges + [point, elem]
        affected_elems_points = []
        for affected_elem in affected_elems:
          for affected_elem_point in elems_to_points[affected_elem]:
            if affected_elem_point != point_other:
              affected_elems_points.append(affected_elem_point)
          del elems_to_points[affected_elem]
        
        for affected_elems_point in affected_elems_points:
          del points_to_elem[affected_elems_point]
          if affected_elems_point not in marked_for_delete and affected_elems_points != point_other:
            min_dist, min_elem = min_dist_and_elem(affected_elems_point, virtual_edges)
            elems_to_points[min_elem].append(affected_elems_point)
            points_to_elem[affected_elems_point] = min_elem
            dist_to_pairs.put((min_dist, Pair(affected_elems_point, min_elem)))
        break_point += 1
geo.deletePoints(marked_for_delete)