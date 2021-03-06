import hou
import glob
import math
import numpy as np
import numpy.random as random
import os
from PIL import Image, ImageDraw
from collections import defaultdict

node = hou.pwd()
geo = node.geometry()
inputs = node.inputs()
is_DA = "mesh_repairer" in node.parent().name()

bbox_node = hou.node(hou.parent().path() + "/oz_bbox")
uv_bbox_node = hou.node(hou.parent().path() + "/uv_viewer_bbox")
uv_node = hou.node(hou.parent().path() + "/uv_viewer")

scene = hou.ui.curDesktop().paneTabOfType(hou.paneTabType.SceneViewer)
viewport = scene.curViewport()
frame = hou.frame()
path_name = hou.hipFile.name().split(".")[0]
if not os.path.exists(path_name):
  os.makedirs(path_name)
boundaries = geo.pointGroups()
edge_boundaries = geo.edgeGroups()
try:
  for excess in range(len(boundaries-1), len(glob.glob(path_name + "/*.png"))):
    os.remove(path_name + "/" + str(excess) + "_opening.png")
except:
  for f in glob.glob(path_name + "/*.png"):
    os.remove(f)

hou.session.reset_camera_info()

def clockwise_neighbors(p, points_neighbors):
  # p_1 = left of p, p_2 = right of p
  p_a, p_b = points_neighbors[p]
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

# 1-3: Choose 3D Context Region
i = 0
resolutions_x = []
resolutions_y = []

for boundary, edge_boundary in zip(boundaries, edge_boundaries):
  if i == 0:
    i += 1
    continue
  '''
  1. Compute a Projection Plane via Least Squares
  Plane eqn: ax + by + c = z
       A        x   =   b
  | x0 y0 1 |         | z0 |
  | x1 y1 1 | | a |   | z1 |  => | a |
  | ....... | | b | = | .. |     | b | = (A^T*A)^-1*A^T*b
  | xn yn 1 | | c |   | zn |     | c |
  '''
  A = []
  b = []
  boundary_center = np.array([0, 0, 0])
  boundary_normal = hou.Vector3((0, 0, 0))
  points = boundary.points()
  edges = edge_boundary.edges()
  points_neighbors = defaultdict(list)
  for edge in edges:
    p_1, p_2 = edge.points()
    points_neighbors[p_1].append(p_2)
    points_neighbors[p_2].append(p_1)
  for point in points:
    point_pos = point.position()
    A.append([point_pos[0], point_pos[1], 1])
    b.append(point_pos[2])
    boundary_center = boundary_center + np.array(point_pos)

    p_l, p_r = clockwise_neighbors(point, points_neighbors)
    e_dir1 = hou.Vector3(p_l.position() - point.position()).normalized()
    e_dir2 = hou.Vector3(p_r.position() - point.position()).normalized()
    point_normal = e_dir2.cross(e_dir1).normalized()
    boundary_normal += point_normal
  boundary_normal /= len(points)
  A = np.matrix(A)
  b = np.matrix(b).T
  boundary_center /= len(points)
  fit_fail = False
  det = np.linalg.det(A.T * A)
  if (det > 0.000001):
    fit = (A.T * A).I * A.T * b
  else:
    # This plane is almost parallel to xz plane
    fit = np.linalg.pinv(A.T * A) * A.T * b
    fit_fail = True
  a = fit.item(0)
  b = fit.item(1)
  c = fit.item(2)
  errors = b - A * fit
  residual = np.linalg.norm(errors)
  '''
  2. Camera is fit to Projection Plane via translation + rotation
  '''
  if (hou.node('/obj/oz_camera_' + str(i))):
    camera = hou.node('/obj/oz_camera_' + str(i))
  else:
    camera = hou.node('/obj').createNode('cam', 'oz_camera_' + str(i))
  if (not fit_fail):
    plane_normal = np.array([a, b, -1]) / math.sqrt(math.pow(a, 2) + math.pow(b, 2) + 1)
    plane_dist = c / math.sqrt(math.pow(a, 2) + math.pow(b, 2) + 1) # TODO: p > 0 or p < 0 half-space origin test
  else:
    plane_normal = np.array([0.001, 1, 0.001])* (1 if (a >= 0) else -1)
    plane_dist = 0.001
  if boundary_normal.angleTo(hou.Vector3(plane_normal)) > 90:
    plane_normal = -1 * plane_normal

  translation = hou.Matrix4((1, 0, 0, boundary_center[0],
                              0, 1, 0, boundary_center[1],
                              0, 0, 1, boundary_center[2], 
                              0, 0, 0, 1)).transposed()
  v = math.sqrt(math.pow(plane_normal[0], 2) + math.pow(plane_normal[2], 2))
  rotation_y = hou.Matrix4((plane_normal[2]/v, 0, -1 * plane_normal[0]/v, 0,
                            0, 1, 0, 0,
                            plane_normal[0]/v, 0, plane_normal[2]/v, 0, 
                            0, 0, 0, 1))
  d = math.sqrt(math.pow(plane_normal[0], 2) + math.pow(plane_normal[1], 2) + math.pow(plane_normal[2], 2))
  rotation_x = hou.Matrix4((1, 0, 0, 0,
                          0, v/d,  -1 * plane_normal[1]/d, 0,
                          0, plane_normal[1]/d, v/d, 0, 
                          0, 0, 0, 1))
  camera.setWorldTransform(rotation_x * rotation_y * translation)
  resolution = (1280, 720)
  camera.parm('resx').set(resolution[0])
  camera.parm('resy').set(resolution[1])
  resolutions_x.append(resolution[0])
  resolutions_y.append(resolution[1])
  '''
  3. Camera is zoomed out until entire mesh is visible.
    A point is viewable if it is given a valid UV coordinate
    (uv_x, uv_y where 0 <= uv_x, uv_y <= 1) when the mesh is
    unwrapped using camera perspective.
  '''
  uv_bbox_node.parm("campath").set(camera.path())
  uv_node.parm("campath").set(camera.path())

  zoom_out = 0
  zoom_step = 0.1
  uv_bbox_node.setInput(0, bbox_node)
  visible_points = 0
  max_visible_points = len(bbox_node.geometry().points())
  uv_points = uv_bbox_node.geometry().points()
  while (visible_points != max_visible_points):
    visible_points = 0
    for uv_point in uv_points:
      uv_coord = uv_point.attribValue("uv")
      if (uv_coord[0] >= 0 and uv_coord[0] <= 1 and uv_coord[1] >= 0 and uv_coord[1] <= 1 and not all(v == 0 for v in uv_coord)):
        visible_points += 1
    if (visible_points != uv_points):
      zoom_out += zoom_step
      camera_normal = plane_normal * zoom_out
      new_translation = hou.Matrix4((1, 0, 0, boundary_center[0] + camera_normal[0],
                          0, 1, 0, boundary_center[1] + camera_normal[1],
                          0, 0, 1, boundary_center[2] + camera_normal[2], 
                          0, 0, 0, 1)).transposed()
      hou.session.reset_camera(camera)
      camera.setWorldTransform(rotation_x * rotation_y * new_translation)

  hou.session.cameras_info["centers"].append(boundary_center)
  hou.session.cameras_info["rotationsx"].append(rotation_x)
  hou.session.cameras_info["rotationsy"].append(rotation_y)
  hou.session.cameras_info["normals"].append(plane_normal)
  hou.session.cameras_info["zooms"].append(zoom_out)

  if (hou.node("/out/oz_render_" + str(i))):
    render = hou.node("/out/oz_render_" + str(i))
  else:
    render = hou.node("/out").createNode("ifd", "oz_render_" + str(i))
  image_path = path_name + "/" + str(i) + "_opening.png"
  if (not os.path.isfile(image_path)):
    temp_img = Image.new('RGB', (60,30), color=(0, 0, 0))
    draw = ImageDraw.Draw(temp_img)
    draw.text((10, 10), "Temp Img", fill=(255, 255, 255))
    temp_img.save(image_path)
    temp_img.close()
  render.parm("camera").set(camera.path())
  render.parm("vm_picture").set(image_path)
  i += 1

if not(geo.findGlobalAttrib("resolutionsx") or geo.findGlobalAttrib("resolutionsy")):
  geo.addAttrib(hou.attribType.Global, "resolutionsx", resolutions_x)
  geo.addAttrib(hou.attribType.Global, "resolutionsy", resolutions_y)
else:
  geo.setGlobalAttribValue("resolutionsx", resolutions_x)
  geo.setGlobalAttribValue("resolutionsy", resolutions_y)
node.bypass(True)