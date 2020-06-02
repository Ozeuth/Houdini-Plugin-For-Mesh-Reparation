import glob
import math
import numpy as np
import numpy.random as random
import os
from PIL import Image, ImageDraw

def reset_camera(camera):
  camera.parmTuple('t').set((0, 0, 0))
  camera.parmTuple('r').set((0, 0, 0))

node = hou.pwd()
geo = node.geometry()
inputs = node.inputs()
is_DA = "mesh_repairer" in node.parent().name()

if (is_DA):
  input_node = node.parent().indirectInputs()[0]
else:
  assert (inputs[0].type().name() == "file"), ("ERROR: Input must be chosen geometry")
  input_node = inputs[0]
bbox_node = hou.node(hou.parent().path() + "/oz_bbox")
uv_bbox_node = hou.node(hou.parent().path() + "/uv_viewer_bbox")
uv_node = hou.node(hou.parent().path() + "/uv_viewer")

scene = hou.ui.curDesktop().paneTabOfType(hou.paneTabType.SceneViewer)
viewport = scene.curViewport()
frame = hou.frame()
path_name = hou.hipFile.name().split(".")[0]
if not os.path.exists(path_name):
  os.makedirs(path_name)
boundaries = inputs[1].geometry().pointGroups()
try:
  for excess in range(len(boundaries-1), len(glob.glob(path_name + "/*.png"))):
    os.remove(path_name + "/opening_" + str(excess) + ".png")
except:
  for f in glob.glob(path_name + "/*.png"):
    os.remove(f)

# 1-3: Choose 3D Context Region
i = 0
resolutions_x = []
resolutions_y = []
for boundary in boundaries:
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
  points = boundary.points()
  for point in points:
    point_pos = point.position()
    A.append([point_pos[0], point_pos[1], 1])
    b.append(point_pos[2])
    boundary_center = boundary_center + np.array(point_pos)
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
  3. Camera is zoomed out until it views all boundary point
    A point is viewable if it is given a valid UV coordinate
    (uv_x, uv_y where 0 <= uv_x, uv_y <= 1) when the mesh is
    unwrapped using camera perspective
    While not all boundary points are viewable, We zoom the camera
    to a value within the zoom range (converging on ideal values),
    re-unwrap and check again
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
      reset_camera(camera)
      camera.setWorldTransform(rotation_x * rotation_y * new_translation)

  zoom_range_max = zoom_out
  best_zoom_out = zoom_out
  best_visible_points = 0
  max_visible_points = len(points)
  tries = 0
  sample_size = 10
  max_tries = 10
  better_zoom_findable = True
  while ((best_visible_points != max_visible_points or better_zoom_findable) and tries < max_tries):
    visible_points_samples = []
    zoom_out_samples = []
    for sample in range(sample_size):
      if sample == 0:
        zoom_out = best_zoom_out
      else:
        zoom_out = random.uniform(0, zoom_range_max)
      visible_points = 0
      camera_normal = plane_normal * zoom_out
      new_translation = hou.Matrix4((1, 0, 0, boundary_center[0] + camera_normal[0],
                            0, 1, 0, boundary_center[1] + camera_normal[1],
                            0, 0, 1, boundary_center[2] + camera_normal[2], 
                            0, 0, 0, 1)).transposed()
      reset_camera(camera)
      camera.setWorldTransform(rotation_x * rotation_y * new_translation)
      for point in points:
        uv_coord = point.attribValue("uv")
        if (uv_coord[0] >= 0 and uv_coord[0] <= 1 and uv_coord[1] >= 0 and uv_coord[1] <= 1 and not all(v == 0 for v in uv_coord)):
          visible_points+= 1
      visible_points_samples.append(visible_points)
      zoom_out_samples.append(zoom_out)

    best_visible_points = max(visible_points_samples)
    viable_zoom_outs = []
    for sample in range(sample_size):
      if (visible_points_samples[sample] == best_visible_points):
        viable_zoom_outs.append(zoom_out_samples[sample])
    best_zoom_out = min(viable_zoom_outs)

    if (best_visible_points == max_visible_points):
      if (zoom_range_max == best_zoom_out):
        better_zoom_findable = False
      zoom_range_max = best_zoom_out
    tries += 1
  if (best_visible_points != max_visible_points): # Not all points could be fit to the camera
    i += 1
    continue
  camera_normal = plane_normal * best_zoom_out
  new_translation = hou.Matrix4((1, 0, 0, boundary_center[0] + camera_normal[0],
                        0, 1, 0, boundary_center[1] + camera_normal[1],
                        0, 0, 1, boundary_center[2] + camera_normal[2], 
                        0, 0, 0, 1)).transposed()
  reset_camera(camera)
  camera.setWorldTransform(rotation_x * rotation_y * new_translation) 
  if (hou.node("/out/oz_render_" + str(i))):
    render = hou.node("/out/oz_render_" + str(i))
  else:
    render = hou.node("/out").createNode("ifd", "oz_render_" + str(i))
  image_path = path_name + "/opening_" + str(i) + ".png"
  if (not os.path.isfile(image_path)):
    temp_img = Image.new('RGB', (60,30), color=(0, 0, 0))
    draw = ImageDraw.Draw(temp_img)
    draw.text((10, 10), "Temp Img", fill=(255, 255, 255))
    temp_img.save(image_path)
  render.parm("camera").set(camera.path())
  render.parm("vm_picture").set(image_path)
  i += 1

if (not(geo.findGlobalAttrib("resolutionsx") or geo.findGlobalAttrib("resolutionsy"))):
  geo.addAttrib(hou.attribType.Global, "resolutionsx", resolutions_x)
  geo.addAttrib(hou.attribType.Global, "resolutionsy", resolutions_y)
else:
  geo.setGlobalAttribValue("resolutionsx", resolutions_x)
  geo.setGlobalAttribValue("resolutionsy", resolutions_y)
node.bypass(True)