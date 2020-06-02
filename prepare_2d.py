import PIL
from PIL import Image, ImageDraw

node = hou.pwd()
geo = node.geometry()
boundary_node = node.inputs()[1]
boundaries = boundary_node.geometry().pointGroups()
edge_boundary_node = node.inputs()[2]
edge_boundaries = edge_boundary_node.geometry().edgeGroups()
'''
 7. Map 3D Context Region -> 2D Render
'''
pix = geo.findPointAttrib("pix_new")
path_name = hou.hipFile.name().split(".")[0]
if (pix):
  print("FOUND PIX_NEW")
  for i in range(1, len(boundaries)):
    boundary = boundaries[i]
    edge_boundary = edge_boundaries[i-1]
    points = boundary.points()
    edges = edge_boundary.edges()

    pix_pos = []
    for point in points:
      pix_attrib = point.attribValue(pix)
      pix_point = (pix_attrib[(i-1) * 3], pix_attrib[(i-1) * 3 + 1], pix_attrib[(i-1) * 3 + 2])
      pix_pos.append((point.number(), pix_point))
    pix_pos = dict(pix_pos)

    image = Image.open(path_name + "/opening_" + str(i) + ".png")
    draw = ImageDraw.Draw(image)
    for edge in edges:
      edge_points = edge.points()
      edge_pixel_0 = pix_pos.get(edge_points[0].number())
      edge_pixel_1 = pix_pos.get(edge_points[1].number())
      draw.line([(int(edge_pixel_0[0]), int(edge_pixel_0[1])), (int(edge_pixel_1[0]), int(edge_pixel_1[1]))], "red")
      print(str(i) + " drawing to " + str([(int(edge_pixel_0[0]), int(edge_pixel_0[1])), (int(edge_pixel_1[0]), int(edge_pixel_1[1]))]))
    image.save(path_name + "/opening_" + str(i) + ".png")
node.bypass(True)