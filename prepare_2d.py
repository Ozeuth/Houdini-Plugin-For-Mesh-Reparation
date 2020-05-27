import PIL
from PIL import Image, ImageDraw

node = hou.pwd()
geo = node.geometry()
boundaries = node.inputs()[1].geometry().pointGroups()

pix = geo.findPointAttrib("pix")
path_name = hou.hipFile.name().split(".")[0]
if (pix):
  i = 0
  for boundary in boundaries:
    if i == 0:
      i += 1
      continue
    pix_pos = []
    points = boundaries[i].points()
    for point in points:
      pix_attrib = point.attribValue(pix)
      pix_point = (pix_attrib[(i-1) * 3], pix_attrib[(i-1) * 3 + 1], pix_attrib[(i-1) * 3 + 2])
      pix_pos.append(pix_point)
    image = Image.open(path_name + "/opening_" + str(i) + ".png")
    draw = ImageDraw.Draw(image)
    for j in range(len(points)):
      draw.point((int(pix_pos[j][0]), int(pix_pos[j][1])), "red")
    image.save(path_name + "/opening_" + str(i) + ".png")
    i += 1
node.bypass(True)