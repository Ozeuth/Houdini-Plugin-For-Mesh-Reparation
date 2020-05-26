import os.path
import time
from PIL import Image, ImageDraw
node = hou.pwd()
geo = node.geometry()
boundaries = node.inputs()[1].geometry().pointGroups()

pos = geo.findPointAttrib("pos")
pix = geo.findPointAttrib("pix")
path_name = hou.hipFile.name().split(".")[0]
if (pos and pix):
  i = 0
  for boundary in boundaries:
    if i == 0:
      i+= 1
      continue
    cam_pos = []
    pix_pos = []
    points = boundaries[i].points()
    for point in points:
      cam_pos.append(point.attribValue(pos)[i])
      pix_pos.append(point.attribValue(pix)[i])
    if i == 1:
      print("Pixels")
      print(pix_pos)
    while (os.path.isfile(path_name + "/opening_" + str(i) + ".png.mantra_checkpoint")):
      time.sleep(1) # Wait until image is rendered
    image = Image.open(path_name + "/opening_" + str(i) + ".png")
    draw = ImageDraw.Draw(image)
    for j in range(len(points)):
      draw.point((int(pix_pos[j][0]), int(pix_pos[j][1])), "red")
    image.save(path_name + "/opening_" + str(i) + ".png")
    i += 1