import os.path
import time
import PIL
from PIL import Image, ImageDraw
node = hou.pwd()
geo = node.geometry()
boundaries = node.inputs()[1].geometry().pointGroups()

cam = geo.findPointAttrib("cam")
pix = geo.findPointAttrib("pix")
path_name = hou.hipFile.name().split(".")[0]
if (cam and pix):
  i = 0
  for boundary in boundaries:
    if i == 0:
      i+= 1
      continue
    cam_pos = []
    pix_pos = []
    points = boundaries[i].points()
    for point in points:
      cam_attrib = point.attribValue(cam)
      pix_attrib = point.attribValue(pix)
      cam_point = (cam_attrib[(i-1) * 3], cam_attrib[(i-1) * 3 + 1], cam_attrib[(i-1) * 3 + 2])
      pix_point = (pix_attrib[(i-1) * 3], pix_attrib[(i-1) * 3 + 1], pix_attrib[(i-1) * 3 + 2])
      cam_pos.append(cam_point)
      pix_pos.append(pix_point)
    while (os.path.isfile(path_name + "/opening_" + str(i) + ".png.mantra_checkpoint")):
      time.sleep(1) # This image has not yet been rendered
    image = Image.open(path_name + "/opening_" + str(i) + ".png")
    # Pixel y coordinates are inversed. So we flip vertically the image, draw and flip back
    image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    draw = ImageDraw.Draw(image)
    for j in range(len(points)):
      draw.point((int(pix_pos[j][0]), int(pix_pos[j][1])), "red")
    image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    image.save(path_name + "/opening_" + str(i) + ".png")
    i += 1