import hou
import objecttoolutils
import os

node = hou.pwd()
geo = node.geometry()

point_boundaries = []
point_patches = []
for point_group in geo.pointGroups():
  if "patch" in point_group.name():
    point_patches.append(point_group)
  else:
    point_boundaries.append(point_group)

synthesizer_path = hou.session.find_parm(hou.parent(), "synth_path")

old_blast_nodes = hou.session.find_nodes("oz_blast_")
old_tri_nodes = hou.session.find_nodes("oz_tri_")
old_file_input_nodes = hou.session.find_nodes("oz_input_")
old_transform_input_nodes = hou.session.find_nodes("oz_transform_input_")
old_file_output_nodes = hou.session.find_nodes("oz_output_")
old_transform_output_nodes = hou.session.find_nodes("oz_transform_output_")
old_merge_nodes = hou.session.find_nodes("oz_merge_")

for i in range(0, len(point_boundaries)):
  points = point_boundaries[i] 
  point_patch = point_patches[i]

  lf_3d_node = hou.session.find_nodes("lf_3d", num_nodes=1)
  # blast_node = patch_i
  blast_node = old_blast_nodes[i] if len(old_blast_nodes) > i else node.parent().createNode("blast", "oz_blast_" + points.name())
  blast_node.setPosition(node.position() + hou.Vector2(i * 2, -1))
  blast_node.parm("negate").set(1)
  blast_node.parm("group").set(point_patch.name() + " " + points.name())
  blast_node.parm("grouptype").set(3)
  blast_node.setInput(0, node)

  # triangulate_node
  tri_node = old_tri_nodes[i] if len(old_tri_nodes) > i else node.parent().createNode("divide", "oz_tri_" + points.name())
  tri_node.setPosition(blast_node.position() + hou.Vector2(0, -1))
  tri_node.parm("usemaxsides").set(3)
  tri_node.setInput(0, blast_node)

  # file_input_node
  file_input_node = old_file_input_nodes[i] if len(old_file_input_nodes) > i else node.parent().createNode("file", "oz_input_" + points.name())
  file_input_node.setPosition(tri_node.position() + hou.Vector2(0, -1))
  file_input_node.setInput(0, tri_node)
  dataset_path = synthesizer_path + "/dataset"
  if not os.path.isdir(dataset_path):
    os.mkdir(dataset_path)
  raw_path = dataset_path + "/raw"
  if not os.path.isdir(raw_path):
    os.mkdir(raw_path)
  file_input_node.parm("file").set(raw_path + "/" + point_patch.name() + ".obj")
  file_input_node.parm("filemode").set(2)

  # transform_input_node
  transform_input_node = old_transform_input_nodes[i] if len(old_transform_input_nodes) > i else node.parent().createNode("xform", "oz_transform_input_" + points.name())
  transform_input_node.setPosition(file_input_node.position() + hou.Vector2(0, -1))
  transform_input_node.setInput(0, file_input_node)

  # file_output_node 
  file_output_node = old_file_output_nodes[i] if len(old_file_output_nodes) > i else node.parent().createNode("file", "oz_output_" + points.name())
  file_output_node.setPosition(file_input_node.position() + hou.Vector2(1, -0.5))
  file_output_node.parm("file").set(raw_path + "/" + point_patch.name() + "_hi.obj")
  file_output_node.parm("filemode").set(1)

  # transform_output_node
  transform_output_node = old_transform_output_nodes[i] if len(old_transform_output_nodes) > i else node.parent().createNode("xform", "oz_transform_output_" + points.name())
  transform_output_node.setPosition(file_output_node.position() + hou.Vector2(0, -1))
  transform_output_node.setInput(0, file_output_node)

  # merge_node
  merge_node = old_merge_nodes[i] if len(old_merge_nodes) > i else node.parent().createNode("merge", "oz_merge_" + points.name())
  merge_node.setPosition((transform_input_node.position() + transform_output_node.position()) / 2 + hou.Vector2(0, -1))
  merge_node.setInput(0, transform_input_node)
  merge_node.setInput(1, transform_output_node)

# delete excess nodes from prior reparations
for j in range(i+1, len(old_blast_nodes)):
  old_blast_nodes[j].destroy(True)
  old_tri_nodes[j].destroy(True)
  old_file_input_nodes[j].destroy(True)
  old_transform_input_nodes[j].destroy(True)
  old_file_output_nodes[j].destroy(True)
  old_transform_output_nodes[j].destroy(True)
  old_merge_nodes[j].destroy(True)

node.bypass(True)