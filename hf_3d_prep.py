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
# NOTE: Houdini parm template ignores the first / in path
if not os.path.exists(synthesizer_path):
  if os.path.exists("/" + synthesizer_path):
    synthesizer_path = "/" + synthesizer_path
  else:
    raise Exception("ERROR: Path to geometric texture synthesizer invalid")

old_blast_nodes = hou.session.find_nodes("oz_blast_")
old_tri_nodes = hou.session.find_nodes("oz_tri_")
old_file_nodes = hou.session.find_nodes("oz_file_")

for i in range(0, len(point_boundaries)):
  points = point_boundaries[i] 
  point_patch = point_patches[i]

  lf_3d_node = hou.session.find_nodes("lf_3d", num_nodes=1)
  # blast_node = patch_i
  blast_node = old_blast_nodes[i] if len(old_blast_nodes) > i else node.parent().createNode("blast", "oz_blast_" + points.name())
  blast_node.setPosition(node.position() + hou.Vector2(i, -1))
  blast_node.parm("negate").set(1)
  blast_node.parm("group").set(point_patch.name() + " " + points.name())
  blast_node.parm("grouptype").set(3)
  blast_node.setInput(0, node)

  # triangulate_node
  tri_node = old_tri_nodes[i] if len(old_tri_nodes) > i else node.parent().createNode("divide", "oz_tri_" + points.name())
  tri_node.setPosition(blast_node.position() + hou.Vector2(0, -1))
  tri_node.parm("usemaxsides").set(3)
  tri_node.setInput(0, blast_node)

  # file_node
  file_node = old_file_nodes[i] if len(old_file_nodes) > i else node.parent().createNode("file", "oz_file_" + points.name())
  file_node.setPosition(tri_node.position() + hou.Vector2(0, -1))
  file_node.setInput(0, tri_node)

  dataset_path = synthesizer_path + "/dataset"
  if not os.path.isdir(dataset_path):
    os.mkdir(dataset_path)
  raw_path = dataset_path + "/raw"
  if not os.path.isdir(raw_path):
    os.mkdir(raw_path)
  file_node.parm("file").set(raw_path + "/" + point_patch.name() + ".obj")
  file_node.parm("filemode").set(2)

# delete excess nodes from prior reparations
for j in range(i+1, len(old_blast_nodes)):
  old_blast_nodes[j].destroy(True)
  old_file_nodes[j].destroy(True)

node.bypass(True)