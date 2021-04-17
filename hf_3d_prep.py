import hou
import objecttoolutils

node = hou.pwd()
geo = node.geometry()

point_boundaries = []
point_patches = []
for point_group in geo.pointGroups():
  if "patch" in point_group.name():
    point_patches.append(point_group)
  else:
    point_boundaries.append(point_group)

old_blast_nodes = hou.session.find_nodes("blast_")
old_patch_nodes = hou.session.find_nodes("patch_")

for i in range(0, len(point_boundaries)): 
  point_patch = point_patches[i]

  # blast_node = repaired_patch \ patch_i
  blast_node = old_blast_nodes[i] if len(old_blast_nodes) > i else node.parent().createNode("blast", "blast_" + point_boundaries[i].name())
  blast_node.setPosition(node.position() + hou.Vector2(i, -1))
  blast_node.parm("group").set(point_patch.name())
  blast_node.parm("grouptype").set(3)
  blast_node.setInput(0, node)

  # patch_node = patch_i
  patch_node = old_patch_nodes[i] if len(old_patch_nodes) > i else node.parent().createNode("boolean::2.0", point_patch.name())
  patch_node.setPosition(blast_node.position() + hou.Vector2(0, -1))
  patch_node.parm("booleanop").set(2)
  patch_node.setInput(0, node.input(0))
  patch_node.setInput(1, blast_node)

# delete excess nodes from prior reparations
for j in range(i+1, len(old_blast_nodes)):
  old_blast_nodes[j].destroy(True)
  old_patch_nodes[j].destroy(True)
