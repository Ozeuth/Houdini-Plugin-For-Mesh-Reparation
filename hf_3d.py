import hou

merge_nodes = hou.session.find_nodes("oz_merge_")
for merge_node in merge_nodes:
  lo_node = merge_node.inputs()[0]
  hi_node = merge_node.inputs()[1]