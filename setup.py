def find_parm(name):
  params = hou.parent().parms()
  found_eval = None
  for param in params:
    if (name in param.name()):
      found_eval = param.eval()
      return found_eval
  return None

output_node = hou.node(hou.pwd().parent().path() + "/output")
repair_path =r"C:\Users\Ozeuth\Python-Houdini-Mesh-Repair"
if (hou.parent()):
  if (find_parm("repairPath")): repair_path = find_parm("repairPath")

if not ("# -- Houdini Mesh Repairer -- #" in hou.sessionModuleSource()):
  session_file = open(repair_path + "/session.py", "r")
  source = session_file.read()
  hou.appendSessionModuleSource(source)
  session_file.close()

output_node.setDisplayFlag(True)
output_node.setRenderFlag(True) 
hou.session.repair() 