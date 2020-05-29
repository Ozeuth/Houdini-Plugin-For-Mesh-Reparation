node = hou.pwd() 
geo = node.geometry()
geometry_name = node.parent().name()
path = "/obj/" + geometry_name + "/"

group_1 = hou.node(path + "find_border")
group_2 = hou.node(path + "view_borders")
group_3 = hou.node(path + "edge_border")
py_3d = hou.node(path + "prepare_3d")
vex = hou.node(path + "3d_to_2d")
py_2d = hou.node(path + "prepare_2d")
output = hou.node(path + "output")

subnet = node.parent().collapseIntoSubnet((group_1, group_2, group_3, py_3d, vex, py_2d, output), "mesh_repairer")

if subnet.canCreateDigitalAsset():
  asset = subnet.createDigitalAsset(
    name="Mesh_Repairer_Oz",
    min_num_inputs = 1,
    max_num_inputs = 1,
    ignore_external_references = True)
  asset.layoutChildren()
  parm_group = asset.parmTemplateGroup()

  inputs_folder = hou.FolderParmTemplate("inputs_folder", "Inputs", folder_type = hou.folderType.Tabs)
  inputs_folder.addParmTemplate(hou.StringParmTemplate("repairPath", "Repairer Path", 1, help="Path to Mesh Repairer", script_callback='if not ("# -- Houdini Mesh Repairer -- #" in hou.sessionModuleSource()): hou.appendSessionModuleSource(open(hou.pwd().parm("repairPath").eval() + "/session.py", "r").read())', script_callback_language = hou.scriptLanguage.Python))
  inputs_folder.addParmTemplate(hou.ButtonParmTemplate("new", "New Reparation", script_callback = "hou.session.repair()", script_callback_language = hou.scriptLanguage.Python, help="Begin New Reparation"))

  parm_group.append(inputs_folder)
  asset.setParmTemplateGroup(parm_group)
