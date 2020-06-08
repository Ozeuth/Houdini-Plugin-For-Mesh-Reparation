node = hou.pwd() 
geo = node.geometry()
geometry_name = node.parent().name()
path = "/obj/" + geometry_name + "/"

bbox = hou.node(path + "oz_bbox")
uv_bbox = hou.node(path + "uv_viewer_bbox")
uv_viewer_all = hou.node(path + "uv_viewer_all")
uv_viewer = hou.node(path + "uv_viewer")
group_1 = hou.node(path + "find_border")
group_2 = hou.node(path + "view_borders")
group_3 = hou.node(path + "edge_border")
py_3d = hou.node(path + "prepare_3d")
py_op_3d = hou.node(path + "optimize_3d")
vex_clean = hou.node(path + "clean_3d_to_2d")
vex_1 = hou.node(path + "3d_to_2d_1")
vex_2 = hou.node(path + "3d_to_2d_2")
vex_ratio = hou.node(path + "3d_to_2d_ratio")
py_2d = hou.node(path + "prepare_2d")
output = hou.node(path + "output")

subnet = node.parent().collapseIntoSubnet((bbox, uv_bbox, uv_viewer_all, group_1, group_2, group_3, uv_viewer, py_3d, vex_1, py_op_3d, vex_clean, vex_2, vex_ratio, py_2d, output), "mesh_repairer")

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


  mapping_3d_to_2d_folder = hou.FolderParmTemplate("3d_to_2d_folder", "3D to 2D", folder_type = hou.folderType.Tabs)
  mapping_3d_to_2d_folder.addParmTemplate(hou.ToggleParmTemplate("isAlpha", "Change Alpha", 0, help="Check if you wish to use a different alpha to determine optimal occupancy ratio"))
  alpha = hou.FloatParmTemplate("alpha", "Occupancy Ratio Alpha", 1, default_value=(0.87,), min=0.0, max=1.0, min_is_strict=True, max_is_strict=True, help="alpha used in S Salamanca,P Merchan,A Adan,E Perez,C Cerrada[2008]")
  alpha.setConditional(hou.parmCondType.DisableWhen, "{ isAlpha == 0 }")
  mapping_3d_to_2d_folder.addParmTemplate(alpha)

  parm_group.append(inputs_folder)
  parm_group.append(mapping_3d_to_2d_folder)
  asset.setParmTemplateGroup(parm_group)
