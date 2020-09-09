import hou

node = hou.pwd()
path = "/obj/" + node.parent().name() + "/"
file_node = hou.node(path + "mesh_to_repair")

DA_nodes = []
for curr_node in node.parent().children():
  if curr_node != node and curr_node != file_node:
    DA_nodes.append(curr_node)

subnet = node.parent().collapseIntoSubnet(tuple(DA_nodes))
subnet.setName("mesh_repairer")
if subnet.canCreateDigitalAsset():
  asset = subnet.createDigitalAsset(
    name="Mesh_Repairer_Oz",
    min_num_inputs = 1,
    max_num_inputs = 1,
    ignore_external_references = True)
  parm_group = asset.parmTemplateGroup()

  # Input Folder
  inputs_folder = hou.FolderParmTemplate("inputs_folder", "Inputs", folder_type=hou.folderType.Tabs)
  inputs_folder.addParmTemplate(hou.StringParmTemplate("repairPath", "Repairer Path", 1, help="Path to Mesh Repairer", script_callback='if not ("# -- Houdini Mesh Repairer -- #" in hou.sessionModuleSource()): hou.appendSessionModuleSource(open(hou.pwd().parm("repairPath").eval() + "/session.py", "r").read())', script_callback_language = hou.scriptLanguage.Python))
  inputs_folder.addParmTemplate(hou.ToggleParmTemplate("isSmooth", "Smooth Boundaries", 0, help="Smooth input hole boundaries"))
  inputs_folder.addParmTemplate(hou.ButtonParmTemplate("new", "Full Reparation", script_callback = "hou.session.repair()", script_callback_language = hou.scriptLanguage.Python, help="Begin New Reparation"))
  
  # Low Frequency Folder
  low_folder = hou.FolderParmTemplate("low folder", "Low Frequency", folder_type = hou.folderType.Tabs)
  low_alpha_beta = hou.FloatParmTemplate("low_alpha_beta", "Alpha:Beta", 
    1, default_value=(0.5,), min=0.0, max=1.0, min_is_strict=True, max_is_strict=True,
    help="Used in normal correction. Try larger alpha for rounder output topologies, larger beta for flatter output topologies")
  low_w1_w2 = hou.FloatParmTemplate("low_w1_w2", "w1:w2",
    1, default_value=(0.5,), min=0.0, max=1.0, min_is_strict=True, max_is_strict=True,
    help="Used in learning of optimized new point positions")
  low_folder.addParmTemplate(low_alpha_beta)
  low_folder.addParmTemplate(low_w1_w2)

  low_folder.addParmTemplate(hou.ToggleParmTemplate("isIter", "Use Iteration Threshold", 0, help="Check to limit number of iterations of AFT"))
  low_iter_threshold = hou.IntParmTemplate("low_iter_threshold", "Iteration Threshold", 
    1, default_value=(2000,), min=0, max=3000,
    min_is_strict=True, max_is_strict=False, 
    help="Maximum number of iterations AFT can run for")
  low_iter_threshold.setConditional(hou.parmCondType.DisableWhen, "{ isIter == 0 }")
  low_folder.addParmTemplate(low_iter_threshold)
  low_folder.addParmTemplate(hou.ToggleParmTemplate("isAngle", "Use Angle Threshold", 1, help="Check to allow randomized AFT generation"))
  low_angle_threshold = hou.IntParmTemplate("low_angle_threshold", "Angle Threshold", 
    1, default_value=(140,), min=0, max=360,
    min_is_strict=True, max_is_strict=True, 
    help="Maximum angle for a boundary point to be considered for AFT randomized selection")
  low_angle_threshold.setConditional(hou.parmCondType.DisableWhen, "{ isAngle == 0 }")
  low_folder.addParmTemplate(low_angle_threshold)
  low_folder.addParmTemplate(hou.ToggleParmTemplate("isDistance", "Use Distance Threshold", 1, help="Check to allow merger of new AFT points by distance"))
  low_distance_threshold = hou.FloatParmTemplate("low_distance_threshold", "Distance Threshold",
    1, default_value=(0.25,), min=0.0, max=1.0,
    min_is_strict=True, max_is_strict=True,
    help="Maximum proportion of distance to another point to be considered for AFT point merger")
  low_distance_threshold.setConditional(hou.parmCondType.DisableWhen, "{ isDistance == 0 }")
  low_folder.addParmTemplate(low_distance_threshold)
  low_folder.addParmTemplate(hou.ButtonParmTemplate("low_new", "Low Frequency Reparation", script_callback = "hou.session.low_repair()", script_callback_language = hou.scriptLanguage.Python, help="Begin New Low-Frequency Reparation"))

  # High Frequency Folder
  high_folder = hou.FolderParmTemplate("high_folder", "High Frequency", folder_type = hou.folderType.Tabs)
  high_folder.addParmTemplate(hou.ToggleParmTemplate("isAlpha", "Change Alpha", 0, help="Check if you wish to use a different alpha to determine optimal occupancy ratio"))
  alpha = hou.FloatParmTemplate("alpha", "Occupancy Ratio Alpha", 1, default_value=(0.87,), min=0.0, max=1.0, min_is_strict=True, max_is_strict=True, help="alpha used in S Salamanca,P Merchan,A Adan,E Perez,C Cerrada[2008]")
  alpha.setConditional(hou.parmCondType.DisableWhen, "{ isAlpha == 0 }")
  high_folder.addParmTemplate(alpha)
  high_folder.addParmTemplate(hou.ButtonParmTemplate("high_new", "High Frequency Reparation", script_callback = "hou.session.high_repair()", script_callback_language = hou.scriptLanguage.Python, help="Begin New High-Frequency Reparation"))

  parm_group.append(inputs_folder)
  parm_group.append(low_folder)
  parm_group.append(high_folder)
  asset.setParmTemplateGroup(parm_group)
  asset.setName("Mesh_Repairer_Oz")

node.bypass(True)