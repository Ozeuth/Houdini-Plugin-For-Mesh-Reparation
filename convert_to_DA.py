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
  inputs_folder.addParmTemplate(hou.ToggleParmTemplate("isPath", "Use Unique Path", 0, help="Check to change path to look for session file", script_callback='if not bool(hou.pwd().parm("isPath").eval()): hou.pwd().parm("repairPath").set("Using path to HIP")', script_callback_language = hou.scriptLanguage.Python))
  inputs_repair_path = hou.StringParmTemplate("repairPath", "Repairer Path", 1, default_value=("Using path to HIP",), help="Path to Mesh Repairer")
  inputs_repair_path.setConditional(hou.parmCondType.DisableWhen, "{ isPath == 0 }")
  inputs_folder.addParmTemplate(inputs_repair_path)
  input_control = 'repair_path = hou.pwd().parm("repairPath").eval() if bool(hou.pwd().parm("isPath").eval()) else hou.getenv("HIP");\
  hou.appendSessionModuleSource(open(repair_path + "/session.py", "r").read() if not "# -- Houdini Mesh Repairer -- #" in hou.sessionModuleSource() else "")'
  inputs_folder.addParmTemplate(hou.ButtonParmTemplate("inputs_init", "Initialize Repairer Session",
   script_callback = input_control, script_callback_language = hou.scriptLanguage.Python, help="Initialize Houdini session using Repairer Path"))

  inputs_folder.addParmTemplate(hou.ToggleParmTemplate("isSmooth", "Smooth Boundaries", 0,
    script_callback='hou.node(hou.pwd().path() + "/smooth_boundaries/smooth").parm("strength").set(int(hou.pwd().parm("inputs_smooth_factor").eval()) * int(hou.pwd().parm("isSmooth").eval()))',
    script_callback_language=hou.scriptLanguage.Python, help="Smooth input hole boundaries"))
  inputs_smooth_factor = hou.IntParmTemplate("inputs_smooth_factor", "Smooth Boundaries Factor", 
    1, default_value=(50,), min=0, max=100,
    min_is_strict=True, max_is_strict=False,
    script_callback='hou.node(hou.pwd().path() + "/smooth_boundaries/smooth").parm("strength").set(int(hou.pwd().parm("inputs_smooth_factor").eval()) * int(hou.pwd().parm("isSmooth").eval()))',
    script_callback_language=hou.scriptLanguage.Python, help="Intensity of pre-repair smoothing")
  inputs_smooth_factor.setConditional(hou.parmCondType.DisableWhen, "{ isSmooth == 0 }")
  inputs_folder.addParmTemplate(inputs_smooth_factor)
  inputs_folder.addParmTemplate(hou.ButtonParmTemplate("new", "Full Reparation", script_callback = "hou.session.repair()", script_callback_language = hou.scriptLanguage.Python, help="Begin New Reparation"))
  
  # Low Frequency Folder
  low_folder = hou.FolderParmTemplate("low folder", "Low Frequency", folder_type = hou.folderType.Tabs)

  # - Low Frequency Small Folder
  low_small_folder = hou.FolderParmTemplate("low small folder", "Small Hole", folder_type = hou.folderType.Collapsible)
  low_small_options = hou.MenuParmTemplate("low_small_type", "Small Hole Repairer", ("MCT", "Centroid"), default_value = 0, help="hole filling technique to use on small holes")
  low_small_folder.addParmTemplate(low_small_options)

  # - Low Frequency Medium Folder
  low_med_folder = hou.FolderParmTemplate("low med folder", "Medium Hole", folder_type = hou.folderType.Collapsible)
  low_med_options =  hou.MenuParmTemplate("low_med_type", "Medium Hole Repairer", ("MCT with MRF",), default_value = 0, help="hole filling technique to use on medium holes")
  low_med_folder.addParmTemplate(low_med_options)

  # - Low Frequency Large Folder
  low_large_folder = hou.FolderParmTemplate("low large folder", "Large Hole", folder_type = hou.folderType.Collapsible)
  low_large_options =  hou.MenuParmTemplate("low_large_type", "Large Hole Repairer", ("MLS with MCT", "Improved AFT"), default_value = 0, help="hole filling technique to use on large holes")
  low_large_options.setMenuUseToken(True)
  low_large_folder.addParmTemplate(low_large_options)
  # -- AFT
  low_alpha_beta = hou.FloatParmTemplate("low_alpha_beta", "Alpha:Beta", 
    1, default_value=(0.5,), min=0.0, max=1.0, min_is_strict=True, max_is_strict=True,
    help="Used in normal correction. Try larger alpha for rounder output topologies, larger beta for flatter output topologies")
  low_w1_w2 = hou.FloatParmTemplate("low_w1_w2", "w1:w2",
    1, default_value=(0.5,), min=0.0, max=1.0, min_is_strict=True, max_is_strict=True,
    help="Used in learning of optimized new point positions")
  low_alpha_beta.setConditional(hou.parmCondType.HideWhen, "{ low_large_type == 0 }")
  low_w1_w2.setConditional(hou.parmCondType.HideWhen, "{ low_large_type == 0 }")
  low_large_folder.addParmTemplate(low_alpha_beta)
  low_large_folder.addParmTemplate(low_w1_w2)

  low_isIter = hou.ToggleParmTemplate("isIter", "Use Iteration Threshold", 0, help="Check to limit number of iterations of AFT")
  low_iter_threshold = hou.IntParmTemplate("low_iter_threshold", "Iteration Threshold", 
    1, default_value=(2000,), min=0, max=3000,
    min_is_strict=True, max_is_strict=False, 
    help="Maximum number of iterations AFT can run for")
  low_isIter.setConditional(hou.parmCondType.HideWhen, "{ low_large_type == 0 }")
  low_iter_threshold.setConditional(hou.parmCondType.DisableWhen, "{ isIter == 0 }")
  low_iter_threshold.setConditional(hou.parmCondType.HideWhen, "{ low_large_type == 0 }")
  low_large_folder.addParmTemplate(low_isIter)
  low_large_folder.addParmTemplate(low_iter_threshold)

  low_isAngle = hou.ToggleParmTemplate("isAngle", "Use Angle Threshold", 1, help="Check to allow randomized AFT generation")
  low_angle_threshold = hou.IntParmTemplate("low_angle_threshold", "Angle Threshold", 
    1, default_value=(140,), min=0, max=360,
    min_is_strict=True, max_is_strict=True, 
    help="Maximum angle for a boundary point to be considered for AFT randomized selection")
  low_isAngle.setConditional(hou.parmCondType.HideWhen, "{ low_large_type == 0 }")
  low_angle_threshold.setConditional(hou.parmCondType.DisableWhen, "{ isAngle == 0 }")
  low_angle_threshold.setConditional(hou.parmCondType.HideWhen, "{ low_large_type == 0 }")
  low_large_folder.addParmTemplate(low_isAngle)
  low_large_folder.addParmTemplate(low_angle_threshold)

  low_isDistance = hou.ToggleParmTemplate("isDistance", "Use Distance Threshold", 1, help="Check to allow merger of new AFT points by distance")
  low_distance_threshold = hou.FloatParmTemplate("low_distance_threshold", "Distance Threshold",
    1, default_value=(0.25,), min=0.0, max=1.0,
    min_is_strict=True, max_is_strict=True,
    help="Maximum proportion of distance to another point to be considered for AFT point merger")
  low_isDistance.setConditional(hou.parmCondType.HideWhen, "{ low_large_type == 0 }")
  low_distance_threshold.setConditional(hou.parmCondType.DisableWhen, "{ isDistance == 0 }")
  low_distance_threshold.setConditional(hou.parmCondType.HideWhen, "{ low_large_type == 0 }")
  low_large_folder.addParmTemplate(low_isDistance)
  low_large_folder.addParmTemplate(low_distance_threshold)

  # -- MLS
  low_rank_factor = hou.FloatParmTemplate("low_rank_factor", "Rank Tolerance Factor",
    1, default_value=(5.0,), min=0.0, max=10.0,
    min_is_strict=True, max_is_strict=False,
    help="Change as instructed until hole is correctly detected as coplanar/ not coplanar")
  low_rank_factor.setConditional(hou.parmCondType.HideWhen, "{ low_large_type == 1 }")
  low_large_folder.addParmTemplate(low_rank_factor)
  
  low_scale_factor = hou.FloatParmTemplate("low_scale_factor", "Projection Scale Factor",
    1, default_value=(1.0,), min=0.0, max=2.0,
    min_is_strict=True, max_is_strict=False,
    help="Increase if MLS image, ./demo/see_new_sampling.png is blurry or completely black")
  low_scale_factor.setConditional(hou.parmCondType.HideWhen, "{ low_large_type == 1 }")
  low_large_folder.addParmTemplate(low_scale_factor)
  
  low_folder.addParmTemplate(low_small_folder)
  low_folder.addParmTemplate(low_med_folder)
  low_folder.addParmTemplate(low_large_folder)
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