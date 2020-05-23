node = hou.node("/obj/chosen_geometry/fill_hole")
node.bypass(False)
render_all()

def render_all():
  renders = hou.nodeType(hou.ropNodeTypeCategory(), "ifd").instances()
  for render in renders:
    print("Render")  
    render.render()

