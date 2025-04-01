from pxr import Usd, UsdPhysics

def list_usd_prims(file_path):
    """List all prims in a USD file."""
    # Open the USD stage
    stage = Usd.Stage.Open(file_path)
    
    # Iterate over all prims in the stage
    for prim in stage.Traverse():
        print(prim.GetPath())

def get_robot_mass(file_path):
    stage = Usd.Stage.Open(file_path)
    total_mass = 0
    
    for prim in stage.Traverse():
        # Check if the prim has mass information
        mass_api = UsdPhysics.MassAPI(prim)
        if mass_api:
            # Get the mass value
            mass_attr = mass_api.GetMassAttr()
            if mass_attr and mass_attr.HasValue():
                mass = mass_attr.Get()
                print(f"{prim.GetPath()}: {mass} kg")
                total_mass += mass
    
    print(f"Total robot mass: {total_mass} kg")
    return total_mass

def get_joint_properties(file_path):
    stage = Usd.Stage.Open(file_path)
    
    for prim in stage.Traverse():
        # Check if the prim is a joint
        if prim.IsA(UsdPhysics.Joint):
            joint_path = prim.GetPath()
            
            # Get joint friction
            friction_attr = prim.GetAttribute("physics:jointFriction")
            friction = friction_attr.Get() if friction_attr and friction_attr.HasValue() else "N/A"
            
            # Get joint armature
            armature_attr = prim.GetAttribute("physics:armature")
            armature = armature_attr.Get() if armature_attr and armature_attr.HasValue() else "N/A"
            
            print(f"Joint: {joint_path}")
            print(f"  Friction: {friction}")
            print(f"  Armature: {armature}")
            

# Example usage
list_usd_prims("USD_files/parallell_spring_jumper/parallell_spring_jumper.usd")
get_robot_mass("USD_files/parallell_spring_jumper/parallell_spring_jumper.usd")
get_joint_properties("USD_files/parallell_spring_jumper/parallell_spring_jumper.usd")