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
            print(f"Joint: {joint_path}")

            # Joint Type
            print(f"  Type: {prim.GetTypeName()}")

            # Get joint limits (specific to joint type)
            if prim.IsA(UsdPhysics.RevoluteJoint):
                lower_limit_attr = prim.GetAttribute("physics:lowerLimit")
                upper_limit_attr = prim.GetAttribute("physics:upperLimit")
                lower_limit = lower_limit_attr.Get() if lower_limit_attr and lower_limit_attr.HasValue() else "N/A"
                upper_limit = upper_limit_attr.Get() if upper_limit_attr and upper_limit_attr.HasValue() else "N/A"
                print(f"  Lower Limit: {lower_limit} (deg)")
                print(f"  Upper Limit: {upper_limit} (deg)")
            elif prim.IsA(UsdPhysics.PrismaticJoint):
                lower_limit_attr = prim.GetAttribute("physics:lowerLimit")
                upper_limit_attr = prim.GetAttribute("physics:upperLimit")
                lower_limit = lower_limit_attr.Get() if lower_limit_attr and lower_limit_attr.HasValue() else "N/A"
                upper_limit = upper_limit_attr.Get() if upper_limit_attr and upper_limit_attr.HasValue() else "N/A"
                print(f"  Lower Limit: {lower_limit} (m)")
                print(f"  Upper Limit: {upper_limit} (m)")
            # Add checks for other joint types like SphericalJoint, FixedJoint etc. if needed

            # Get Drive parameters if available (for actuated joints)
            # Check for specific drive types based on joint type
            drive_api = None
            if prim.IsA(UsdPhysics.RevoluteJoint):
                # Try to get the angular drive API
                drive_api = UsdPhysics.DriveAPI.Get(prim, "angular")
            elif prim.IsA(UsdPhysics.PrismaticJoint):
                 # Try to get the linear drive API
                 drive_api = UsdPhysics.DriveAPI.Get(prim, "linear")
            # Add elif for other joint types if they support drives (e.g., D6Joint)

            if drive_api: # Check if a valid DriveAPI was found
                drive_type_attr = drive_api.GetTypeAttr()
                drive_type = drive_type_attr.Get() if drive_type_attr and drive_type_attr.HasValue() else "N/A"
                damping_attr = drive_api.GetDampingAttr()
                stiffness_attr = drive_api.GetStiffnessAttr()
                max_force_attr = drive_api.GetMaxForceAttr()

                damping = damping_attr.Get() if damping_attr and damping_attr.HasValue() else "N/A"
                stiffness = stiffness_attr.Get() if stiffness_attr and stiffness_attr.HasValue() else "N/A"
                max_force = max_force_attr.Get() if max_force_attr and max_force_attr.HasValue() else "N/A"
                
                print(f"  Drive Type: {drive_type}")
                print(f"  Damping: {damping}")
                print(f"  Stiffness: {stiffness}")
                print(f"  Max Force (Effort Limit): {max_force}")

            # Get Max Velocity (often defined on the joint itself, not the drive)
            max_vel_attr = prim.GetAttribute("physics:maxVelocity")
            max_velocity = max_vel_attr.Get() if max_vel_attr and max_vel_attr.HasValue() else "N/A"
            print(f"  Max Velocity: {max_velocity}")

            # Get joint friction
            friction_attr = prim.GetAttribute("physics:jointFriction")
            friction = friction_attr.Get() if friction_attr and friction_attr.HasValue() else "N/A"
            
            # Get joint armature
            armature_attr = prim.GetAttribute("physics:armature")
            armature = armature_attr.Get() if armature_attr and armature_attr.HasValue() else "N/A"
            
            print(f"  Friction: {friction}")
            print(f"  Armature: {armature}")
            print("-" * 20) # Separator for clarity
            

# Example usage
list_usd_prims("USD_files/parallell_spring_jumper/parallell_spring_jumper.usd")
get_robot_mass("USD_files/parallell_spring_jumper/parallell_spring_jumper.usd")
get_joint_properties("USD_files/parallell_spring_jumper/parallell_spring_jumper.usd")