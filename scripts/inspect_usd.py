from pxr import Usd

def list_usd_prims(file_path):
    """List all prims in a USD file."""
    # Open the USD stage
    stage = Usd.Stage.Open(file_path)
    
    # Iterate over all prims in the stage
    for prim in stage.Traverse():
        print(prim.GetPath())

# Example usage
list_usd_prims("submodules/cad/simplified_robot/simplified_robot.usd")