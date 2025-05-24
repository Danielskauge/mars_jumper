colours = [
    (240, 218, 161), # khaki ish base
    (255, 0, 0), # red hip
    (209, 213, 235), # silver thigh
    (50, 142, 250), # blue shank
    (0,0,0), # black paw
]

def rgb_to_urdf(r, g, b):
    """Convert 0-255 RGB values to 0-1 URDF format"""
    return (r / 255.0, g / 255.0, b / 255.0, 1.0)

if __name__ == "__main__":
    # Inline RGB values

    for r, g, b in colours:
        urdf_rgba = rgb_to_urdf(r, g, b)
        print(" ".join(str(c) for c in urdf_rgba))