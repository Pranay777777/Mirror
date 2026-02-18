import numpy as np
import math

# User Formula (The Truth)
def decompose_user(R):
    sy = np.sqrt(R[0,0]**2 + R[0,1]**2)
    singular = sy < 1e-6
    
    if not singular:
        pitch = math.atan2(R[1,2], R[2,2])
        yaw = math.atan2(-R[0,2], sy)
        roll = math.atan2(R[0,1], R[0,0])
    else:
        pitch = 0 # To be found
        yaw = 0
        roll = 0
        
    return np.rad2deg(pitch), np.rad2deg(yaw), np.rad2deg(roll)

# Euler Rotations
def Rx(theta):
    c, s = np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def Ry(theta):
    c, s = np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def Rz(theta):
    c, s = np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

# Try all orders with sign flips on input
axes_funcs = {'X': Rx, 'Y': Ry, 'Z': Rz}
orders = ['XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX']
signs = [1, -1]

best_config = None
min_error = 1e9

p_in, y_in, r_in = 10, 20, 30

for order in orders:
    for sp in signs:
        for sy in signs:
            for sr in signs:
                # Construct Matrix
                # Order XYZ means R = Rz(r) @ Ry(y) @ Rx(p)? Or Rx @ Ry @ Rz?
                # Let's assume Active rotations R_last @ ... @ R_first
                # 'XYZ' -> Rz @ Ry @ Rx (Standard Extrinsic XYZ?)
                # Wait, standard notation "XYZ" usually means R = Rx * Ry * Rz (Intrinsic)
                # Let's try both compositions for each string
                
                # Composition 1: F1 @ F2 @ F3
                m1 = axes_funcs[order[0]](p_in * sp)
                m2 = axes_funcs[order[1]](y_in * sy)
                m3 = axes_funcs[order[2]](r_in * sr) # Wait, params are p,y,r.
                # But order 'XZY' uses p, r, y?
                # Let's map axes to values
                val_map = {'X': p_in*sp, 'Y': y_in*sy, 'Z': r_in*sr}
                
                # Composition A: M[0] @ M[1] @ M[2]
                R_A = axes_funcs[order[0]](val_map[order[0]]) @ axes_funcs[order[1]](val_map[order[1]]) @ axes_funcs[order[2]](val_map[order[2]])
                
                p_out, y_out, r_out = decompose_user(R_A)
                err = abs(p_out - p_in) + abs(y_out - y_in) + abs(r_out - r_in)
                if err < min_error:
                    min_error = err
                    best_config = (order, 'A', sp, sy, sr)
                    
                # Composition B: M[2] @ M[1] @ M[0]
                R_B = axes_funcs[order[2]](val_map[order[2]]) @ axes_funcs[order[1]](val_map[order[1]]) @ axes_funcs[order[0]](val_map[order[0]])
                p_out, y_out, r_out = decompose_user(R_B)
                err = abs(p_out - p_in) + abs(y_out - y_in) + abs(r_out - r_in)
                if err < min_error:
                    min_error = err
                    best_config = (order, 'B', sp, sy, sr)

print(f"Best Config: {best_config}, Error: {min_error}")

if best_config:
    order, comp, sp, sy, sr = best_config
    print("Verifying Singular Case...")
    # Generate Singular Matrix (Yaw=90)
    p_s, y_s, r_s = 10, 90, 0
    val_map = {'X': p_s*sp, 'Y': y_s*sy, 'Z': r_s*sr}
    
    m0 = axes_funcs[order[0]](val_map[order[0]])
    m1 = axes_funcs[order[1]](val_map[order[1]])
    m2 = axes_funcs[order[2]](val_map[order[2]])
    
    if comp == 'A':
        R_sing = m0 @ m1 @ m2
    else:
        R_sing = m2 @ m1 @ m0
        
    print(f"Singular Matrix (p=10, y=90, r=0):\n{R_sing}")
    
    # Brute force search for Pitch (10)
    # Check atan2(±R[i,j], ±R[k,l])
    found = []
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    # Try simple pairs
                    try:
                        val = math.atan2(R_sing[i,j], R_sing[k,l])
                        deg = np.rad2deg(val)
                        if abs(deg - p_s) < 0.1:
                            found.append(f"atan2(R[{i},{j}], R[{k},{l}])")
                        if abs(deg - (-p_s)) < 0.1:
                            found.append(f"-atan2(R[{i},{j}], R[{k},{l}])")
                            
                        # Try with negative numerator
                        val = math.atan2(-R_sing[i,j], R_sing[k,l])
                        deg = np.rad2deg(val)
                        if abs(deg - p_s) < 0.1:
                            found.append(f"atan2(-R[{i},{j}], R[{k},{l}])")
                    except:
                        pass
                        
    print("Candidates for Pitch Fallback:")
    for f in found[:5]:
        print(f)
