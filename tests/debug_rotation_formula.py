import numpy as np
import math

# User Formula
def decompose_user(R):
    sy = np.sqrt(R[0,0]**2 + R[0,1]**2)
    singular = sy < 1e-6
    
    if not singular:
        pitch = math.atan2(R[1,2], R[2,2])
        yaw = math.atan2(-R[0,2], sy)
        roll = math.atan2(R[0,1], R[0,0])
    else:
        pitch = 0
        yaw = 90 if R[0,2] < 0 else -90 # Placeholder logic based on R[0,2]
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

orders = {
    'XYZ': lambda p, y, r: Rx(p) @ Ry(y) @ Rz(r),
    'XZY': lambda p, y, r: Rx(p) @ Rz(r) @ Ry(y),
    'YXZ': lambda p, y, r: Ry(y) @ Rx(p) @ Rz(r),
    'YZX': lambda p, y, r: Ry(y) @ Rz(r) @ Rx(p),
    'ZXY': lambda p, y, r: Rz(r) @ Rx(p) @ Ry(y),
    'ZYX': lambda p, y, r: Rz(r) @ Ry(y) @ Rx(p)
}

# Also test reversed multiplication (Pre-multiplication vs Post-multiplication)
# Usually R_global = R_z * R_y * R_x (Intrinsic rotations around moving axis X then Y then Z) -> No wait.
# R_intrinsic = R_x * R_y * R_z?
# Let's test all 12 combinations (6 orders * 2 directions)

p_target, y_target, r_target = 10, 20, 30

print(f"Target: p={p_target}, y={y_target}, r={r_target}")

best_match = None
min_error = 1e9

for name, func in orders.items():
    # Forward Order (e.g. Rx @ Ry ...)
    R = func(p_target, y_target, r_target)
    p, y, r = decompose_user(R)
    err = abs(p-p_target) + abs(y-y_target) + abs(r-r_target)
    #print(f"Order {name}: p={p:.2f}, y={y:.2f}, r={r:.2f} (Err: {err:.4f})")
    
    if err < min_error:
        min_error = err
        best_match = name
    
    # Reverse Order
    R_rev = func(p_target, y_target, r_target).T # effectively reverse multiply for rotation matrices
    # Wait, (A B C).T = C.T B.T A.T. Not quite reverse order with same angles.
    # Let's explicitly construct reverse order.
    # Actually, let's just test "Intrinsic ZYX" vs "Extrinsic XYZ".
    # Standard decomposition usually recovers Intrinsic ZYX or XYZ.
    
print(f"Best Match Order: {best_match} with Error: {min_error}")

# Let's inspect the Best Match Order specifically
if best_match:
    print(f"\n--- Analyzing Singular Case for {best_match} ---")
    
    # Pitch case: Y = 90
    R_sing = orders[best_match](10, 90, 0) # y=90 -> singular
    print(f"R at Pitch=10, Yaw=90, Roll=0:\n{R_sing}")
    
    # Check elements manually to deduce formula
    # We want to recover Pitch=10. Roll=0.
    # Standard derivation:
    # If sy ~ 0, then cos(y) ~ 0.
    # R[0,2] = -sin(y) -> -1. Correct.
    # We need pitch.
    # R[1,2], R[2,2] are 0/0.
    # Look at R[1,1] or R[2,1] etc.
    # Let's print values
    print(f"R[1,1]={R_sing[1,1]:.4f}, R[2,1]={R_sing[2,1]:.4f}")
    print(f"R[1,0]={R_sing[1,0]:.4f}, R[2,0]={R_sing[2,0]:.4f}")
    
    # We want atan2(?, ?) -> 10 deg.
    # tan(10) = 0.176. sin(10)=0.174, cos(10)=0.985
    # R[1,1] ~ 0? R[2,1] ~ ?
    pass
