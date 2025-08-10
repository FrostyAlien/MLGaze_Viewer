"""Coordinate system conversion utilities for Unity to Rerun transformations."""

import numpy as np
from typing import List


def quaternion_rotate_vector(q: List[float], v: List[float]) -> List[float]:
    """Rotate a 3D vector by a quaternion.
    
    Args:
        q: Quaternion in XYZW format [x, y, z, w]
        v: 3D vector to rotate [x, y, z]
        
    Returns:
        Rotated 3D vector [x, y, z]
    """
    qx, qy, qz, qw = q
    vx, vy, vz = v

    # Convert to standard quaternion rotation formula
    # v' = v + 2 * cross(q_xyz, cross(q_xyz, v) + q_w * v)
    q_xyz = np.array([qx, qy, qz])
    v_arr = np.array([vx, vy, vz])

    cross1 = np.cross(q_xyz, v_arr)
    cross2 = np.cross(q_xyz, cross1 + qw * v_arr)
    result = v_arr + 2 * cross2

    return result.tolist()


def quaternion_to_matrix(q: List[float]) -> np.ndarray:
    """Convert quaternion to 3x3 rotation matrix.
    
    Args:
        q: Quaternion in XYZW format [x, y, z, w]
        
    Returns:
        3x3 rotation matrix as numpy array
    """
    x, y, z, w = q

    # Normalize quaternion
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm > 0:
        x, y, z, w = x / norm, y / norm, z / norm, w / norm

    # Build rotation matrix
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    matrix = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
    ])

    return matrix


def matrix_to_quaternion(matrix: np.ndarray) -> List[float]:
    """Convert 3x3 rotation matrix to quaternion.
    
    Args:
        matrix: 3x3 rotation matrix as numpy array
        
    Returns:
        Quaternion in XYZW format [x, y, z, w]
    """
    m = matrix

    # Based on method from Shepperd (1978)
    trace = m[0, 0] + m[1, 1] + m[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    return [float(x), float(y), float(z), float(w)]


def compose_quaternions(q1: List[float], q2: List[float]) -> List[float]:
    """Compose (multiply) two quaternions: result = q1 * q2.
    
    Args:
        q1: First quaternion in XYZW format [x, y, z, w]
        q2: Second quaternion in XYZW format [x, y, z, w]
        
    Returns:
        Composed quaternion in XYZW format [x, y, z, w]
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    # Quaternion multiplication formula
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return [x, y, z, w]


def unity_to_rerun_position(pos: List[float]) -> List[float]:
    """Convert Unity left-handed Y-up to Rerun RDF (Right-Down-Forward).
    Unity: X=right, Y=up, Z=forward (left-handed)
    RDF: X=right, Y=down, Z=forward (right-handed)
    
    Need to:
    1. Flip Y-axis: Y_rdf = -Y_unity (up to down)
    2. Keep Z-axis: Z_rdf = Z_unity (forward stays forward)
    """
    return [pos[0], -pos[1], pos[2]]


def unity_to_rerun_quaternion(q: List[float]) -> List[float]:
    """Convert Unity left-handed Y-up quaternion to Rerun RDF coordinate system.
    Unity: X=right, Y=up, Z=forward (left-handed)
    RDF: X=right, Y=down, Z=forward (right-handed)
    
    This requires:
    1. Converting the rotation from Unity's coordinate system
    2. Handling the change from left-handed to right-handed system
    """
    # Convert quaternion to rotation matrix
    rot_matrix = quaternion_to_matrix(q)

    # Unity to RDF coordinate transformation
    unity_to_rdf = np.array([
        [1, 0, 0],   # X stays right
        [0, -1, 0],  # Y flips (up to down)
        [0, 0, 1]    # Z stays forward
    ])

    # For rotation matrices, the transformation is: R_rdf = T * R_unity * T^(-1)
    # Calculate T^(-1) (which happens to equal T in this case)
    unity_to_rdf_inv = unity_to_rdf.T  # For orthogonal matrices, inverse = transpose

    # Apply the transformation
    transformed_matrix = unity_to_rdf @ rot_matrix @ unity_to_rdf_inv

    # Convert back to quaternion
    result_q = matrix_to_quaternion(transformed_matrix)

    # Ensure proper conversion to list of floats
    return [float(result_q[0]), float(result_q[1]), float(result_q[2]), float(result_q[3])]