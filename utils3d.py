from typing import List

import numpy as np
import torch

def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (
            one, zero, zero, zero,
            zero, cos, -sin, zero,
            zero, sin, cos, zero,
            zero, zero, zero, one
        )
    elif axis == "Y":
        R_flat = (
            cos, zero, sin, zero,
            zero, one, zero, zero,
            -sin, zero, cos, zero,
            zero, zero, zero, one
        )
    elif axis == "Z":
        R_flat = (
            cos, -sin, zero, zero, 
            sin, cos, zero, zero,
            zero, zero, one, zero,
            zero, zero, zero, one
        )
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (4, 4))

def euler_angle_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Code MODIFIED from pytorch3d
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3), XYZ
        convention: permutation of "X", "Y" or "Z", representing the order of Euler rotations to apply.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, euler_angles[..., 'XYZ'.index(c)])
        for c in convention
    ]
    # return functools.reduce(torch.matmul, matrices)
    return matrices[2] @ matrices[1] @ matrices[0]

def intrinsic_from_fov(fov: float, width: int, height: int) -> np.ndarray:
    normed_int =  np.array([
        [0.5 / (np.tan(fov / 2) * (width / max(width, height))), 0., 0.5],
        [0., 0.5 / (np.tan(fov / 2) * (height / max(width, height))), 0.5],
        [0., 0., 1.],
    ], dtype=np.float32)
    return normed_int * np.array([width, height, 1], dtype=np.float32).reshape(3, 1)

def moving_least_square(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor):
    # 1-D MLS: x: (..., N), y: (..., N), w: (..., N)
    p = torch.stack([torch.ones_like(x), x], dim=-2)             # (..., 2, N)
    print("moving_least_square 0000000000 x.shape:", x.shape, " x:",x)# x: tensor([[[-0.0690, -0.0345,  0.0000]]])
    print("moving_least_square 0000000000 y.shape:", y.shape, " y:",y)
    print("moving_least_square 0000000000 y.shape:", w.shape, " w:",w)

    # p: tensor([[[[1.0000, 1.0000, 1.0000], [-0.0690, -0.0345, 0.0000]]]])
    # M.shape: torch.Size([1, 1, 2, 2])
    # M: tensor([[[[1.4483, -0.0476], [-0.0476, 0.0027]]]])
    print("moving_least_square 1111 p.shape:", p.shape, " p:",p)

    M = p @ (w[..., :, None] * p.transpose(-2, -1))
    print("moving_least_square 2222 M.shape:", M.shape, " M:",M) # M.shape: torch.Size([1, 1, 2, 2])
    pwy = p @ (w * y)
    print("moving_least_square 2222 aaaaaa pwy.shape:", pwy.shape, " pwy:", pwy) # torch.Size([1, 50, 2, 3])
    pwyn = p @ (w * y)[..., :, None]
    print("moving_least_square 2222 bbbbbb pwyn.shape:", pwyn.shape, " pwyn:", pwyn) # torch.Size([50, 3, 2, 1])

    a = torch.linalg.solve(M, (p @ (w * y)[..., :, None]))
    print("moving_least_square 3333 a:",a)
    a = a.squeeze(-1)
    print("moving_least_square 4444 a:",a)

    return a

def mls_smooth(input_t: List[float], input_y: List[np.ndarray], query_t: float, smooth_range: float):
    # 1-D MLS: input_t: (N), input_y: (..., N), query_t: scalar
    if len(input_y) == 1:
        print("mls_smooth 0000000")
        return input_y[0]

    print("mls_smooth 111111")
    input_t = torch.tensor(input_t) - query_t
    print("mls_smooth 222222 input_t:", input_t)
    input_y = torch.stack(input_y, axis=-1)
    print("mls_smooth 333333 input_y:", input_y)
    broadcaster = (None,)*(len(input_y.shape) - 1)
    print("mls_smooth 444444 broadcaster:", broadcaster)
    w = torch.maximum(smooth_range - torch.abs(input_t), torch.tensor(0))
    print("mls_smooth 555555 w:",w)
    coef = moving_least_square(input_t[broadcaster], input_y, w[broadcaster])
    print("mls_smooth coef")
    return coef[..., 0]

def moving_least_square_numpy(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    # 1-D MLS: x: (..., N), y: (..., N), w: (..., N)
    p = np.stack([np.ones_like(x), x], axis=-2)
    # M = p @ (w[..., :, None] * p.swapaxes(-2, -1))
    w1 = w[..., :, None]  # (..., 2, N)
    p1 = p.swapaxes(-2, -1) # swapaxes https://blog.csdn.net/qq_38103303/article/details/105345096
    wp = w1 * p1
    M = p @ (wp)
    a = np.linalg.solve(M, (p @ (w * y)[..., :, None]))
    a = a.squeeze(-1) # squeeze https://blog.csdn.net/zenghaitao0128/article/details/78512715
    return a

def mls_smooth_numpy(input_t: List[float], input_y: List[np.ndarray], query_t: float, smooth_range: float):
    # 1-D MLS: input_t: (N), input_y: (..., N), query_t: scalar
    if len(input_y) == 1:
        return input_y[0]
    input_t = np.array(input_t) - query_t
    input_y = np.stack(input_y, axis=-1)
    broadcaster = (None,)*(len(input_y.shape) - 1)
    w = np.maximum(smooth_range - np.abs(input_t), 0)
    coef = moving_least_square_numpy(input_t[broadcaster], input_y, w[broadcaster])
    # return coef[..., 0]
    coef1 = coef[..., 0]
    return coef1
