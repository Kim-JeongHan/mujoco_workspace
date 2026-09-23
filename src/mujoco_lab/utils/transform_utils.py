"""Workspace interface to SciPy rigid transforms with column-vector conventions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import RigidTransform, Rotation

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d import Axes3D


class Transform:
    """A rigid transform ``p_a = R_ab @ p_b + t_ab``.

    ``T_ab @ T_bc`` maps coordinates from frame c to frame a. Frame names are a
    caller convention, not stored or checked by this class. Rotation must be a
    single SciPy Rotation or a proper orthonormal 3x3 matrix; translation must
    have shape (3,) and is expressed in meters. Matrix and coordinate validation,
    conversion, and rotation normalization are delegated to SciPy.

    Inputs are copied into float64 arrays. Conversion methods return independent
    values, so a transform is a snapshot even when built from live simulator data.
    Pose exports use fixed-axis xyz Euler angles in roll, pitch, yaw order.
    Quaternion pose exports use MuJoCo's scalar-first wxyz order. Frames are
    right-handed, with rotation columns giving local axes in the parent frame.
    """

    __slots__ = ("_transform",)

    def __init__(
        self,
        rotation: Rotation | ArrayLike | None = None,
        translation: ArrayLike | None = None,
    ) -> None:
        if rotation is None:
            rotation = Rotation.identity()
        elif isinstance(rotation, Rotation):
            if not rotation.single:
                raise ValueError("rotation must represent a single rotation")
        else:
            rotation = Rotation.from_matrix(rotation)
        translation = np.zeros(3) if translation is None else translation
        self._transform = RigidTransform.from_components(translation, rotation)

    @classmethod
    def from_transform(cls, transform: RigidTransform) -> Transform:
        instance = cls.__new__(cls)
        instance._transform = transform
        return instance

    @classmethod
    def identity(cls) -> Transform:
        """Return the identity transform."""
        return cls.from_transform(RigidTransform.identity())

    @classmethod
    def from_matrix(cls, matrix: ArrayLike) -> Transform:
        """Construct from a 4x4 matrix with bottom row [0, 0, 0, 1]."""
        return cls.from_transform(RigidTransform.from_matrix(matrix))

    @classmethod
    def from_pose_mrad(cls, pose: ArrayLike) -> Transform:
        """Construct from [x, y, z, roll, pitch, yaw] in meters and radians.

        RPY uses extrinsic (fixed-axis) xyz Euler angles.
        """
        return cls(rotation=Rotation.from_euler("xyz", pose[3:]), translation=pose[:3])

    @classmethod
    def from_pose_mmdeg(cls, pose: ArrayLike) -> Transform:
        """Construct from [x, y, z, roll, pitch, yaw] in millimeters and degrees.

        RPY uses extrinsic (fixed-axis) xyz Euler angles.
        """
        pose = np.asarray(pose, dtype=float)
        return cls(
            rotation=Rotation.from_euler("xyz", pose[3:], degrees=True),
            translation=pose[:3] / 1000.0,
        )

    def as_rotation(self) -> Rotation:
        """Return the rotation as an independent SciPy Rotation object."""
        return self._transform.rotation

    def as_translation(self) -> NDArray[np.float64]:
        """Return a copy of [x, y, z] in meters."""
        return self._transform.translation

    def as_pose_mrad(self) -> NDArray[np.float64]:
        """Return [x, y, z, roll, pitch, yaw] in meters and radians.

        RPY uses extrinsic (fixed-axis) xyz Euler angles, equivalent to
        Rz(yaw) @ Ry(pitch) @ Rx(roll) for column vectors.
        """
        return np.concatenate((self.as_translation(), self.as_rotation().as_euler("xyz")), axis=-1)

    def as_pose_mmrad(self) -> NDArray[np.float64]:
        """Return [x, y, z, roll, pitch, yaw] in millimeters and radians."""
        pose = self.as_pose_mrad()
        pose[..., :3] *= 1000.0
        return pose

    def as_xyzquat(self) -> NDArray[np.float64]:
        """Return [x, y, z, qw, qx, qy, qz] in meters and MuJoCo quaternion order."""
        return np.concatenate(
            (self.as_translation(), self.as_rotation().as_quat(scalar_first=True)), axis=-1
        )

    def as_mmdeg(self) -> NDArray[np.float64]:
        """Return [x, y, z, roll, pitch, yaw] in millimeters and degrees."""
        return np.concatenate(
            (self.as_translation() * 1000.0, self.as_rotation().as_euler("xyz", degrees=True)),
            axis=-1,
        )

    def as_matrix(self) -> NDArray[np.float64]:
        """Return a new 4x4 homogeneous transformation matrix."""
        return self._transform.as_matrix()

    def inverse(self) -> Transform:
        """Return the inverse transform, mapping destination coordinates to source."""
        return self.from_transform(self._transform.inv())

    def __matmul__(self, other: Transform) -> Transform:
        """Compose transforms: apply ``other`` first, then ``self``."""
        if not isinstance(other, Transform):
            return NotImplemented
        return self.from_transform(self._transform * other._transform)

    def apply(self, points: ArrayLike) -> NDArray[np.float64]:
        """Transform a point (3,) or points (..., 3), including translation.

        Each trailing triple is one point; an (N, 3) array stores points in rows.
        """
        return self._transform.apply(points)

    def apply_vectors(self, vectors: ArrayLike) -> NDArray[np.float64]:
        """Rotate free 3D vectors of shape (..., 3), without translation.

        This operation is for directions and other free vectors, not 6D twists
        or wrenches, which require an adjoint transformation.
        """
        return self.as_rotation().apply(vectors)

    def plot(
        self,
        ax: Axes3D | None = None,
        label: str = "",
        length: float = 1.0,
        *,
        show: bool | None = None,
    ) -> Axes3D:
        """Draw this frame's x/y/z axes in red/green/blue and return the 3D Axes.

        Translation and arrow length are in meters. All frames plotted on the
        same Axes must use the same parent frame. Limits include arrow endpoints
        and existing data, with equal spatial scales along all three axes.

        A new figure uses a Z-up orthographic view. By default, show the figure
        only when this method creates it. Use show=False to compose frames or
        save a figure without opening a window. Matplotlib is imported lazily.
        """
        import matplotlib.pyplot as plt

        created_fig = ax is None
        if created_fig:
            fig = plt.figure(layout="constrained")
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=25, azim=-60, vertical_axis="z")
            ax.set_proj_type("ortho")

        had_data = ax.has_data()
        origin = self.as_translation()
        directions = self.as_rotation().as_matrix().T * length
        for axis, color, direction in zip("xyz", "rgb", directions):
            ax.quiver(
                *origin,
                *direction,
                color=color,
                pivot="tail",
                normalize=False,
                arrow_length_ratio=0.15,
                linewidth=1.5,
                label=f"{label}_{axis}" if label else None,
            )
        if label:
            ax.text(*origin, label, fontsize=10, color="k")

        points = np.vstack((origin, origin + directions))
        ax.set_autoscale_on(True)
        ax.margins(0.1)
        ax.auto_scale_xyz(*points.T, had_data=had_data)
        ax.set(xlabel="X [m]", ylabel="Y [m]", zlabel="Z [m]")
        ax.set_box_aspect((1, 1, 1))
        ax.set_aspect("equal", adjustable="datalim")

        if show is None:
            show = created_fig
        if show:
            plt.show()
        return ax

    def __repr__(self) -> str:
        return f"Transform(rotation={self.as_rotation()!r}, translation={self.as_translation()!r})"


if __name__ == "__main__":
    from mujoco_lab.utils.logger import Logger

    logger = Logger()
    H1 = Transform.from_pose_mrad([0, 0, 0, 0, 0, 0])
    H2 = Transform.from_pose_mmdeg([500, 200, 400, 20, -15, 35])
    logger.info(f"Tool pose (m/rad): {H2.as_pose_mrad()}")
    logger.info(f"Tool pose (mm/deg): {H2.as_mmdeg()}")
    logger.info(f"Tool pose (m, wxyz): {H2.as_xyzquat()}")
    ax = H1.plot(label="world", length=0.3, show=False)
    H2.plot(ax=ax, label="tool", length=0.3, show=True)
