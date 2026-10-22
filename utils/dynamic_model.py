from __future__ import annotations

import numpy as np

def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64
    )

def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if norm < 1e-15:
        raise ValueError("near-zero vector cannot be normalized")
    return v / norm

def _cross(v: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Cross product of a 3-vector with each column of a (3, n) matrix."""
    return _skew(v) @ M

def _cross_columns(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Column-wise cross product for two (3, n) matrices."""
    return np.cross(A, B, axis=0)

class UCUDynamics:
    """Parameterized complete 6-UCU dynamics."""

    def __init__(
        self,
        rP: float,  # 上平台半径
        dP: float,
        rB: float,
        dB: float,
        neutral_height: float, # 平台高度
        platform_mass: float, # 平台质量
        cylinder_mass: float, # 上杆质量
        piston_mass: float, # 下杆质量
        platform_inertia: np.ndarray, # 上平台惯量
        cylinder_inertia: np.ndarray, # 上杆惯量
        piston_inertia: np.ndarray, # 下杆惯量
        platform_com: np.ndarray, # 上平台惯量
        cylinder_com_distance: float,
        piston_com_distance: float,
        cylinder_fixed_length: float,
        gravity: np.ndarray = np.array([0.0, 0.0, -9.81]),
        top_joint: str = "U",
        hinge_damping: float = 0.0,
    ) -> None:
        self.rP = float(rP)
        self.dP = float(dP)
        self.rB = float(rB)
        self.dB = float(dB)
        self.neutral_height = float(neutral_height)
        self.mC = float(platform_mass)
        self.m1 = np.broadcast_to(cylinder_mass, (6,)).astype(np.float64)
        self.m2 = np.broadcast_to(piston_mass, (6,)).astype(np.float64)
        self.IC = np.asarray(platform_inertia, dtype=np.float64).reshape(3, 3)
        self.I1 = np.asarray(cylinder_inertia, dtype=np.float64).reshape(3, 3)
        self.I2 = np.asarray(piston_inertia, dtype=np.float64).reshape(3, 3)
        self.c_com = np.asarray(platform_com, dtype=np.float64).reshape(3)
        self.c1_dist = np.broadcast_to(cylinder_com_distance, (6,)).astype(np.float64)
        self.c2_dist = np.broadcast_to(piston_com_distance, (6,)).astype(np.float64)
        self.l2_fixed = float(cylinder_fixed_length)
        self.g = np.asarray(gravity, dtype=np.float64).reshape(3)
        self.top_joint = top_joint
        self.hinge_damping = float(hinge_damping)
        self._geometry()

    def _geometry(self) -> None:
        eta_P = np.pi / 3.0 - np.arcsin(self.dP / (2.0 * self.rP))
        eta_B = np.arcsin(self.dB / (2.0 * self.rB))

        theta_P = np.array(
            [
                eta_P,
                2.0 * np.pi / 3.0 - eta_P,
                2.0 * np.pi / 3.0 + eta_P,
                4.0 * np.pi / 3.0 - eta_P,
                4.0 * np.pi / 3.0 + eta_P,
                -eta_P,
            ]
        )
        theta_B = np.array(
            [
                eta_B,
                2.0 * np.pi / 3.0 - eta_B,
                2.0 * np.pi / 3.0 + eta_B,
                4.0 * np.pi / 3.0 - eta_B,
                4.0 * np.pi / 3.0 + eta_B,
                -eta_B,
            ]
        )

        self.p_local = np.vstack([self.rP * np.cos(theta_P), self.rP * np.sin(theta_P), np.zeros(6)])
        self.b = np.vstack([self.rB * np.cos(theta_B), self.rB * np.sin(theta_B), np.zeros(6)])

        # Base U-joint axis s0 is perpendicular to the short side and points
        # outward from the base-circle centre.
        mid_B = np.array(
            [0.0, 2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0, 4.0 * np.pi / 3.0, 0.0]
        )
        self.s0 = np.vstack(
            [np.cos(mid_B), np.sin(mid_B), np.zeros(6)]
        )

        # Platform U-joint axis s3 is perpendicular to the platform short side
        # and points outward from the platform-circle centre.
        mid_P = np.array(
            [np.pi / 3.0, np.pi / 3.0, np.pi, np.pi, 5.0 * np.pi / 3.0, 5.0 * np.pi / 3.0]
        )
        self.s3 = np.vstack(
            [np.cos(mid_P), np.sin(mid_P), np.zeros(6)]
        )

    def inverse_dynamics(
        self,
        position: np.ndarray,
        rotation: np.ndarray,
        velocity: np.ndarray,
        angular_velocity: np.ndarray,
        acceleration: np.ndarray,
        angular_acceleration: np.ndarray,
        external_force: np.ndarray | None = None,
        external_moment: np.ndarray | None = None,
    ) -> np.ndarray:
        
        position = np.asarray(position, dtype=np.float64).reshape(3)
        R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        v = np.asarray(velocity, dtype=np.float64).reshape(3)
        omega = np.asarray(angular_velocity, dtype=np.float64).reshape(3)
        a = np.asarray(acceleration, dtype=np.float64).reshape(3)
        alpha = np.asarray(angular_acceleration, dtype=np.float64).reshape(3)

        # 外力与外力矩
        F_e = np.zeros(3) if external_force is None else np.asarray(external_force, dtype=np.float64).reshape(3)
        M_e = np.zeros(3) if external_moment is None else np.asarray(external_moment, dtype=np.float64).reshape(3)

        p_i = R @ self.p_local
        h = position[:, None] + p_i - self.b
        lengths = np.linalg.norm(h, axis=0)
        n = h / lengths

        s1 = np.cross(self.s0, n, axis=0)
        s1 /= np.linalg.norm(s1, axis=0)
        s2 = np.cross(self.s3, n, axis=0)
        s2 /= np.linalg.norm(s2, axis=0)

        v_P = v[:, None] + _cross(omega, p_i)
        a_P = a[:, None] + _cross(alpha, p_i) + _cross(omega, _cross(omega, p_i))
        l_dot = np.einsum("ij,ij->j", n, v_P)

        y_hat = _cross_columns(n, s1)
        s0_cross_n = _cross_columns(self.s0, n)
        denom_omega0 = lengths * np.linalg.norm(s0_cross_n, axis=0)
        omega0 = np.einsum("ij,ij->j", v_P, s1) / denom_omega0
        omega1 = np.einsum("ij,ij->j", v_P, y_hat) / (-lengths)
        omega1p = omega0[None, :] * self.s0 + omega1[None, :] * s1

        s2_cross_s3 = _cross_columns(s2, self.s3)
        s2s3_dot_n = np.einsum("ij,ij->j", s2_cross_s3, n)
        omega_ni = np.einsum("ij,ij->j", s2_cross_s3, omega[:, None] - omega1p) / s2s3_dot_n
        omega2p = omega1p + omega_ni[None, :] * n

        l1 = lengths - self.l2_fixed
        u = (
            a_P
            - 2.0 * _cross_columns(omega1p, l_dot[None, :] * n)
            - _cross_columns(omega1p, _cross_columns(omega1p, l1[None, :] * n))
            - _cross_columns(omega2p, _cross_columns(omega2p, self.l2_fixed * n))
        )
        h_i = (omega0 * omega1)[None, :] * _cross_columns(self.s0, s1) + omega_ni[None, :] * _cross_columns(
            omega1p, n
        )

        s0_cross_s1 = _cross_columns(self.s0, s1)
        alpha0 = np.einsum(
            "ij,ij->j",
            u
            - (omega0 * omega1 * l1)[None, :] * _cross_columns(s0_cross_s1, n)
            - _cross_columns(h_i, self.l2_fixed * n),
            s1,
        ) / denom_omega0
        alpha1 = np.einsum(
            "ij,ij->j",
            (omega0 * omega1 * l1)[None, :] * _cross_columns(s0_cross_s1, n)
            + _cross_columns(h_i, self.l2_fixed * n)
            - u,
            y_hat,
        ) / lengths
        alpha1p = (
            alpha0[None, :] * self.s0
            + alpha1[None, :] * s1
            + (omega0 * omega1)[None, :] * s0_cross_s1
        )

        omega2 = np.einsum("ij,ij->j", s2, omega[:, None] - omega1p)
        omega3 = np.einsum("ij,ij->j", self.s3, omega[:, None] - omega2p)
        k_i = (
            alpha0[None, :] * self.s0
            + alpha1[None, :] * s1
            + _cross_columns(omega0[None, :] * self.s0, omega1[None, :] * s1)
            + _cross_columns(omega1p, omega_ni[None, :] * n)
            + _cross_columns(omega2p, omega2[None, :] * s2)
            + _cross_columns(omega2p + omega2[None, :] * s2, omega3[None, :] * self.s3)
        )
        alpha_ni = np.einsum("ij,ij->j", s2_cross_s3, alpha[:, None] - k_i) / s2s3_dot_n
        alpha2p = alpha1p + alpha_ni[None, :] * n + omega_ni[None, :] * _cross_columns(omega1p, n)

        c1 = self.c1_dist * n
        c2 = -self.c2_dist * n
        a1 = _cross_columns(alpha1p, c1) + _cross_columns(omega1p, _cross_columns(omega1p, c1))
        a2 = a_P + _cross_columns(alpha2p, c2) + _cross_columns(omega2p, _cross_columns(omega2p, c2))

        F1 = self.m1 * (self.g[:, None] - a1)
        F2 = self.m2 * (self.g[:, None] - a2)
        M1 = -self.I1 @ alpha1p - _cross_columns(omega1p, self.I1 @ omega1p) + self.hinge_damping * omega1p
        M2 = -self.I2 @ alpha2p - _cross_columns(omega2p, self.I2 @ omega2p) + self.hinge_damping * omega2p
        w = _cross_columns(lengths[None, :] * n + c2, F2) + _cross_columns(c1, F1) + M1 + M2

        m_dir_B = _cross_columns(s1, self.s0)
        m_dir_B /= np.linalg.norm(m_dir_B, axis=0)
        m_dir_P = _cross_columns(s2, self.s3)
        m_dir_P /= np.linalg.norm(m_dir_P, axis=0)

        M_B = -np.einsum("ij,ij->j", _cross_columns(c1, F1) + M1, n) / np.einsum("ij,ij->j", m_dir_B, n)
        M_P = (
            -np.einsum("ij,ij->j", w, n)
            - M_B * np.einsum("ij,ij->j", m_dir_B, n)
        ) / np.einsum("ij,ij->j", m_dir_P, n)
        if self.top_joint == "S":
            M_P[:] = 0.0
        F_n = _cross_columns(n, w + M_P[None, :] * m_dir_P + M_B[None, :] * m_dir_B) / lengths

        a_C = a + _cross(alpha, self.c_com) + _cross(omega, _cross(omega, self.c_com))
        F_C = self.mC * (self.g - a_C) + F_e
        M_C = -self.IC @ alpha - _cross(omega, self.IC @ omega) + M_e

        N_k = np.concatenate(
            [
                F_C - np.sum(F_n, axis=1),
                _cross(self.c_com, F_C) + M_C
                - np.sum(_cross_columns(p_i, F_n) + M_P[None, :] * m_dir_P, axis=1),
            ]
        )

        J = np.hstack([n.T, _cross_columns(p_i, n).T])
        f_a = np.linalg.solve(J.T, N_k)
        tau = -(f_a + np.einsum("ij,ij->j", n, F2))
        return tau

    @classmethod
    def from_mujoco(
        cls,
        model,
        data,
        control_site: str = "circle_center",
        platform_body: str = "top",
        platform_sites: tuple[str, ...] = ("con1", "con2", "con3", "con4", "con5", "con6"),
        base_bodies: tuple[str, ...] = (
            "legdown00",
            "legdown10",
            "legdown01",
            "legdown11",
            "legdown02",
            "legdown12",
        ),
        piston_bodies: tuple[str, ...] = (
            "legup00",
            "legup10",
            "legup01",
            "legup11",
            "legup02",
            "legup12",
        ),
        top_joint: str = "U",
        piston_lower_end_distance: float | None = None,
        hinge_damping: float = 0.0,
    ) -> "UCUDynamics":

        import mujoco
        from scipy.spatial.transform import Rotation as R

        mujoco.mj_forward(model, data)

        top_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, platform_body)
        control_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, control_site)
        control_local = np.asarray(model.site_pos[control_id])
        platform_com = np.asarray(model.body_ipos[top_id]) - control_local

        p_local = np.column_stack(
            [
                np.asarray(model.site_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s)])
                - control_local
                for s in platform_sites
            ]
        )

        b = np.zeros((3, 6))
        s0 = np.zeros((3, 6))
        s3 = np.zeros((3, 6))

        base_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in base_bodies
        ]
        for i, body_id in enumerate(base_ids):
            hinge_jid = next(
                jid
                for jid in range(model.njnt)
                if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE
                and model.jnt_bodyid[jid] == body_id
            )
            b[:, i] = data.xanchor[hinge_jid]
            s0[:, i] = data.xaxis[hinge_jid]

        for pair in ((0, 1), (2, 3), (4, 5)):
            mid = _unit(p_local[:2, pair[0]] + p_local[:2, pair[1]])
            s3[0:2, pair[0]] = mid
            s3[0:2, pair[1]] = mid

        def world_inertia(body_id: int) -> np.ndarray:
            R_body = np.asarray(data.xmat[body_id]).reshape(3, 3)
            q = np.asarray(model.body_iquat[body_id])[[1, 2, 3, 0]]
            R_inertial_body = R.from_quat(q).as_matrix()
            R_inertial_world = R_body @ R_inertial_body
            return (
                R_inertial_world
                @ np.diag(model.body_inertia[body_id])
                @ R_inertial_world.T
            )

        piston_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in piston_bodies
        ]
        cylinder_mass = np.array([model.body_mass[i] for i in base_ids])
        piston_mass = np.array([model.body_mass[i] for i in piston_ids])
        cylinder_inertia = np.mean([world_inertia(i) for i in base_ids], axis=0)
        piston_inertia = np.mean([world_inertia(i) for i in piston_ids], axis=0)

        def body_com_world(body_id: int) -> np.ndarray:
            R_body = np.asarray(data.xmat[body_id]).reshape(3, 3)
            return data.xpos[body_id] + R_body @ model.body_ipos[body_id]

        p0 = np.asarray(data.site_xpos[control_id])
        p_i_world = p0[:, None] + np.asarray(data.xmat[top_id]).reshape(3, 3) @ p_local
        n_i = (p_i_world - b) / np.linalg.norm(p_i_world - b, axis=0)

        if piston_lower_end_distance is None:
            slide_distances = []
            for i, piston_id in enumerate(piston_ids):
                slide_jid = next(
                    jid
                    for jid in range(model.njnt)
                    if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE
                    and model.jnt_bodyid[jid] == piston_id
                )
                slide_distances.append(
                    np.dot(data.xanchor[slide_jid] - b[:, i], n_i[:, i])
                )
            piston_lower_end_distance = float(np.mean(slide_distances))

        c1_dist = np.array(
            [np.dot(body_com_world(base_ids[i]) - b[:, i], n_i[:, i]) for i in range(6)]
        )
        c2_dist = np.array(
            [-np.dot(body_com_world(piston_ids[i]) - p_i_world[:, i], n_i[:, i]) for i in range(6)]
        )
        neutral_height = float(data.site_xpos[control_id][2])

        obj = cls(
            rP=0.2, dP=0.2, rB=0.35, dB=0.30, neutral_height=neutral_height,
            platform_mass=model.body_mass[top_id],
            cylinder_mass=cylinder_mass,
            piston_mass=piston_mass,
            platform_inertia=world_inertia(top_id),
            cylinder_inertia=cylinder_inertia,
            piston_inertia=piston_inertia,
            platform_com=platform_com,
            cylinder_com_distance=c1_dist,
            piston_com_distance=c2_dist,
            cylinder_fixed_length=piston_lower_end_distance,
            gravity=np.asarray(model.opt.gravity),
            top_joint=top_joint,
            hinge_damping=hinge_damping,
        )
        obj.p_local = p_local
        obj.b = b
        obj.s0 = s0
        obj.s3 = s3
        return obj
