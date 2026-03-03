import warnings

import numpy as np
from scipy.spatial.transform import Rotation

# Executed on import!
try:
    import vispy
    from vispy import scene, app, io
    from vispy.scene.cameras import TurntableCamera
    from vispy.scene.visuals import InstancedMesh
    from vispy.visuals.filters import InstancedShadingFilter
    from vispy.geometry import MeshData
    from vispy.geometry.generation import create_box, create_cylinder
    from vispy.visuals.transforms import STTransform
    vispy.use("glfw", gl="gl+")
except OSError:
    warnings.warn("Unable to initialize OpenGL. 3D Graphics will not work.")


VELOCITY_DRAG = 0.1
ANGULAR_VELOCITY_DRAG = 0.1
GRAV = np.array([0, 0, -9.81])
DT = 0.01
STATE_DIM = 18
ACTION_DIM = 4


def _quadratic_drag(vector):
    return -vector * np.linalg.norm(vector, axis=-1)[:, None]


class Quadrotor:
    """Batched quadrotor dynamics simulator.

    Simulates N quadrotors in parallel. The state includes 3D position,
    velocity, rotation matrix, and angular velocity in body frame.

    Actions: 4 normalized motor thrusts in range [0, 1], one for each motor.

    Physical parameters are based on the Crazyflie 2.0 quadrotor.
    """

    def __init__(self, n_envs, seed=0):
        """Initializes the Quadrotor environment.

        Args:
            n_envs (int): Number of parallel environments to maintain.
            seed (int): Random seed for reproducibility.
        """
        self.n_envs = n_envs

        # State variables.
        self.position = None  # (N, 3) world frame position.
        self.velocity = None  # (N, 3) world frame velocity.
        self.rotation = None  # (N, 3, 3) rotation matrix from body to world.
        self.angular_velocity = None  # (N, 3) body frame angular velocity.

        # Crazyflie 2.0 physical constants.
        self.mass = 0.027
        self.inertia = np.array([1.6e-5, 1.6e-5, 2.9e-5])
        self.arm_length = 0.046
        self.max_thrust_per_motor = 2 * 9.81 * self.mass / 4 # thurst-to-weight 2.
        self.thrust_to_torque = 0.006

        # Mixing matrix converts motor forces to thrust and torque.
        l = self.arm_length / np.sqrt(2)
        k = self.thrust_to_torque
        self.mixing_matrix = np.array([
            [1,    1,    1,    1],  # Total thrust in body-z direction.
            [l,   -l,   -l,    l],  # Torque about x-axis (roll).
            [l,    l,   -l,   -l],  # Torque about y-axis (pitch).
            [-k,   k,   -k,    k],  # Torque about z-axis (yaw).
        ])

        self.rng = np.random.default_rng(seed=seed)
        self.vis = None

    def reseed(self, seed):
        """Reseeds the random number generator for reproducibility.

        Args:
            seed (int): New random seed.
        """
        self.rng = np.random.default_rng(seed=seed)

    def step(self, actions):
        """Advances all environments by one time step using the given actions.

        Args:
            actions (array(N, 4)): Normalized motor thrusts.

        Returns:
            states (array(N, 18)): Flattened state vector with components:
                [position (3), velocity (3), rotation (9), angular_velocity (3)].
            rewards (array(N)): Rewards for each environment.
        """
        if actions.shape != (self.n_envs, 4):
            raise ValueError(f"actions must have shape ({self.n_envs}, 4), got {actions.shape}")

        motor_forces = self.max_thrust_per_motor * np.clip(actions, 0.0, 1.0)
        thrust_torques = motor_forces @ self.mixing_matrix.T
        total_thrust = thrust_torques[:, 0]
        torques = thrust_torques[:, 1:4]

        # Ignoring gyroscopic term for simplicity.
        angular_accel = (
            torques / self.inertia[None, :]
            + ANGULAR_VELOCITY_DRAG * _quadratic_drag(self.angular_velocity)
        )

        thrust = self.rotation[:, :, 2] * total_thrust[:, None]
        accel = (
            thrust / self.mass
            + GRAV
            + VELOCITY_DRAG * _quadratic_drag(self.velocity)
        )

        reward = self.reward(self.position, self.velocity, self.angular_velocity)

        # Update state following rigid body dynamics.
        self.velocity = self.velocity + DT * accel
        self.position = self.position + DT * self.velocity
        self.angular_velocity = self.angular_velocity + DT * angular_accel
        dR = Rotation.from_rotvec(self.angular_velocity * DT).as_matrix()
        self.rotation = self.rotation @ dR

        return self.get_state(), reward

    def reward(self, pos, vel, angvel):
        """Computes reward for hovering with zero angular velocity.

        Args:
            states (array(N, S)): Array of states.

        Returns: rewards (array(N)).
        """
        pos_error = np.linalg.norm(pos, axis=-1) ** 2
        vel_error = np.linalg.norm(vel, axis=-1) ** 2
        angvel_error = np.linalg.norm(angvel, axis=-1) ** 2
        return np.exp(-10*pos_error) + 0.1*np.exp(-10*vel_error) + np.exp(-10*angvel_error)

    def reset(self, randomize=False):
        """Resets environments specified by mask to initial states.

        Args:
            randomize (bool): If True, sample random states within bounds.
                If False, initialize at hover condition.

        Returns:
            states (array(N, 18)): Flattened state vectors for all environments.
        """
        def randvec_box(limit):
            return self.rng.uniform(-limit, limit, size=(self.n_envs, 3))

        self.position = randomize * randvec_box(0.5)
        self.velocity = randomize * randvec_box(0.2)
        self.angular_velocity = randomize * randvec_box(1.0)

        angle_axes = randomize * randvec_box(0.2)
        self.rotation = Rotation.from_rotvec(angle_axes).as_matrix()

        return self.get_state()

    def get_state(self):
        """Returns the current state as a flattened vector.

        Returns:
            state (array(N, 18)): Flattened state with components:
                [position (3), velocity (3), rotation (9), angular_velocity (3)]
        """
        return np.concatenate([
            self.position,
            self.velocity,
            self.rotation.reshape(self.n_envs, 9),
            self.angular_velocity,
        ], axis=1)

    def render(self):
        N = min(self.n_envs, 100)
        if self.vis is None:
            self.vis = Visualizer(N)
        self.vis.update(self.position[:N], self.rotation[:N])


class Visualizer:
    def __init__(self, count):

        self.canvas = scene.SceneCanvas(
            keys="interactive", size=(1024, 768), show=True, resizable=True
        )

        plane_color = 0.25 * np.ones((1, 3))
        bg_color = 0.1 * np.ones((1, 3))

        view = self.canvas.central_widget.add_view()
        view.bgcolor = bg_color
        view.camera = TurntableCamera(
            fov=60.0, elevation=20.0, azimuth=-60.0, center=(0, 0, 0), distance=1.5,
        )

        axis = scene.visuals.XYZAxis(parent=view.scene)
        ground = scene.visuals.Plane(
            20.0, 20.0, direction="+z", color=plane_color, parent=view.scene,
        )
        for visual in [axis, ground]:
            visual.transform = STTransform(translate=(0, 0, -1))
        goal = scene.visuals.Sphere(0.03, color=(1, 1, 1), parent=view.scene, shading="flat")

        instance_positions = np.zeros((count, 3))
        instance_transforms = np.tile(np.eye(3), (count, 1, 1))
        verts, faces = _build_quadrotor_mesh(0.046)

        mesh = InstancedMesh(
            vertices=verts,
            faces=faces,
            color=np.ones(3),
            instance_positions=instance_positions,
            instance_transforms=instance_transforms,
            parent=view.scene
        )
        shading_filter = InstancedShadingFilter(shading='flat')
        mesh.attach(shading_filter)

        self.mesh = mesh

    def update(self, positions, rotations):
        error = np.maximum(np.abs(positions) - 0.1, 0)
        # if we have error in X, we want to show the quad as red.
        # therefore, we need to subtract green and blue.
        err_to_color = -np.ones((3, 3)) + np.eye(3)
        colors = np.clip(1 + error @ err_to_color.T, 0, 1)
        self.mesh.instance_positions = positions
        self.mesh.instance_transforms = rotations
        self.mesh.instance_colors = colors
        self.mesh.update()
        self.canvas.app.process_events()


def _build_quadrotor_mesh(arm_length):
    prop_radius = arm_length * 0.5
    body_size   = arm_length * 0.5
    arm_width   = arm_length * 0.1
    prop_height = prop_radius / 6
    body_height = body_size / 4
    arm_height  = arm_width / 2

    all_verts, all_faces = [], []

    def merge(v, f, rot=np.eye(3), trans=np.zeros(3)):
        v = v @ rot + trans
        f = f + sum(len(a) for a in all_verts)
        all_verts.append(v)
        all_faces.append(f)

    # VisPy's "create*" APIs are weirdly inconsistent
    def _create_box(*args):
        v, f = create_box(*args)[:2]
        return v["position"], f

    def _create_cylinder(*args, **kwargs):
        cyl = create_cylinder(*args, **kwargs)
        return cyl.get_vertices(), cyl.get_faces()

    merge(*_create_box(body_size, body_height, body_size), trans=[0, 0, -arm_height/2])

    for a in np.radians([45, 135, 225, 315]):
        rot = Rotation.from_rotvec([0, 0, a]).as_matrix()
        trans = rot @ [0, arm_length, 0]
        merge(*_create_box(arm_length, arm_height, arm_width), rot=rot, trans=trans/2)
        merge(
            *_create_cylinder(1, 20, [prop_radius, prop_radius], prop_height),
            trans=trans + [0, 0, arm_height],
        )

    return np.vstack(all_verts), np.vstack(all_faces)
