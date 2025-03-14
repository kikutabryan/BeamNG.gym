from typing import Any, List, Dict, Tuple
import gymnasium as gym
import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Damage, Electrics
from beamngpy.misc.quat import angle_to_quat
from shapely.geometry import LinearRing, LineString, Point, Polygon
from dataclasses import dataclass, field
import random


@dataclass
class LidarSettings:
    start_angle: float
    end_angle: float
    num_rays: int
    max_dist: float


@dataclass
class RewardSettings:
    max_damage: float
    damage_penalty: float
    out_of_bounds_penalty: float
    wrong_way_penalty: float
    too_long_penalty: float
    speed_factor: float
    steer_factor: float
    brake_factor: float


@dataclass
class EnvironmentState:
    curr_dist: float = 0
    last_proj: float = None
    last_vehicle_pos: Point = None
    steering_rate: float = 0
    last_steering: float = 0
    remaining_time: float = 0
    spine_speed: float = 0
    vehicle_rel_angle: float = 0
    vehicle_elev_angle: float = 0
    vehicle_velocity: float = 0
    rpm: float = 0
    throttle: float = 0
    brake: float = 0
    steering: float = 0
    gear_index: int = 0
    wheelspeed: float = 0
    lidar_distances: np.ndarray = field(
        default_factory=lambda: np.zeros(271, dtype=np.float32)
    )


class WCARaceGeometry(gym.Env):
    def __init__(self, host="localhost", port=25252, lap_percent=1):
        # Simulation settings
        self.sim_rate = 50  # simulation steps per second
        self.action_rate = 5  # actions per second
        self.steps = self.sim_rate // self.action_rate

        # LiDAR settings
        self.lidar_info = LidarSettings(
            start_angle=-np.deg2rad(135),
            end_angle=np.deg2rad(135),
            num_rays=101,
            max_dist=500,
        )

        # Reward and penalty settings
        self.reward_settings = RewardSettings(
            max_damage=100,
            damage_penalty=-50,
            out_of_bounds_penalty=-50,
            wrong_way_penalty=-50,
            too_long_penalty=-50,
            speed_factor=0.01,
            steer_factor=100,
            brake_factor=10,
        )

        # Track and vehicle setup
        self.lap_percent = lap_percent
        self.max_lap_time = 5 * 60
        self.spine = None
        self.l_edge = None
        self.r_edge = None
        self.polygon = None
        self.steering_rate_factor = 0.2

        # Initialize state
        self.state = EnvironmentState()

        # Define action and observation spaces
        self.action_space = self._action_space()
        self.observation_space = self._observation_space()

        # Initialize BeamNG
        self.bng = BeamNGpy(host, port)
        self.bng.open()
        self.vehicle = self._setup_vehicle()
        self.scenario = self._setup_scenario()
        self._build_racetrack()

        # Start and configure simulation
        self.bng.scenario.start()
        self.bng.control.pause()
        self._configure_simulation()

        # Initialize simulation
        self._initialize_simulation()

    def __del__(self):
        self.bng.close()

    def _initialize_simulation(self):
        # Set states
        self.state.remaining_time = self.max_lap_time * self.lap_percent + 10
        self.state.curr_dist = 0  # Reset current distance

        # BeamNG config
        self.vehicle.control(throttle=0.0, brake=0.0, steering=0.0)
        self.bng.scenario.restart()
        self.bng.control.step(30)
        self.bng.control.pause()
        self.vehicle.set_shift_mode("realistic_automatic")
        self.vehicle.control(gear=2)
        self.vehicle.sensors.poll()

    def _setup_vehicle(self) -> Vehicle:
        vehicle = Vehicle(
            "racecar",
            model="etk800",
            license="BEAMNG",
            color="white",
            part_config="vehicles/etk800/856x_310d_A.pc",
        )
        vehicle.sensors.attach("electrics", Electrics())
        vehicle.sensors.attach("damage", Damage())
        return vehicle

    def _setup_scenario(self) -> Scenario:
        scenario = Scenario("west_coast_usa", "wca_race_geometry_v0")
        scenario.add_vehicle(
            self.vehicle,
            pos=(394.5, -247.925, 145.25),
            rot_quat=angle_to_quat((0, 0, 90)),
        )
        scenario.make(self.bng)
        self.bng.scenario.load(scenario)
        return scenario

    def _configure_simulation(self):
        self.bng.settings.set_deterministic(self.sim_rate)
        self.bng.control.queue_lua_command("be:setPhysicsDeterministic(true)")
        self.bng.control.queue_lua_command(
            'be:executeJS("bngApi.engine.setFrameLimiter(false)")'
        )

    def _action_space(self):
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64)

    def _observation_space(self):
        # Flatten the observation space
        return gym.spaces.Box(
            low=np.array(
                [
                    -np.pi,  # Relative angle
                    -np.pi,  # Elevation angle
                    -np.inf,  # Vehicle velocity
                    0,  # RPM
                    0,  # Throttle
                    0,  # Brake
                    -1.0,  # Steering
                    -1,  # Gear index
                    0,  # Wheelspeed
                    *[0] * self.lidar_info.num_rays,  # LiDAR
                    -np.inf,  # Spine speed
                ]
            ),
            high=np.array(
                [
                    np.pi,  # Relative angle
                    np.pi,  # Elevation angle
                    np.inf,  # Vehicle velocity
                    np.inf,  # RPM
                    1.0,  # Throttle
                    1.0,  # Brake
                    1.0,  # Steering
                    8,  # Gear index
                    np.inf,  # Wheelspeed
                    *[self.lidar_info.max_dist] * self.lidar_info.num_rays,  # LiDAR
                    np.inf,  # Spine speed
                ]
            ),
            dtype=np.float64,
        )

    def _build_racetrack(self):
        roads = self.bng.scenario.get_roads()
        RACETRACK_PID = "064a5d03-61d1-4ed7-9136-905b40928f01"
        track_id, _ = next(
            filter(lambda road: road[1]["persistentId"] == RACETRACK_PID, roads.items())
        )
        track = self.bng.scenario.get_road_edges(track_id)
        l_vtx, s_vtx, r_vtx = [], [], []
        for edges in track:
            r_vtx.append(edges["right"])
            s_vtx.append(edges["middle"])
            l_vtx.append(edges["left"])

        self.spine = LinearRing(s_vtx)
        self.r_edge = LinearRing(r_vtx)
        self.l_edge = LinearRing(l_vtx)
        self.polygon = Polygon([v[0:2] for v in l_vtx], holes=[[v[0:2] for v in r_vtx]])

    def step(self, action):
        self._update(action)
        observation = self._get_obs()
        reward, terminated, truncated = self._get_reward(observation)
        print(f"Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}")
        return observation, reward, terminated, truncated, {}

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)
        # Get random point on spine
        self.state.last_proj = random.uniform(0, self.spine.length)
        self.state.last_vehicle_pos = self.spine.interpolate(self.state.last_proj)

        # Get angle of spine at point
        start_angle = self._get_spine_angle(self.state.last_vehicle_pos)

        # Random offset
        random_offset = (random.uniform(-2, 2), random.uniform(-2, 2), 0)

        # Randomize starting position
        start_pos = (
            self.state.last_vehicle_pos.x + random_offset[0],
            self.state.last_vehicle_pos.y + random_offset[1],
            self.state.last_vehicle_pos.z + 0.5,
        )
        self.vehicle.teleport(
            pos=start_pos, rot_quat=angle_to_quat((0, 0, np.rad2deg(-start_angle) - 90))
        )
        self._initialize_simulation()
        return self._get_obs(), {}

    def _update(self, action):
        action = [*np.clip(action, -1, 1)]
        action = [float(act) for act in action]
        throttle, self.state.steering_rate = (
            action[0],
            action[1] * self.steering_rate_factor,
        )
        brake = -throttle if throttle < 0 else 0
        throttle = max(throttle, 0)

        # Compute steering angle given a steering rate
        steering = float(
            np.clip(self.state.last_steering + self.state.steering_rate, -1, 1)
        )
        self.state.last_steering = steering

        self.vehicle.control(steering=steering, throttle=throttle, brake=brake)
        self.bng.step(self.steps, wait=True)
        self.state.remaining_time -= 1 / self.action_rate

    def _get_obs(self) -> np.ndarray:
        self.vehicle.sensors.poll()
        electrics = self.vehicle.sensors["electrics"]
        vehicle_state = self.vehicle.state
        vehicle_pos = Point(*vehicle_state["pos"])
        vehicle_dir = vehicle_state["dir"]

        # Update state with current observation values
        self.state.vehicle_rel_angle = self._get_vehicle_rel_angle(
            vehicle_pos, vehicle_dir
        )
        self.state.vehicle_elev_angle = self._get_vehicle_elev_angle(vehicle_dir)
        self.state.vehicle_velocity = self._get_vehicle_velocity(vehicle_pos)
        self.state.rpm = electrics["rpm"]
        self.state.throttle = electrics["throttle"]
        self.state.brake = electrics["brake"]
        self.state.steering = electrics["steering"]
        self.state.gear_index = electrics["gear_index"]
        self.state.wheelspeed = electrics["wheelspeed"]
        self.state.lidar_distances = self._get_lidar_distances(
            vehicle_pos, np.arctan2(vehicle_dir[1], vehicle_dir[0])
        )
        self.state.spine_speed = self._get_spine_speed(vehicle_pos)

        # Flatten the observation
        return np.array(
            [
                self.state.vehicle_rel_angle,
                self.state.vehicle_elev_angle,
                self.state.vehicle_velocity,
                self.state.rpm,
                self.state.throttle,
                self.state.brake,
                self.state.steering,
                self.state.gear_index,
                self.state.wheelspeed,
                *self.state.lidar_distances,
                self.state.spine_speed,
            ],
            dtype=np.float32,
        )

    def _get_vehicle_rel_angle(self, vehicle_pos, vehicle_dir) -> float:
        spine_angle = self._get_spine_angle(vehicle_pos)
        vehicle_angle = np.arctan2(vehicle_dir[1], vehicle_dir[0])
        rel_angle = (vehicle_angle - spine_angle + np.pi) % (2 * np.pi) - np.pi
        return rel_angle

    def _get_spine_angle(self, position) -> float:
        spine_proj_dist = self.spine.project(position)
        spine_fwd = self.spine.interpolate((spine_proj_dist + 1) % self.spine.length)
        spine_bwd = self.spine.interpolate((spine_proj_dist - 1) % self.spine.length)
        spine_angle = np.arctan2(spine_fwd.y - spine_bwd.y, spine_fwd.x - spine_bwd.x)
        return spine_angle

    def _get_vehicle_elev_angle(self, vehicle_dir) -> float:
        return np.arctan2(vehicle_dir[2], np.linalg.norm(vehicle_dir[0:2]))

    def _get_vehicle_velocity(self, vehicle_pos) -> float:
        if not self.state.last_vehicle_pos:
            self.state.last_vehicle_pos = vehicle_pos
            return 0
        delta_time = 1 / self.action_rate
        velocity_vector = np.array(
            [
                (vehicle_pos.x - self.state.last_vehicle_pos.x) / delta_time,
                (vehicle_pos.y - self.state.last_vehicle_pos.y) / delta_time,
                (vehicle_pos.z - self.state.last_vehicle_pos.z) / delta_time,
            ]
        )
        self.state.last_vehicle_pos = vehicle_pos
        return np.linalg.norm(velocity_vector)

    def _get_spine_dist(self, vehicle_pos) -> float:
        curr_proj = self.spine.project(vehicle_pos)
        dist = curr_proj - self.state.last_proj
        self.state.last_proj = curr_proj

        # Check if distance is valid
        if abs(dist) > 0.9 * self.spine.length:
            if dist < 0:
                # Passed end point
                dist += self.spine.length
            elif dist > 0:
                # Went behind start
                dist -= self.spine.length

        # Update distance covered
        self.state.curr_dist += dist

        return dist

    def _get_spine_speed(self, vehicle_pos) -> float:
        dist = self._get_spine_dist(vehicle_pos)
        time_step = 1 / self.action_rate
        spine_speed = dist / time_step
        return spine_speed

    def _ray_distance(
        self,
        point: Point,
        angle: float,
        linear_rings: Tuple[LinearRing, LinearRing],
        max_dist: float,
    ) -> float:
        point_2d = Point(point.x, point.y)
        ray_end = Point(
            point_2d.x + max_dist * np.cos(angle), point_2d.y + max_dist * np.sin(angle)
        )
        ray = LineString([point_2d, ray_end])
        closest_dist = max_dist
        for ring in linear_rings:
            intersection = ring.intersection(ray)
            if not intersection.is_empty:
                closest_point = (
                    min(intersection.geoms, key=lambda p: point_2d.distance(p))
                    if intersection.geom_type == "MultiPoint"
                    else intersection
                )
                closest_dist = min(closest_dist, point_2d.distance(closest_point))
        return closest_dist

    def _get_lidar_distances(
        self, vehicle_pos: Point, vehicle_angle: float
    ) -> np.ndarray:
        tracks = (self.l_edge, self.r_edge)
        angles = (
            np.linspace(
                self.lidar_info.start_angle,
                self.lidar_info.end_angle,
                self.lidar_info.num_rays,
            )
            + vehicle_angle
        )
        return np.array(
            [
                self._ray_distance(vehicle_pos, angle, tracks, self.lidar_info.max_dist)
                for angle in angles
            ],
            dtype=np.float32,
        )

    def _get_reward(self, observation: np.ndarray) -> Tuple[float, bool, bool]:
        # Maximum damage
        if self.vehicle.sensors["damage"]["damage"] > self.reward_settings.max_damage:
            print("reset: damage exceeded")
            return self.reward_settings.damage_penalty, True, False

        # Out of bounds
        vehicle_pos = Point(*self.vehicle.state["pos"])
        if not self.polygon.contains(vehicle_pos):
            print("reset: out of bounds")
            return self.reward_settings.out_of_bounds_penalty, True, False

        # Wrong way
        spine_speed = observation[-1]
        if spine_speed < -5:
            print("reset: wrong way")
            return self.reward_settings.wrong_way_penalty, True, False

        # Too long
        if self.state.remaining_time <= 0:
            print("reset: too long")
            return self.reward_settings.too_long_penalty, False, True

        # Spine speed reward
        speed_reward = (
            self.reward_settings.speed_factor * np.sign(spine_speed) * spine_speed**2
        )

        # Steering rate punishment
        steer_punishment = (
            -self.reward_settings.steer_factor * self.state.steering_rate**2
        )

        # Sum of rewards and punishment
        total_reward = speed_reward + steer_punishment

        # Race complete
        if self.state.curr_dist > self.spine.length * self.lap_percent:
            print("race complete")
            return total_reward, True, False

        return total_reward, False, False
