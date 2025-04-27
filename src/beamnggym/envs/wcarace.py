from typing import Any, List, Dict, Tuple
import gymnasium as gym
import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Damage, Electrics
from beamngpy.misc.quat import angle_to_quat
from shapely.geometry import LinearRing, LineString, Point, Polygon
from dataclasses import dataclass, field
import random
from collections import deque
import time
import os


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
    min_speed: float


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
    def __init__(
        self,
        host="localhost",
        port=25252,
        start_lap_percent=0.05,
        final_lap_percent=1.0,
        lap_percent_increment=0.05,
        learn_starts=0,
        real_time=False,
        vehicle_index=0,
        randomize_start=False,
        enable_logging=False,
        log_path=None,
    ):
        # Progressive difficulty settings
        self.final_lap_percent = final_lap_percent
        self.lap_percent_increment = lap_percent_increment
        self.current_lap_percent = start_lap_percent
        self.learn_starts = learn_starts
        self.total_steps = 0
        self.randomize_start = randomize_start
        self.enable_logging = enable_logging
        self.log_path = log_path
        self.episode_time = 0

        # Simulation settings
        self.sim_rate = 20  # simulation steps per second
        self.action_rate = 5  # actions per second
        self.steps = self.sim_rate // self.action_rate
        self.real_time = real_time

        # Define default start position and rotation
        self.default_pos = (394.5, -247.925, 145.25)
        self.default_rot = angle_to_quat((0, 0, 90))

        # Spawn settings
        self.random_position_range = (
            2.5  # Maximum random offset in x,y coordinates (meters)
        )
        self.random_angle_range = (
            20.0  # Maximum random offset in rotation angle (degrees)
        )
        self.spawn_height_offset = 0.6  # Height offset from track surface (meters)

        # Define available vehicles
        self.vehicles = [
            {
                "name": "ETK 800",
                "model": "etk800",
                "part_config": "vehicles/etk800/856x_310d_A.pc",
                "color": "white",
            },
            {
                "name": "Bruckell LeGran",
                "model": "legran",
                "part_config": "vehicles/legran/sport_se_v6_awd_facelift_A.pc",
                "color": "white",
            },
            {
                "name": "Bruckell Bastion",
                "model": "bastion",
                "part_config": "vehicles/bastion/sport_gt_A.pc",
                "color": "black",
            },
            {
                "name": "Gavril H-Series",
                "model": "van",
                "part_config": "vehicles/van/h15_ext_vanster.pc",
                "color": "white",
            },
            {
                "name": "Gavril T-Series",
                "model": "us_semi",
                "part_config": "vehicles/us_semi/tc82s_cargobox.pc",
                "color": "white",
            },
            {
                "name": "Wentward DT40L",
                "model": "citybus",
                "part_config": "vehicles/citybus/highway.pc",
                "color": "white",
            },
            {
                "name": "ETK K-Series",
                "model": "etkc",
                "part_config": "vehicles/etkc/kc6x_trackday_A.pc",
                "color": "white",
            },
        ]

        # Validate and set vehicle index
        if vehicle_index < 0 or vehicle_index >= len(self.vehicles):
            print(f"Invalid vehicle index {vehicle_index}, using default vehicle 0")
            self.vehicle_index = 0
        else:
            self.vehicle_index = vehicle_index

        # LiDAR settings
        self.lidar_info = LidarSettings(
            start_angle=-np.deg2rad(135),
            end_angle=np.deg2rad(135),
            num_rays=31,
            max_dist=500,
        )

        # Reward and penalty settings
        self.reward_settings = RewardSettings(
            max_damage=1000,
            damage_penalty=-200,
            out_of_bounds_penalty=-200,
            wrong_way_penalty=-200,
            too_long_penalty=-200,
            speed_factor=0.05,
            steer_factor=0.5,
            min_speed=0,
        )

        # Track and vehicle setup
        self.max_lap_time = 5 * 60
        self.spine = None
        self.l_edge = None
        self.r_edge = None
        self.polygon = None
        self.max_steer_rate = 0.2

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

        # Initialize simulation
        self._initialize_simulation()

        # Logging setup
        if self.enable_logging:
            self._log_data()

    def __del__(self):
        if hasattr(self, "log_file") and self.log_file:
            self.log_file.close()
        self.bng.close()

    def close(self):
        if hasattr(self, "log_file") and self.log_file:
            self.log_file.close()
        if hasattr(self, "bng"):
            self.bng.close()

    def _initialize_simulation(self):
        # Set simulation speed
        self._configure_simulation()

        # Set states
        self.state.remaining_time = self.max_lap_time * self.current_lap_percent + 30
        self.state.curr_dist = 0  # Reset current distance

        # BeamNG config
        self.vehicle.control(throttle=0.0, brake=0.0, steering=0.0)
        self.bng.scenario.restart()
        self.vehicle.recover()
        self.bng.control.pause()
        self.vehicle.set_shift_mode("realistic_automatic")
        self.vehicle.control(gear=2)
        self.vehicle.sensors.poll()

    def _setup_vehicle(self) -> Vehicle:
        vehicle_config = self.vehicles[self.vehicle_index]
        print(f"Using vehicle: {vehicle_config['name']}")

        vehicle = Vehicle(
            "racecar",
            model=vehicle_config["model"],
            license="BEAMNG",
            color=vehicle_config["color"],
            part_config=vehicle_config["part_config"],
        )
        vehicle.sensors.attach("electrics", Electrics())
        vehicle.sensors.attach("damage", Damage())
        return vehicle

    def _setup_scenario(self) -> Scenario:
        scenario = Scenario("west_coast_usa", "wca_race_geometry_v0")
        scenario.add_vehicle(
            self.vehicle,
            pos=self.default_pos,
            rot_quat=self.default_rot,
        )
        scenario.make(self.bng)
        self.bng.scenario.load(scenario)
        return scenario

    def _configure_simulation(self):
        self.bng.control.queue_lua_command("be:setPhysicsDeterministic(true)")
        if self.real_time:
            self.bng.control.queue_lua_command(f"Engine.setFPSLimiter({self.sim_rate})")
            self.bng.control.queue_lua_command("Engine.setFPSLimiterEnabled(true)")
        else:
            self.bng.control.queue_lua_command("Engine.setFPSLimiterEnabled(false)")

    def _action_space(self):
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

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
                ],
                dtype=np.float32,
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
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
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

        # Log track geometry if enabled
        if self.enable_logging:
            # Create log directory if it doesn't exist
            if self.log_path:
                os.makedirs(self.log_path, exist_ok=True)
                spine_file = os.path.join(self.log_path, "track_spine.csv")
                left_file = os.path.join(self.log_path, "track_left_edge.csv")
                right_file = os.path.join(self.log_path, "track_right_edge.csv")
            else:
                spine_file = "track_spine.csv"
                left_file = "track_left_edge.csv"
                right_file = "track_right_edge.csv"

            # Log spine points
            with open(spine_file, "w") as f:
                f.write("x,y,z\n")
                for point in s_vtx:
                    f.write(f"{point[0]},{point[1]},{point[2]}\n")

            # Log left edge points
            with open(left_file, "w") as f:
                f.write("x,y,z\n")
                for point in l_vtx:
                    f.write(f"{point[0]},{point[1]},{point[2]}\n")

            # Log right edge points
            with open(right_file, "w") as f:
                f.write("x,y,z\n")
                for point in r_vtx:
                    f.write(f"{point[0]},{point[1]},{point[2]}\n")

    def _log_data(self):
        if not self.enable_logging:
            return

        vehicle_state = self.vehicle.state
        vehicle_pos = vehicle_state["pos"]
        vehicle_dir = vehicle_state["dir"]
        vehicle_angle = np.arctan2(vehicle_dir[1], vehicle_dir[0])

        # Create log directory if it doesn't exist
        if self.log_path:
            os.makedirs(self.log_path, exist_ok=True)
            log_file = os.path.join(self.log_path, "race_track_data.csv")
            episode_log_file = os.path.join(self.log_path, "episode_results.csv")
        else:
            log_file = "race_track_data.csv"
            episode_log_file = "episode_results.csv"

        # Initialize log files if they don't exist
        if not hasattr(self, "log_file"):
            self.log_file = open(log_file, "w")
            self.log_file.write("step,x,y,z,angle,speed,spine_speed\n")
            self.episode_log_file = open(episode_log_file, "w")
            self.episode_log_file.write("steps,lap_time,lap_percentage\n")

        self.log_file.write(
            f"{self.total_steps},{vehicle_pos[0]},{vehicle_pos[1]},{vehicle_pos[2]},"
            f"{vehicle_angle},{self.state.vehicle_velocity},{self.state.spine_speed}\n"
        )
        self.log_file.flush()

    def _log_episode_results(self, terminated, truncated):
        if not self.enable_logging:
            return

        if terminated or truncated:
            lap_percentage = self.state.curr_dist / self.spine.length
            self.episode_log_file.write(
                f"{self.total_steps},{self.episode_time},{lap_percentage}\n"
            )
            self.episode_log_file.flush()

    def step(self, action):
        self._update(action)
        self.total_steps += 1  # Increment total steps counter
        self.episode_time += 1 / self.action_rate  # Increment episode time by time step
        observation = self._get_obs()
        reward, terminated, truncated = self._get_reward(observation)
        self._log_data()  # Log data after each step
        self._log_episode_results(
            terminated, truncated
        )  # Log episode results if episode ended
        print(f"Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}")
        return observation, reward, terminated, truncated, {}

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)
        self.episode_time = 0  # Reset episode time

        if self.randomize_start:
            # Get random point on spine
            self.state.last_proj = random.uniform(0, self.spine.length)
            self.state.last_vehicle_pos = self.spine.interpolate(self.state.last_proj)

            # Get angle of spine at point
            start_angle = self._get_spine_angle(self.state.last_vehicle_pos)

            # Random offset
            random_pos_offset = (
                random.uniform(-self.random_position_range, self.random_position_range),
                random.uniform(-self.random_position_range, self.random_position_range),
                0,
            )
            random_angle_offest = random.uniform(
                -self.random_angle_range, self.random_angle_range
            )

            # Randomize starting position
            start_pos = (
                self.state.last_vehicle_pos.x + random_pos_offset[0],
                self.state.last_vehicle_pos.y + random_pos_offset[1],
                self.state.last_vehicle_pos.z + self.spawn_height_offset,
            )
            rot_quat = angle_to_quat(
                (0, 0, np.rad2deg(-start_angle) - 90 + random_angle_offest)
            )
        else:
            # Use the fixed starting position from _setup_scenario
            start_pos = self.default_pos
            rot_quat = self.default_rot

            # Initialize self.state.last_proj and self.state.last_vehicle_pos
            self.state.last_vehicle_pos = Point(*start_pos)
            self.state.last_proj = self.spine.project(self.state.last_vehicle_pos)

        # Teleport the vehicle to the starting position
        self.vehicle.teleport(pos=start_pos, rot_quat=rot_quat)
        self._initialize_simulation()
        return self._get_obs(), {}

    def _update(self, action):
        action = [*np.clip(action, -1, 1)]
        action = [float(act) for act in action]
        throttle, self.state.steering_rate = (
            float(action[0]),
            float(action[1] * self.max_steer_rate),
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

        # Ensure other values stay within observation bounds
        elev_angle = np.clip(self.state.vehicle_elev_angle, -np.pi, np.pi)
        brake = np.clip(self.state.brake, 0, 1.0)
        gear_index = np.clip(self.state.gear_index, -1, 8)

        # Ensure LiDAR values are within bounds
        lidar_distances = np.clip(
            self.state.lidar_distances, 0, self.lidar_info.max_dist
        )

        # Flatten the observation
        return np.array(
            [
                self.state.vehicle_rel_angle,
                elev_angle,
                self.state.vehicle_velocity,
                self.state.rpm,
                self.state.throttle,
                brake,
                self.state.steering,
                gear_index,
                self.state.wheelspeed,
                *lidar_distances,
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
        # Crash penalty
        # Only used if the maximum damage is exceeded, or out of bounds, or wrong way
        angle = observation[0]
        velocity = observation[2]
        crash_penalty = 5 * (
            -np.abs(angle)
            / (np.pi / 2)
            * self.reward_settings.speed_factor
            * velocity**2
        )

        # Maximum damage
        if self.vehicle.sensors["damage"]["damage"] > self.reward_settings.max_damage:
            print("reset: damage exceeded")
            return self.reward_settings.damage_penalty + crash_penalty, True, False

        # Out of bounds
        vehicle_pos = Point(*self.vehicle.state["pos"])
        if not self.polygon.contains(vehicle_pos):
            print("reset: out of bounds")
            return (
                self.reward_settings.out_of_bounds_penalty + crash_penalty,
                True,
                False,
            )

        # Wrong way
        spine_speed = observation[-1]
        if spine_speed < -5:
            print("reset: wrong way")
            return self.reward_settings.wrong_way_penalty + crash_penalty, True, False

        # Too long
        if self.state.remaining_time <= 0:
            print("reset: too long")
            return self.reward_settings.too_long_penalty, False, True

        # Spine speed reward
        # Quadratic function utilized as linear would result in the same reward for
        # a given track regardless of the car's spine speed.
        speed_reward = (
            self.reward_settings.speed_factor * np.sign(spine_speed) * spine_speed**2
        )

        # Steering rate punishment
        # Punishment is linear wrt steering rate, and proportional to speed reward.
        # Designed to not exceed speed reward so that there is always a reward for
        # progressing along the track, stops converging at local optimum.
        steer_punishment = (
            -np.abs(speed_reward)
            * self.reward_settings.steer_factor
            * (self.state.steering_rate / self.max_steer_rate)
        )

        # Sum of rewards and punishment
        total_reward = speed_reward + steer_punishment

        # Race complete
        if self.state.curr_dist > self.spine.length * self.current_lap_percent:
            print(f"race complete at {self.current_lap_percent:.3f} lap percent")

            # Increase lap percent for progressive difficulty only after learn_starts
            if (
                self.current_lap_percent < self.final_lap_percent
                and self.total_steps > self.learn_starts
            ):
                self.current_lap_percent = min(
                    self.current_lap_percent + self.lap_percent_increment,
                    self.final_lap_percent,
                )
                print(f"increasing to {self.current_lap_percent:.3f} lap percent")
            else:
                if self.total_steps <= self.learn_starts:
                    print(
                        f"still in learning phase ({self.total_steps}/{self.learn_starts}), not increasing difficulty"
                    )

            return total_reward, True, False

        return total_reward, False, False
