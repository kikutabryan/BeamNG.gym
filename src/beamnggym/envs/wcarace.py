from typing import Any, List, Dict, Tuple
import gymnasium as gym
import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Damage, Electrics
from beamngpy.misc.quat import angle_to_quat
from shapely import affinity
from shapely.geometry import LinearRing, LineString, Point, Polygon


class WCARaceGeometry(gym.Env):
    def __init__(self, host="localhost", port=25252):
        # Simulation settings
        self.sim_rate = 50  # simulation steps per second
        self.action_rate = 5  # actions per second
        self.steps = self.sim_rate // self.action_rate

        # LiDAR settings
        self.lidar_info = {
            "start_angle": -np.pi / 2,
            "end_angle": np.pi / 2,
            "num_rays": 31,
            "max_dist": 500,
        }

        # Reward and penalty settings
        self.completion_reward = 50
        self.progress_multiplier = 0.2
        self.time_penalty = -0.1

        # Ensure penalties are greater than the maximum time penalty
        self.max_damage = 100
        self.damage_penalty = -50
        self.out_of_bounds_penalty = -50
        self.wrong_way_penalty = -50
        self.too_long_penalty = -50

        # Define action and observation spaces
        self.action_space = self._action_space()
        self.observation_space = self._observation_space()

        # Track and vehicle setup
        self.start_proj = 10
        self.final_proj = 4320
        self.spine = None
        self.l_edge = None
        self.r_edge = None
        self.polygon = None
        self.last_vehicle_pos = None
        self.last_remaining_dist = None
        self.last_steering = None

        # Remaining time
        self.remaining_time = 5 * 60 * (self.start_proj / self.final_proj) + 10

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

    def __del__(self):
        self.bng.close()

    def _setup_vehicle(self) -> Vehicle:
        vehicle = Vehicle(
            "racecar",
            model="sunburst",
            license="BEAMNG",
            color="red",
            part_config="vehicles/sunburst/hillclimb.pc",
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
                    -np.pi,
                    -np.pi,
                    -np.inf,
                    0,
                    0,
                    0,
                    -1.0,
                    -1,
                    0,  # vehicle_info
                    *[0] * self.lidar_info["num_rays"],
                    -np.inf,
                    0,  # track_info
                ]
            ),
            high=np.array(
                [
                    np.pi,
                    np.pi,
                    np.inf,
                    np.inf,
                    1.0,
                    1.0,
                    1.0,
                    8,
                    np.inf,  # vehicle_info
                    *[self.lidar_info["max_dist"]] * self.lidar_info["num_rays"],
                    np.inf,
                    np.inf,  # track_info
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
        self.remaining_time = 5 * 60 * (self.start_proj / self.final_proj) + 10
        self.last_remaining_dist = None
        self.last_vehicle_pos = None
        self.vehicle.control(throttle=0.0, brake=0.0, steering=0.0)
        self.bng.scenario.restart()
        self.bng.control.step(30)
        self.bng.control.pause()
        self.vehicle.set_shift_mode("realistic_automatic")
        self.vehicle.control(gear=2)
        self.vehicle.sensors.poll()
        return self._get_obs(), {}

    def _update(self, action):
        action = [*np.clip(action, -1, 1)]
        action = [float(act) for act in action]
        throttle, steering = action[0], action[1]
        brake = -throttle if throttle < 0 else 0
        throttle = max(throttle, 0)
        self.vehicle.control(steering=steering, throttle=throttle, brake=brake)
        self.bng.step(self.steps, wait=True)
        self.remaining_time -= 1 / self.action_rate

    def _get_obs(self) -> np.ndarray:
        self.vehicle.sensors.poll()
        electrics = self.vehicle.sensors["electrics"]
        vehicle_state = self.vehicle.state
        vehicle_pos = Point(*vehicle_state["pos"])
        vehicle_dir = vehicle_state["dir"]

        # Flatten the observation
        return np.array(
            [
                self._get_vehicle_rel_angle(vehicle_pos, vehicle_dir),
                self._get_vehicle_elev_angle(vehicle_dir),
                self._get_vehicle_velocity(vehicle_pos),
                electrics["rpm"],
                electrics["throttle"],
                electrics["brake"],
                electrics["steering"],
                electrics["gear_index"],
                electrics["wheelspeed"],
                *self._get_lidar_distances(
                    vehicle_pos, np.arctan2(vehicle_dir[1], vehicle_dir[0])
                ),
                self.remaining_time,
                self._get_remaining_track_dist(vehicle_pos),
            ],
            dtype=np.float32,
        )

    def _get_vehicle_rel_angle(self, vehicle_pos, vehicle_dir) -> float:
        spine_proj_dist = self.spine.project(vehicle_pos)
        spine_proj = self.spine.interpolate(spine_proj_dist)
        spine_end = self.spine.interpolate(spine_proj_dist + 1)
        spine_angle = np.arctan2(spine_end.y - spine_proj.y, spine_end.x - spine_proj.x)
        vehicle_angle = np.arctan2(vehicle_dir[1], vehicle_dir[0])
        rel_angle = (vehicle_angle - spine_angle + np.pi) % (2 * np.pi) - np.pi
        return rel_angle

    def _get_vehicle_elev_angle(self, vehicle_dir) -> float:
        return np.arctan2(vehicle_dir[2], np.linalg.norm(vehicle_dir[0:2]))

    def _get_vehicle_velocity(self, vehicle_pos) -> float:
        if not self.last_vehicle_pos:
            self.last_vehicle_pos = vehicle_pos
            return 0
        delta_time = 1 / self.action_rate
        velocity_vector = np.array(
            [
                (vehicle_pos.x - self.last_vehicle_pos.x) / delta_time,
                (vehicle_pos.y - self.last_vehicle_pos.y) / delta_time,
                (vehicle_pos.z - self.last_vehicle_pos.z) / delta_time,
            ]
        )
        self.last_vehicle_pos = vehicle_pos
        return np.linalg.norm(velocity_vector)

    def _get_remaining_track_dist(self, vehicle_pos) -> float:
        curr_proj_dist = self.spine.project(vehicle_pos) - self.start_proj
        if curr_proj_dist < 0:
            curr_proj_dist += self.spine.length
        return self.spine.length - curr_proj_dist

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
                self.lidar_info["start_angle"],
                self.lidar_info["end_angle"],
                self.lidar_info["num_rays"],
            )
            + vehicle_angle
        )
        return np.array(
            [
                self._ray_distance(
                    vehicle_pos, angle, tracks, self.lidar_info["max_dist"]
                )
                for angle in angles
            ],
            dtype=np.float32,
        )

    def _get_reward(self, observation: np.ndarray) -> Tuple[float, bool, bool]:
        # Maximum damage
        if self.vehicle.sensors["damage"]["damage"] > self.max_damage:
            print("reset: damage exceeded")
            return self.damage_penalty, True, False

        # Out of bounds
        vehicle_pos = Point(*self.vehicle.state["pos"])
        if not self.polygon.contains(vehicle_pos):
            print("reset: out of bounds")
            return self.out_of_bounds_penalty, True, False

        # Wrong way
        remaining_dist = observation[-1]
        if (
            self.last_remaining_dist is not None
            and remaining_dist > self.last_remaining_dist + 0.5
        ):
            print("reset: wrong way")
            return self.wrong_way_penalty, True, False

        # Progress reward
        progress_reward = 0
        if (
            self.last_remaining_dist is not None
            and remaining_dist < self.last_remaining_dist
        ):
            progress_reward = (
                self.last_remaining_dist - remaining_dist
            ) * self.progress_multiplier

        if self.last_remaining_dist is not None:
            self.last_remaining_dist = min(self.last_remaining_dist, remaining_dist)
        else:
            self.last_remaining_dist = remaining_dist

        # Race complete
        if remaining_dist < 20:
            print("race complete")

            # Increment start_proj but do not exceed final_proj
            self.start_proj = min(self.start_proj + 1, self.final_proj)

            return self.completion_reward, True, False

        # Too long
        if self.remaining_time <= 0:
            print("reset: too long")
            return self.too_long_penalty, False, True

        # Total reward
        total_reward = progress_reward + self.time_penalty

        return total_reward, False, False
