#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Line, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils
from vmas.simulator.controllers.velocity_controller import VelocityController

from scenarios.hit_the_switch.load_config import load_scenario_config
from scenarios.hit_the_switch.language import LanguageUnit, load_decoder, load_sequence_model, load_task_data

FIND_SWITCH = 0
FIND_GOAL = 1

class HitSwitchScenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        load_scenario_config(kwargs, self)
        assert 1 <= self.n_passages <= 20

        world = World(batch_dim, device, x_semidim=self.x_semidim, y_semidim=self.y_semidim)
        
        self._add_passages(world)
        self._add_switch(world)
        self._init_language_unit(world)
        self._add_agents_and_goals(world)

        return world
    
    def _add_passages(self, world):
        self.landmarks = []
        num_passages = int((2 * world.x_semidim + 2 * self.agent_radius) // self.passage_length)
        if self.break_all_wall:
            self.n_passages = num_passages
        for i in range(num_passages):
            passage = Landmark(
                name=f"passage {i}",
                collide=True,
                movable=False,
                shape=Box(length=self.passage_length, width=self.passage_width),
                color=Color.RED,
                collision_filter=lambda e: not isinstance(e.shape, Box),
            )
            self.landmarks.append(passage)
            world.add_landmark(passage)

    def _add_switch(self, world):
        self.switch = Landmark(
            name="switch",
            collide=False,
            movable=False,
            shape=Box(length=self.switch_length, width=self.switch_width),
            color=Color.YELLOW,
            collision_filter=lambda e: not isinstance(e.shape, Box),
        )
        world.add_landmark(self.switch)

    def _init_language_unit(self, world: World):
        if self.use_rnn:
            load_decoder(model_path=self.decoder_model_path, embedding_size=self.embedding_size, device=world.device)
            load_sequence_model(model_path=self.sequence_model_path, embedding_size=self.embedding_size, event_size=self.event_size, state_size=self.state_size, device=world.device)
            load_task_data(json_path=self.data_json_path, device=world.device)

        self.language_unit = LanguageUnit(
            batch_size=world.batch_dim,
            embedding_size=self.embedding_size,
            use_embedding_ratio=self.use_embedding_ratio,
            device=world.device,
        )
        self.visualize_semidims = False

    def _add_agents_and_goals(self, world: World):
        self.goals = []
        self.team_hit_switch = torch.zeros((world.batch_dim,), dtype=torch.bool, device=world.device)

        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}",
                shape=Sphere(self.agent_radius),
                v_range=self.agent_v_range,
                f_range=self.agent_f_range,
                u_range=self.agent_u_range,
            )
            if self.use_velocity_controller:
                agent.controller = VelocityController(
                    agent, world, ctrl_params=(2.0, 6.0, 0.002), pid_form="standard"
                )

            goal = Landmark(
                name=f"goal {i}",
                collide=False,
                shape=Sphere(radius=self.agent_radius),
                color=Color.LIGHT_GREEN,
            )
            agent.goal = goal
            self.goals.append(goal)
            world.add_landmark(goal)

            self._set_initial_goal_coords(agent, world)
            agent.hit_switch = torch.zeros((world.batch_dim,), dtype=torch.bool, device=world.device)

            world.add_agent(agent)

    def _set_initial_goal_coords(self, agent: Agent, world: World):
        
        agent.goal_coords = torch.zeros((world.batch_dim, 2), dtype=torch.float32, device=world.device)
        
        find_switch_indices = torch.where(self.language_unit.states == FIND_SWITCH)[0]
        find_goal_indices = torch.where(self.language_unit.states == FIND_GOAL)[0]
        
        agent.goal_coords[find_switch_indices] = self.switch.state.pos[find_switch_indices]
        agent.goal_coords[find_goal_indices] = agent.goal.state.pos[find_goal_indices]

    def reset_world_at(self, env_index: int = None):
        central_agent_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1 + (3 * self.agent_radius + self.agent_spacing),
                    1 - (3 * self.agent_radius + self.agent_spacing),
                ),
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1 + (3 * self.agent_radius + self.agent_spacing),
                    -(3 * self.agent_radius + self.agent_spacing)
                    - self.passage_width / 2,
                ),
            ],
            dim=1,
        )
        central_goal_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1 + (3 * self.agent_radius + self.agent_spacing),
                    1 - (3 * self.agent_radius + self.agent_spacing),
                ),
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    (3 * self.agent_radius + self.agent_spacing)
                    + self.passage_width / 2,
                    1 - (3 * self.agent_radius + self.agent_spacing),
                ),
            ],
            dim=1,
        )

        order = torch.randperm(self.n_agents).tolist()
        agents = [self.world.agents[i] for i in order]
        goals = [self.goals[i] for i in order]
        for i, goal in enumerate(goals):
            if i == self.n_agents - 1:
                goal.set_pos(
                    central_goal_pos,
                    batch_index=env_index,
                )
            else:
                goal.set_pos(
                    central_goal_pos
                    + torch.tensor(
                        [
                            [
                                (
                                    0.0
                                    if i % 2
                                    else (
                                        self.agent_spacing
                                        if i == 0
                                        else -self.agent_spacing
                                    )
                                ),
                                (
                                    0.0
                                    if not i % 2
                                    else (
                                        self.agent_spacing
                                        if i == 1
                                        else -self.agent_spacing
                                    )
                                ),
                            ],
                        ],
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
        for i, agent in enumerate(agents):
            
            # Goal coordinates are set to zero
            agent.hit_switch[env_index].zero_() if env_index is not None else agent.hit_switch.zero_()

            if i == self.n_agents - 1:
                agent.set_pos(
                    central_agent_pos,
                    batch_index=env_index,
                )
            else:
                agent.set_pos(
                    central_agent_pos
                    + torch.tensor(
                        [
                            [
                                (
                                    0.0
                                    if i % 2
                                    else (
                                        self.agent_spacing
                                        if i == 0
                                        else -self.agent_spacing
                                    )
                                ),
                                (
                                    0.0
                                    if not i % 2
                                    else (
                                        self.agent_spacing
                                        if i == 1
                                        else -self.agent_spacing
                                    )
                                ),
                            ],
                        ],
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
            if env_index is None:
                agent.global_shaping = (
                    torch.linalg.vector_norm(
                        agent.state.pos - agent.goal_coords, dim=1
                    )
                    * self.shaping_factor
                )
            else:
                agent.global_shaping[env_index] = (
                    torch.linalg.vector_norm(
                        agent.state.pos[env_index] - agent.goal_coords[env_index]
                    )
                    * self.shaping_factor
                )

        order = torch.randperm(len(self.landmarks)).tolist()
        passages = [self.landmarks[i] for i in order]
        for i, passage in enumerate(passages):
            # if not passage.collide:
            #     passage.is_rendering[:] = False
            passage.set_pos(
                torch.tensor(
                    [
                        -1
                        - self.agent_radius
                        + self.passage_length / 2
                        + self.passage_length * i,
                        0.0,
                    ],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
        self.landmark_poses = [landmark.state.pos.clone() for landmark in self.landmarks]

        # Add switch at random position in lower half of the world
        # Ensure switch is not on the passage
        # Randomly generate x and y coordinates for the switch
        rx = torch.rand(
            (self.world.batch_dim, 1),
            device=self.world.device,
            dtype=torch.float32,
        )
        ry = torch.rand(
            (self.world.batch_dim, 1),
            device=self.world.device,
            dtype=torch.float32,
        )
        rand_x = rx * (1 - self.switch_length/2) * 2 - (1 - self.switch_length/2)
        rand_y = torch.max(-ry * (1 - self.switch_width / 2 + self.agent_radius) - self.passage_length / 2, -1 * torch.ones_like(ry) + self.switch_width / 2 - self.agent_radius)

        if env_index is None:
            pos = torch.cat((rand_x, rand_y), dim=1)
        else:
            pos = torch.tensor(
                [rand_x[env_index], rand_y[env_index]],
                dtype=torch.float32,
                device=self.world.device,
            )

        self.switch.set_pos(
            pos,
            batch_index=env_index,
        )
        
        self.team_hit_switch[env_index].zero_() if env_index is not None else self.team_hit_switch.zero_()
        
        if self.use_rnn:
            self.language_unit.reset_env(env_index)
            self.language_unit.sample_dataset(env_index)

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if self.shared_reward:
            if is_first:
                self.rew = torch.zeros(
                    self.world.batch_dim,
                    device=self.world.device,
                    dtype=torch.float32,
                )
                for a in self.world.agents:
                    dist_to_goal = torch.linalg.vector_norm(
                        a.state.pos - a.goal_coords, dim=1
                    )
                    agent_shaping = dist_to_goal * self.shaping_factor
                    self.rew += a.global_shaping - agent_shaping
                    a.global_shaping = agent_shaping
        else:
            self.rew = torch.zeros(
                self.world.batch_dim,
                device=self.world.device,
                dtype=torch.float32,
            )
            dist_to_goal = torch.linalg.vector_norm(
                agent.state.pos - agent.goal_coords, dim=1
            )
            agent_shaping = dist_to_goal * self.shaping_factor
            self.rew += agent.global_shaping - agent_shaping
            agent.global_shaping = agent_shaping

        if agent.collide:
            for a in self.world.agents:
                if a != agent:
                    self.rew[self.world.is_overlapping(a, agent)] -= 10
            for landmark in self.landmarks:
                if landmark.collide:
                    self.rew[self.world.is_overlapping(agent, landmark)] -= 10
            hit_switch_mask = self.world.is_overlapping(agent, self.switch).bool()
            # only the *first-time* hits (overlap & never hit before)
            new_hits = hit_switch_mask & (~self.team_hit_switch)
            self.rew[new_hits] += 10
            
            # Update hit switch status
            agent.hit_switch |= hit_switch_mask
            self.team_hit_switch |= hit_switch_mask
        
        # Despawn passage if switch is hit
        if self.team_hit_switch.any():
            indices = torch.where(self.team_hit_switch)[0]
            self.language_unit.states[indices] = FIND_GOAL
            passages = self.landmarks[: self.n_passages]
            for passage in passages:
                passage.state.pos[indices] = self._get_outside_pos(None)[indices]

        return self.rew * 0.01

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        obs_dict = {}
        obs_components = []
        
        if not self.break_all_wall:
            passage_obs = []
            for pose in self.landmark_poses[: self.n_passages]:
                passage_obs.append(pose - agent.state.pos)
            passage_obs = torch.stack(passage_obs, dim=1).view(self.world.batch_dim, -1)
            obs_components.append(passage_obs)
 
        obs_dict["pos"] = agent.state.pos
        obs_dict["vel"] = agent.state.vel
        #obs_dict["sentence_embedding"] = self.language_unit.observe_subtask_embeddings()
        obs_dict["sentence_embedding"] = torch.empty((self.world.batch_dim,0), device=self.world.device)  # Placeholder for LLM embeddings
        obs_components.append(agent.goal.state.pos - agent.state.pos)
        obs_components.append(self.switch.state.pos - agent.state.pos)
        obs_components.append((self.language_unit.states == FIND_GOAL).float().unsqueeze(1))  # Convert bool to float for observation
        obs_dict["obs"] = torch.cat(obs_components, dim=-1)

        return obs_dict
        
    def _get_outside_pos(self, env_index):
        """Get a position far outside the environment to hide entities."""
        return torch.empty(
            (
                (1, self.world.dim_p)
                if env_index is not None
                else (self.world.batch_dim, self.world.dim_p)
            ),
            device=self.world.device,
        ).uniform_(-1000 * self.world.x_semidim, -10 * self.world.x_semidim)   
    
    def pre_step(self):
        # Here we will implement the rnn logic to switch the sequence state
        
        find_switch_indices = torch.where(self.language_unit.states == FIND_SWITCH)[0]
        hit_switch_indices = torch.where(self.language_unit.states == FIND_GOAL)[0]
        
        for agent in self.world.agents:
            # Update goal coordinates based on the new state
            agent.goal_coords[find_switch_indices] = self.switch.state.pos[find_switch_indices]
            agent.goal_coords[hit_switch_indices] = agent.goal.state.pos[hit_switch_indices]

    def done(self):
        return torch.all(
            torch.stack(
                [
                    torch.linalg.vector_norm(a.state.pos - a.goal.state.pos, dim=1)
                    <= a.shape.radius / 2
                    for a in self.world.agents
                ],
                dim=1,
            ),
            dim=1,
        )

    def extra_render(self, env_index: int = 0):
        from vmas.simulator import rendering

        geoms = []
        for i in range(4):
            geom = Line(length=2 + self.agent_radius * 2).get_geometry()
            xform = rendering.Transform()
            geom.add_attr(xform)

            xform.set_translation(
                (
                    0.0
                    if i % 2
                    else (
                        self.world.x_semidim + self.agent_radius
                        if i == 0
                        else -self.world.x_semidim - self.agent_radius
                    )
                ),
                (
                    0.0
                    if not i % 2
                    else (
                        self.world.x_semidim + self.agent_radius
                        if i == 1
                        else -self.world.x_semidim - self.agent_radius
                    )
                ),
            )
            xform.set_rotation(torch.pi / 2 if not i % 2 else 0.0)
            color = Color.BLACK.value
            if isinstance(color, torch.Tensor) and len(color.shape) > 1:
                color = color[env_index]
            geom.set_color(*color)
            geoms.append(geom)
        
        for i, agent1 in enumerate(self.world.agents):
            for j, agent2 in enumerate(self.world.agents):
                if j <= i:
                    continue
                if self.world.get_distance(agent1, agent2)[env_index] <= self.comms_radius * (self.x_semidim + self.y_semidim)/2:
                    line = rendering.Line(
                        agent1.state.pos[env_index], agent2.state.pos[env_index], width=5
                    )
                    line.set_color(*Color.GREEN.value)
                    geoms.append(line)
        return geoms


if __name__ == "__main__":
    scenario = HitSwitchScenario()
    render_interactively(
       scenario, control_two_agents=True, n_passages=1, shared_reward=False
    )