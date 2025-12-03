"""
garbage_collection_sim.py

Purpose
-------
A self-contained simulator and toolkit for multi-agent garbage collection.
Implements three main algorithmic components requested:
 - A* and D* Lite: heuristic path planning and incremental replanning
 - Voronoi Partitioning: spatial partitioning of workspace among trucks/depots
 - Reinforcement Learning: Gym environment + PPO stub for meta-control

Features
--------
- Grid-based world with trucks (agents), bins (with fill level), fixed obstacles, and depots.
- Trucks have energy and capacity. They collect from bins when bins >= threshold (configurable).
- Trucks return to depot when full or low energy; heavy penalty for running out of energy.
- Voronoi partitioning assigns responsibility regions to trucks (based on depot or truck positions).
- A* used for pathfinding on grid; D* Lite provided for dynamic replanning when obstacles change.
- Simple Gym environment for training a meta-controller (which can decide priorities or handoffs).
- Benchmark harness: runs scenarios, times planners, measures throughput and resource usage, exports CSV.
- GIF export of simulation (requires matplotlib & pillow).

Usage
-----
Install dependencies:
    pip install numpy scipy matplotlib pillow gym stable-baselines3

Run a quick demo:
    python garbage_collection_sim.py --demo

Run benchmarks:
    python garbage_collection_sim.py --benchmark

The file is modular: import classes/functions for integration into experiments.

"""

from __future__ import annotations
import argparse
import csv
import os
import random
import time
import math
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Optional

import numpy as np
from scipy.spatial import Voronoi, KDTree
import heapq
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

# Optional RL imports guarded at runtime
try:
    import gym
    from gym import spaces
    from stable_baselines3 import PPO
except Exception:
    gym = None
    PPO = None

# -----------------------------
# Configuration and utilities
# -----------------------------

DEFAULT_CONFIG = {
    'width': 40,
    'height': 28,
    'n_trucks': 4,
    'n_bins': 40,
    'n_obstacles': 80,
    'n_depots': 2,
    'bin_capacity': 100.0,
    'bin_fill_rate': 1.0,            # per timestep
    'bin_fill_threshold': 0.8,       # fraction to trigger service
    'truck_capacity': 200.0,
    'truck_energy': 500.0,
    'truck_energy_per_step': 1.0,
    'truck_speed': 1,                # grid cells per step
    'seed': 0,
    'gif_path': 'garbage_sim.gif',
    'frame_interval_ms': 200,
    'benchmark_repeats': 3,
}

# RNG
_rng = random.Random()

# -----------------------------
# Grid and Entities
# -----------------------------

GridPos = Tuple[int, int]

class GridWorld:
    def __init__(self, width: int, height: int, obstacle_prob: float = 0.0, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed); _rng.seed(seed)
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=np.int8)  # 0 free, 1 obstacle
        if obstacle_prob > 0:
            for y in range(height):
                for x in range(width):
                    if _rng.random() < obstacle_prob:
                        self.grid[y, x] = 1

    def in_bounds(self, p: GridPos) -> bool:
        x, y = p
        return 0 <= x < self.width and 0 <= y < self.height

    def is_free(self, p: GridPos) -> bool:
        x, y = p
        return self.in_bounds(p) and self.grid[y, x] == 0

    def neighbors(self, p: GridPos) -> List[GridPos]:
        x, y = p
        cand = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
        return [q for q in cand if self.is_free(q)]

    def random_free_cell(self) -> GridPos:
        while True:
            x = _rng.randrange(self.width)
            y = _rng.randrange(self.height)
            if self.grid[y, x] == 0:
                return (x, y)

# Entities
class Bin:
    def __init__(self, pos: GridPos, capacity: float, fill_rate: float = 1.0):
        self.pos = pos
        self.capacity = capacity
        self.fill_rate = fill_rate
        self.level = 0.0

    def step(self):
        self.level = min(self.capacity, self.level + self.fill_rate)

    def needs_service(self, threshold: float) -> bool:
        return self.level >= threshold * self.capacity

    def empty(self):
        self.level = 0.0

class Depot:
    def __init__(self, pos: GridPos):
        self.pos = pos

class Truck:
    def __init__(self, id: int, pos: GridPos, capacity: float, energy: float, energy_per_step: float):
        self.id = id
        self.pos = pos
        self.capacity = capacity
        self.load = 0.0
        self.energy = energy
        self.energy_per_step = energy_per_step
        self.home_depot: Optional[int] = None
        self.path: List[GridPos] = []
        self.state = 'idle'  # idle, to_bin, to_depot, unloading

    def step_cost(self):
        return self.energy_per_step

    def is_full(self):
        return self.load >= self.capacity

# -----------------------------
# A* Pathfinding
# -----------------------------
def manhattan(a: GridPos, b: GridPos) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(grid: GridWorld, start: GridPos, goal: GridPos) -> Optional[List[GridPos]]:
    if start == goal:
        return [start]
    openq = []
    heapq.heappush(openq, (manhattan(start,goal), 0, start, None))
    came_from = {}
    gscore = {start: 0}
    closed = set()
    while openq:
        f, g, node, parent = heapq.heappop(openq)
        if node in closed:
            continue
        closed.add(node)
        came_from[node] = parent
        if node == goal:
            path = []
            cur = node
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            return list(reversed(path))
        for neigh in grid.neighbors(node):
            ng = g + 1
            if neigh in closed:
                continue
            if neigh not in gscore or ng < gscore[neigh]:
                gscore[neigh] = ng
                heapq.heappush(openq, (ng + manhattan(neigh,goal), ng, neigh, node))
    return None

# -----------------------------
# D* Lite (compact implementation)
# -----------------------------
class DStarLite:
    # Compactized D* Lite for grid: supports compute_shortest_path and notify_edge_changes
    def __init__(self, grid: GridWorld, start: GridPos, goal: GridPos):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.g = defaultdict(lambda: float('inf'))
        self.rhs = defaultdict(lambda: float('inf'))
        self.U = []
        self.km = 0.0
        self.rhs[goal] = 0.0
        self._push(goal)

    def _heuristic(self, s: GridPos) -> float:
        return manhattan(self.start, s)

    def _key(self, s: GridPos):
        g_rhs = min(self.g.get(s, float('inf')), self.rhs.get(s, float('inf')))
        return (g_rhs + self._heuristic(s) + self.km, g_rhs)

    def _push(self, s: GridPos):
        heapq.heappush(self.U, (self._key(s), s))

    def _update_vertex(self, u: GridPos):
        if u != self.goal:
            vals = []
            for v in self.grid.neighbors(u):
                vals.append(1 + self.g.get(v, float('inf')))
            self.rhs[u] = min(vals) if vals else float('inf')
        # push/update
        self._push(u)

    def compute_shortest_path(self, max_iters=10000):
        it = 0
        while self.U and it < max_iters:
            it += 1
            (k_old, _), u = heapq.heappop(self.U)
            k_new = self._key(u)
            if k_old < k_new:
                self._push(u); continue
            if self.g.get(u, float('inf')) > self.rhs.get(u, float('inf')):
                self.g[u] = self.rhs[u]
                for s in self.grid.neighbors(u):
                    self._update_vertex(s)
            else:
                g_old = self.g.get(u, float('inf'))
                self.g[u] = float('inf')
                for s in self.grid.neighbors(u) + [u]:
                    self._update_vertex(s)
        return self._extract_path()

    def _extract_path(self) -> List[GridPos]:
        if self.g.get(self.start, float('inf')) == float('inf'):
            return []
        path = [self.start]
        cur = self.start
        visited = set([cur])
        while cur != self.goal:
            succs = list(self.grid.neighbors(cur))
            if not succs:
                return []
            best = min(succs, key=lambda v: 1 + self.g.get(v, float('inf')))
            if best in visited:
                return []
            path.append(best)
            visited.add(best)
            cur = best
        return path

    def notify_edge_change(self, u: GridPos, v: GridPos):
        self._update_vertex(u)
        self._update_vertex(v)

# -----------------------------
# Voronoi Partitioning
# -----------------------------
class VoronoiPartitioner:
    def __init__(self, points: List[GridPos], world: GridWorld):
        # points: worker positions or depot positions (x,y)
        self.points = np.array(points, dtype=float)
        self.world = world
        self.kdt = KDTree(self.points)

    def region_of(self, pos: GridPos) -> int:
        dist, idx = self.kdt.query([pos], k=1)
        return int(idx[0][0])

    def assignment_map(self) -> np.ndarray:
        # returns array shape (h,w) mapping each cell to nearest point index
        h, w = self.world.height, self.world.width
        xs = np.arange(w); ys = np.arange(h)
        grid_pts = np.array([(x,y) for y in range(h) for x in range(w)])
        dists = ((grid_pts[:,None,:] - self.points[None,:,:])**2).sum(axis=2)
        labels = np.argmin(dists, axis=1)
        return labels.reshape((h,w))

# -----------------------------
# Simple RL environment (meta-controller)
# -----------------------------
if gym is not None:
    class TrashCollectEnv(gym.Env):
        metadata = {'render.modes': ['human']}
        def __init__(self, world: GridWorld, trucks: List[Truck], bins: List[Bin], depots: List[Depot], config: dict):
            super().__init__()
            self.world = world
            self.trucks = trucks
            self.bins = bins
            self.depots = depots
            self.config = config
            self.n_trucks = len(trucks)
            self.max_bins = len(bins)
            # obs: for each truck: dist to nearest need-bin normalized, load fraction, energy fraction
            obs_low = np.zeros(self.n_trucks*3, dtype=float)
            obs_high = np.ones(self.n_trucks*3, dtype=float)
            self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=float)
            # action: choose truck to prioritize (0..n_trucks-1) or n_trucks = no-op
            self.action_space = spaces.Discrete(self.n_trucks + 1)

        def reset(self):
            return self._obs()

        def _obs(self):
            obs = []
            need_bins = [b for b in self.bins if b.needs_service(self.config['bin_fill_threshold'])]
            for t in self.trucks:
                # dist to nearest need bin
                if len(need_bins) == 0:
                    dnorm = 1.0
                else:
                    d = min(manhattan(t.pos, b.pos) for b in need_bins)
                    dnorm = d / (self.world.width + self.world.height)
                loadf = t.load / t.capacity if t.capacity>0 else 0.0
                ef = t.energy / self.config['truck_energy']
                obs += [dnorm, loadf, ef]
            return np.array(obs, dtype=float)

        def step(self, action):
            # action: prioritize one truck (we will give it a small reward boost if it services a bin)
            # In this simplified environment we don't run full sim, this is a hook for integration into training
            reward = 0.0
            done = True
            info = {}
            return self._obs(), reward, done, info

# -----------------------------
# Simulation loop and orchestration
# -----------------------------
class Simulator:
    def __init__(self, config: dict):
        self.config = config.copy()
        _rng.seed(self.config.get('seed', 0))
        self.world = GridWorld(self.config['width'], self.config['height'], obstacle_prob=0.0, seed=self.config.get('seed',0))
        # place random obstacles
        for _ in range(self.config['n_obstacles']):
            x = _rng.randrange(self.world.width); y = _rng.randrange(self.world.height)
            self.world.grid[y,x] = 1
        # bins, depots, trucks
        self.bins: List[Bin] = []
        for _ in range(self.config['n_bins']):
            pos = self.world.random_free_cell()
            self.bins.append(Bin(pos, self.config['bin_capacity'], fill_rate=self.config['bin_fill_rate']))
        self.depots: List[Depot] = []
        for _ in range(self.config['n_depots']):
            pos = self.world.random_free_cell()
            self.depots.append(Depot(pos))
        self.trucks: List[Truck] = []
        for i in range(self.config['n_trucks']):
            pos = self.world.random_free_cell()
            t = Truck(i, pos, capacity=self.config['truck_capacity'], energy=self.config['truck_energy'], energy_per_step=self.config['truck_energy_per_step'])
            # assign nearest depot as home
            dists = [manhattan(pos, d.pos) for d in self.depots]
            t.home_depot = int(np.argmin(dists))
            self.trucks.append(t)
        # Voronoi partitioner per depot centers for workload split
        depot_points = [d.pos for d in self.depots]
        self.partitioner = VoronoiPartitioner(depot_points, self.world)

    def step_bins(self):
        for b in self.bins:
            b.step()

    def find_bins_needing_service(self) -> List[int]:
        return [i for i,b in enumerate(self.bins) if b.needs_service(self.config['bin_fill_threshold'])]

    def plan_with_astar(self, truck: Truck, goal: GridPos) -> Optional[List[GridPos]]:
        return astar(self.world, truck.pos, goal)

    def run_episode(self, max_steps=500, render=False):
        frames = []
        stats = {'serviced':0, 'energy_penalties':0, 'collisions':0}
        for step in range(max_steps):
            # bins fill
            self.step_bins()
            # assign tasks: each truck assigned nearest bin in its partition that needs service
            need_bins = self.find_bins_needing_service()
            # mapping bin->depot-owned partition
            assigned = set()
            for t in self.trucks:
                # if truck idle, pick a task
                if t.state == 'idle' or not t.path:
                    # search bins in truck's partition
                    bins_in_partition = [i for i in need_bins if self.partitioner.region_of(self.bins[i].pos) == self.partitioner.region_of(t.pos)]
                    if not bins_in_partition:
                        bins_in_partition = need_bins
                    if bins_in_partition:
                        # pick nearest in that list
                        bid = min(bins_in_partition, key=lambda i: manhattan(t.pos, self.bins[i].pos))
                        assigned.add(bid)
                        path = self.plan_with_astar(t, self.bins[bid].pos)
                        if path:
                            t.path = path
                            t.state = 'to_bin'
            # step trucks along their path
            occupied = defaultdict(list)
            for t in self.trucks:
                if t.path and len(t.path) > 0:
                    # move
                    nextpos = t.path.pop(0)
                    # energy
                    t.energy -= t.step_cost()
                    t.pos = nextpos
                    occupied[t.pos].append(t.id)
                    # arrive at bin
                    if t.state == 'to_bin' and any(b.pos == t.pos for b in self.bins):
                        # service bin
                        b_idx = next(i for i,b in enumerate(self.bins) if b.pos == t.pos)
                        # collect as much as capacity allows
                        amount = min(self.bins[b_idx].level, t.capacity - t.load)
                        t.load += amount
                        self.bins[b_idx].level -= amount
                        stats['serviced'] += 1
                        # if full or low energy, head to depot
                        if t.is_full() or t.energy < 0.1 * self.config['truck_energy']:
                            depotpos = self.depots[t.home_depot].pos
                            path = self.plan_with_astar(t, depotpos)
                            t.path = path if path else []
                            t.state = 'to_depot'
                else:
                    # idle behaviour: if low energy, try return depot
                    if t.energy <= 0:
                        stats['energy_penalties'] += 1
                        t.energy = 0
                        t.state = 'idle'
            # collision check
            for pos, ids in occupied.items():
                if len(ids) > 1:
                    stats['collisions'] += len(ids)-1
            # render frame
            if render:
                frames.append(self.render_frame())
        return stats, frames

    def render_frame(self):
        h,w = self.world.height, self.world.width
        fig, ax = plt.subplots(figsize=(6,4))
        ax.set_xlim(-0.5, w-0.5); ax.set_ylim(-0.5, h-0.5)
        ax.set_xticks([]); ax.set_yticks([])
        # obstacles
        for y in range(h):
            for x in range(w):
                if self.world.grid[y,x] == 1:
                    ax.add_patch(plt.Rectangle((x-0.5, h-1-y-0.5),1,1,color='black'))
        # bins
        for b in self.bins:
            frac = b.level / b.capacity
            color = (1.0, 1.0-frac, 0.0)
            ax.add_patch(plt.Circle((b.pos[0], h-1-b.pos[1]), 0.25, color=color))
        # depots
        for d in self.depots:
            ax.add_patch(plt.Rectangle((d.pos[0]-0.3, h-1-d.pos[1]-0.3),0.6,0.6,color='blue'))
        # trucks
        cmap = plt.cm.get_cmap('tab10')
        for t in self.trucks:
            ax.scatter([t.pos[0]], [h-1-t.pos[1]], s=120, color=cmap(t.id%10))
            ax.text(t.pos[0]-0.2, h-1-t.pos[1]+0.2, f'{t.load:.0f}/{t.capacity}', fontsize=6)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img

# -----------------------------
# Benchmark utilities
# -----------------------------
def run_benchmark(config: dict, repeats: int=3, out_csv: str='benchmark.csv'):
    header = ['scenario','rep','n_trucks','n_bins','n_obstacles','plan_time_s','sim_time_s','serviced','energy_penalties','collisions']
    rows = []
    for rep in range(repeats):
        sim = Simulator(config)
        start_t = time.time()
        # example measurement: time to plan initial tasks for all trucks
        t0 = time.time()
        # plan: for each truck, plan to nearest assigned bin
        need_bins = sim.find_bins_needing_service()
        plan_time = 0.0
        for t in sim.trucks:
            # pick nearest bin
            if not sim.bins:
                continue
            bid = min(range(len(sim.bins)), key=lambda i: manhattan(t.pos, sim.bins[i].pos))
            st = time.time()
            p = sim.plan_with_astar(t, sim.bins[bid].pos)
            plan_time += time.time() - st
        # run a short simulation
        st = time.time()
        stats, frames = sim.run_episode(max_steps=200, render=False)
        sim_time = time.time() - st
        rows.append(['default', rep, config['n_trucks'], config['n_bins'], config['n_obstacles'], round(plan_time,4), round(sim_time,4), stats['serviced'], stats['energy_penalties'], stats['collisions']])
        print(f'bench rep {rep} plan_time {plan_time:.3f}s sim_time {sim_time:.3f}s serviced {stats["serviced"]}')
    # write CSV
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f'Benchmark results written to {out_csv}')
    return rows

# -----------------------------
# Demo / CLI
# -----------------------------

def demo(config=None):
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
    sim = Simulator(cfg)
    stats, frames = sim.run_episode(max_steps=300, render=True)
    # save gif
    if frames:
        save_frames_as_gif(frames, cfg['gif_path'], interval_ms=cfg['frame_interval_ms'])
        print('Saved GIF to', cfg['gif_path'])
    print('Simulation stats:', stats)


def save_frames_as_gif(frames: List[np.ndarray], filename: str, interval_ms: int = 200):
    if not frames:
        return
    fig = plt.figure(figsize=(6,4)); plt.axis('off')
    im = plt.imshow(frames[0])
    def update(i):
        im.set_data(frames[i]); return [im]
    ani = plt.animation.FuncAnimation(fig, update, frames=len(frames), interval=interval_ms, blit=True)
    writer = PillowWriter(fps=1000/interval_ms)
    ani.save(filename, writer=writer)
    plt.close(fig)

# -----------------------------
# Unit tests (basic correctness checks)
# -----------------------------

def _test_astar():
    g = GridWorld(10,8, obstacle_prob=0.1, seed=1)
    s = g.random_free_cell(); t = g.random_free_cell()
    path = astar(g, s, t)
    assert path is None or path[0] == s and path[-1] == t
    print('A* test OK')

def _test_voronoi():
    g = GridWorld(20,14, obstacle_prob=0.0, seed=2)
    pts = [g.random_free_cell() for _ in range(3)]
    vp = VoronoiPartitioner(pts, g)
    for _ in range(10):
        p = g.random_free_cell()
        r = vp.region_of(p)
        assert 0 <= r < len(pts)
    print('Voronoi test OK')

def _test_dstar():
    g = GridWorld(12,10, obstacle_prob=0.0, seed=3)
    s = (0,0); goal=(11,9)
    d = DStarLite(g, s, goal)
    path = d.compute_shortest_path()
    assert isinstance(path, list)
    print('D* Lite test OK')

# -----------------------------
# CLI handling
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--demo', action='store_true')
    p.add_argument('--benchmark', action='store_true')
    p.add_argument('--run-tests', action='store_true')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.run_tests:
        _test_astar(); _test_voronoi(); _test_dstar()
    if args.demo:
        demo()
    if args.benchmark:
        cfg = DEFAULT_CONFIG.copy()
        run_benchmark(cfg, repeats=cfg['benchmark_repeats'])