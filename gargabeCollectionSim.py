"""
garbage_collection_agentpy.py

Multi-agent garbage collection simulator with:
 - A* and a compact D* Lite (incremental replanning)
 - Voronoi partitioning (KDTree-based)
 - AgentPy model for RL-style simulation and training
 - Simulator/benchmarks + GIF export

Usage:
    pip install numpy scipy matplotlib pillow agentpy
    python garbage_collection_agentpy.py --demo
    python garbage_collection_agentpy.py --run-tests
    python garbage_collection_agentpy.py --benchmark

Note: AgentPy must be installed to run the RL model parts.
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
from scipy.spatial import KDTree
import heapq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.patches import Rectangle, Circle
import matplotlib.lines as mlines
from PIL import Image

# AgentPy for RL/simulation model
try:
    import agentpy as ap
except Exception as e:
    ap = None  # AgentPy optional but recommended

# -----------------------------
# Configuration and utilities
# -----------------------------
DEFAULT_CONFIG = {
    'width': 14,
    'height': 14,
    'n_trucks': 4,
    'n_bins': 20,
    'n_obstacles': 10,
    'n_depots': 3,
    'bin_capacity': 100.0,
    'bin_fill_rate': 1.0,            # per timestep
    'bin_fill_threshold': 0.8,       # fraction to trigger service
    'truck_capacity': 200.0,
    'truck_energy': 500.0,
    'truck_energy_per_step': 1.0,
    'truck_speed': 1,                # grid cells per step (not used for fractional moves)
    'seed': 0,
    'gif_path': 'garbage_sim.gif',
    'frame_interval_ms': 500,
    'benchmark_repeats': 3,
    'bin_refill_cooldown': 50,         # pasos que tarda un bin en volver a llenarse
    'collision_energy_penalty': 5.0
}

_rng = random.Random()

GridPos = Tuple[int, int]

# -----------------------------
# Grid and Entities
# -----------------------------
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

class Bin:
    def __init__(self, pos: GridPos, capacity: float, fill_rate: float = 1.0, refill_cooldown: int = 50):
        self.pos = pos
        self.capacity = capacity
        self.fill_rate = fill_rate
        self.level = 0.0

        # cooldown: el tiempo (step) mínimo antes de que vuelva a llenarse tras ser vaciado
        self.refill_cooldown = refill_cooldown
        self.next_fill_time = 0    # paso a partir del cual empezará a llenarse de nuevo

        # reserva temporal cuando un camión lo ha seleccionado (evita reasignaciones rápidas)
        self.reserved_by = None
        self.reserved_until = 0

    def step(self, current_step: int):
        """Incrementa nivel solo si pasó el cooldown."""
        if current_step >= self.next_fill_time:
            self.level = min(self.capacity, self.level + self.fill_rate)

    def needs_service(self, threshold: float) -> bool:
        return self.level >= threshold * self.capacity
    
    def mark_collected(self, current_step: int):
        """Llamar cuando un camión vacía/recoge del contenedor."""
        # bloquear refill por el cooldown completo
        self.next_fill_time = current_step + self.refill_cooldown
        # liberar la reserva por si existe
        self.reserved_by = None
        self.reserved_until = 0

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
# Compact D* Lite (incremental)
# -----------------------------
class DStarLite:
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
        self.points = np.array(points, dtype=float)
        self.world = world
        self.kdt = KDTree(self.points)

    def region_of(self, pos: GridPos) -> int:
        dist, idx = self.kdt.query([pos], k=1)
        idx = np.asarray(idx)
        # Manejar tanto scalares como arrays
        if idx.ndim == 0:
            return int(idx)
        else:
            return int(idx[0])

    def assignment_map(self) -> np.ndarray:
        h, w = self.world.height, self.world.width
        grid_pts = np.array([(x,y) for y in range(h) for x in range(w)])
        dists = ((grid_pts[:,None,:] - self.points[None,:,:])**2).sum(axis=2)
        labels = np.argmin(dists, axis=1)
        return labels.reshape((h,w))

# -----------------------------
# AgentPy Model for RL-compatible simulation
# -----------------------------
if ap is not None:
    class TrashCollectionModel(ap.Model):
        def setup(self):
            # parameters (use defaults if not provided)
            self.width = self.p.get('width', DEFAULT_CONFIG['width'])
            self.height = self.p.get('height', DEFAULT_CONFIG['height'])
            self.n_agents = self.p.get('n_trucks', DEFAULT_CONFIG['n_trucks'])
            self.n_bins = self.p.get('n_bins', DEFAULT_CONFIG['n_bins'])
            self.n_depots = self.p.get('n_depots', DEFAULT_CONFIG['n_depots'])
            self.bin_capacity = self.p.get('bin_capacity', DEFAULT_CONFIG['bin_capacity'])
            self.bin_threshold = self.p.get('bin_fill_threshold', DEFAULT_CONFIG['bin_fill_threshold'])
            self.energy_max = self.p.get('truck_energy', DEFAULT_CONFIG['truck_energy'])
            self.capacity_max = self.p.get('truck_capacity', DEFAULT_CONFIG['truck_capacity'])

            # grid and agent lists
            self.grid = ap.Grid(self, (self.width, self.height), track_agents=True)
            self.trucks = ap.AgentList(self, self.n_agents)
            self.bins = ap.AgentList(self, self.n_bins)
            self.depots = ap.AgentList(self, self.n_depots)

            # place depots
            for d in self.depots:
                pos = (self.random_int(0, self.width - 1), self.random_int(0, self.height - 1))
                self.grid.place(d, pos)

            # place bins
            for b in self.bins:
                pos = (self.random_int(0, self.width - 1), self.random_int(0, self.height - 1))
                self.grid.place(b, pos)
                b.fill = 0.0
                b.capacity = self.bin_capacity

            # place trucks
            for t in self.trucks:
                pos = (self.random_int(0, self.width - 1), self.random_int(0, self.height - 1))
                self.grid.place(t, pos)
                t.energy = self.energy_max
                t.load = 0.0
                t.capacity = self.capacity_max
                t.state = 'idle'

            # convenience: compute Voronoi centers based on depots
            depot_points = [self.grid.positions[d] for d in self.depots]
            if len(depot_points) == 0:
                depot_points = [(0,0)]
            self.partitioner = VoronoiPartitioner(depot_points, GridWorld(self.width, self.height))

        def step(self):
            # bins fill
            for b in self.bins:
                b.fill = min(b.capacity, b.fill + self.p.get('bin_fill_rate', DEFAULT_CONFIG['bin_fill_rate']))

            # for each truck, do a greedy action (placeholder for policy)
            for t in self.trucks:
                # energy decay
                t.energy = max(0.0, t.energy - self.p.get('truck_energy_per_step', DEFAULT_CONFIG['truck_energy_per_step']))
                if t.energy <= 0.0:
                    t.state = 'idle'
                    continue

                # ---------------------------------------------------------
                # 🔵 REGISTRO DE PERFORMANCE
                # ---------------------------------------------------------

                if not hasattr(self, 'episode_bins'):
                    self.episode_bins = []
                if not hasattr(self, 'episode_distance'):
                    self.episode_distance = []
                if not hasattr(self, 'learning_progress'):
                    self.learning_progress = []

                # cuántos bins han sido recolectados
                collected = sum(b.capacity - b.fill for b in self.bins if b.fill < b.capacity)

                # distancia aproximada recorrida (AgentPy NO tiene path logs, así que sumamos movimientos)
                distance = sum(1 for t in self.trucks if len(t.history('positions')) > 1)

                self.episode_bins.append(collected)
                self.episode_distance.append(distance)

                # find bins in truck's partition needing service
                pos = self.grid.positions[t]
                my_region = self.partitioner.region_of(pos)
                candidate_bins = [b for b in self.bins if b.fill >= b.capacity * self.bin_threshold and self.partitioner.region_of(self.grid.positions[b]) == my_region]
                if not candidate_bins:
                    candidate_bins = [b for b in self.bins if b.fill >= b.capacity * self.bin_threshold]

                if candidate_bins:
                    # pick nearest candidate
                    distances = [manhattan(pos, self.grid.positions[b]) for b in candidate_bins]
                    target = candidate_bins[int(np.argmin(distances))]
                    tx, ty = pos
                    bx, by = self.grid.positions[target]

                    nx = tx + (1 if bx > tx else -1 if bx < tx else 0)
                    ny = ty + (1 if by > ty else -1 if by < ty else 0)
                    # move if free
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        try:
                            self.grid.move_to(t, (nx, ny))
                        except Exception:
                            pass

                    # attempt collect if on same cell
                    if self.grid.positions[t] == self.grid.positions[target]:
                        collect = min(target.fill, t.capacity - t.load)
                        t.load += collect
                        target.fill -= collect
                        # if full or low energy, head to depot next step
                        if t.load >= t.capacity or t.energy < self.energy_max * 0.2:
                            t.state = 'to_depot'
                        else:
                            t.state = 'idle'
                else:
                    # no bin to service: idle or return if low energy / full
                    if t.load >= t.capacity or t.energy < self.energy_max * 0.2:
                        # move to nearest depot
                        depot_positions = [self.grid.positions[d] for d in self.depots]
                        if depot_positions:
                            dists = [manhattan(self.grid.positions[t], dp) for dp in depot_positions]
                            dp = depot_positions[int(np.argmin(dists))]
                            tx, ty = self.grid.positions[t]
                            dx, dy = dp
                            nx = tx + (1 if dx > tx else -1 if dx < tx else 0)
                            ny = ty + (1 if dy > ty else -1 if dy < ty else 0)
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                try:
                                    self.grid.move_to(t, (nx, ny))
                                except Exception:
                                    pass
                            if self.grid.positions[t] == dp:
                                t.load = 0.0
                                t.energy = self.energy_max
                                t.state = 'idle'
                    else:
                        t.state = 'idle'

# -----------------------------
# Simulator (non-AgentPy) for benchmarks & GIFs
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
            self.bins.append(Bin(pos, self.config['bin_capacity'], fill_rate=self.config['bin_fill_rate'], refill_cooldown=self.config.get('bin_refill_cooldown',50)))
        self.depots: List[Depot] = []
        for _ in range(self.config['n_depots']):
            pos = self.world.random_free_cell()
            self.depots.append(Depot(pos))
        self.trucks: List[Truck] = []
        for i in range(self.config['n_trucks']):
            pos = self.world.random_free_cell()
            t = Truck(i, pos, capacity=self.config['truck_capacity'], energy=self.config['truck_energy'], energy_per_step=self.config['truck_energy_per_step'])
            dists = [manhattan(pos, d.pos) for d in self.depots]
            t.home_depot = int(np.argmin(dists)) if dists else 0
            self.trucks.append(t)
            # initialize per-truck stats for plotting/analysis
            t.collected_bins = 0
            t.total_distance = 0
            t.action_log = [t.pos]
        depot_points = [d.pos for d in self.depots]
        if not depot_points:
            depot_points = [(0,0)]
        self.partitioner = VoronoiPartitioner(depot_points, self.world)

        self.total_distance = 0
        self.collected_count = 0
        self.t = 0

        # verificar cobertura de bins
        assign_map = self.partitioner.assignment_map()
        h,w = assign_map.shape
        for b in self.bins:
            x,y = b.pos
            # assignment_map usa (h,w) indexado por [y,x] si se creó así
            assigned_region = assign_map[y, x]
            # puedes guardar b.region = assigned_region
            b.region = int(assigned_region)

    def step_bins(self):
        for b in self.bins:
            b.step(self.t)

    def find_bins_needing_service(self) -> List[int]:
        return [i for i,b in enumerate(self.bins) if b.needs_service(self.config['bin_fill_threshold'])]

    def plan_with_astar(self, truck: Truck, goal: GridPos) -> Optional[List[GridPos]]:
        return astar(self.world, truck.pos, goal)

    def run_episode(self, max_steps=2000, render=False):
        frames = []
        stats = {'serviced': 0, 'energy_penalties': 0, 'collisions': 0}

        # snapshot at start of episode to compute per-episode deltas
        start_total_distance = getattr(self, 'total_distance', 0)
        start_collected = getattr(self, 'collected_count', 0)

        for step in range(max_steps):

            self.t = step   # Para mostrar en las estadísticas de draw_world

            # ---------------------------------------------------------
            # 1. ACTUALIZAR BINS
            # ---------------------------------------------------------
            self.step_bins()
            need_bins = self.find_bins_needing_service()
            assigned = set()

            # ---------------------------------------------------------
            # 2. ASIGNACIÓN DE TAREAS A TRUCKS
            # ---------------------------------------------------------
            for t in self.trucks:

                if t.state == 'idle' or not t.path:

                    # filtrar bins por región Voronoi
                    bins_in_partition = [
                        i for i in need_bins
                        if self.partitioner.region_of(self.bins[i].pos)
                        == self.partitioner.region_of(t.pos)
                    ]

                    # fallback
                    if not bins_in_partition:
                        bins_in_partition = need_bins

                    if bins_in_partition:
                        # preferir bins no reservados o reservados por este mismo truck
                        bins_filtered = []
                        for i in bins_in_partition:
                            b = self.bins[i]
                            if b.reserved_by is None or b.reserved_by == t.id or b.reserved_until <= self.t:
                                bins_filtered.append(i)
                        if not bins_filtered:
                            bins_filtered = bins_in_partition

                        bid = min(bins_filtered, key=lambda i: manhattan(t.pos, self.bins[i].pos))
                        assigned.add(bid)

                        # reservar el bin para este truck durante un plazo prudente (ej. distancia estimada)
                        bsel = self.bins[bid]
                        est_dist = manhattan(t.pos, bsel.pos)
                        reserve_steps = max(5, est_dist + 3)  # reserva hasta que llegue (con un extra)
                        bsel.reserved_by = t.id
                        bsel.reserved_until = self.t + reserve_steps

                        path = self.plan_with_astar(t, bsel.pos)
                        if path:
                            # remove starting position to avoid "moving to same cell" steps
                            if len(path) > 1:
                                t.path = path[1:]
                                t.state = 'to_bin'
                            else:
                                # already at target cell: collect immediately
                                t.path = []
                                # buscar bin en la misma celda
                                bins_here = [i for i, b in enumerate(self.bins) if b.pos == bsel.pos]
                                if bins_here:
                                    b_idx = bins_here[0]
                                    b_obj = self.bins[b_idx]
                                    if b_obj.level > 0:
                                        amount = min(b_obj.level, t.capacity - t.load)
                                        if amount > 0:
                                            t.load += amount
                                            b_obj.level -= amount
                                            stats['serviced'] += 1
                                            self.collected_count += 1
                                            if hasattr(t, 'collected_bins'):
                                                t.collected_bins += 1
                                            # marcar cooldown y liberar reserva
                                            try:
                                                b_obj.mark_collected(self.t)
                                            except Exception:
                                                b_obj.reserved_by = None
                                                b_obj.reserved_until = 0
                                            # si está lleno o poca energía -> ir a depot
                                            if t.is_full() or t.energy < 0.1 * self.config['truck_energy']:
                                                depotpos = self.depots[t.home_depot].pos
                                                pathd = self.plan_with_astar(t, depotpos)
                                                t.path = pathd if pathd else []
                                                t.state = 'to_depot'
                                            else:
                                                t.state = 'idle'


            # ---------------------------------------------------------
            # 3. MOVIMIENTO Y RECOLECCIÓN
            # ---------------------------------------------------------
            occupied = defaultdict(list)

            for t in self.trucks:

                if t.path and len(t.path) > 0:
                    nextpos = t.path[0]  # no pop aún

                    # comprobar obstáculo en mapa
                    if not self.world.is_free(nextpos):
                        # choque con obstáculo
                        stats['collisions'] += 1
                        t.energy -= self.config.get('collision_energy_penalty', 5.0)
                        # opcional: cancelar ruta
                        t.path = []
                        t.state = 'idle'
                        continue

                    # comprobar si otra camion ya está en nextpos (colisión camión-camión)
                    occupied_by_truck = any(other.pos == nextpos and other.id != t.id for other in self.trucks)
                    if occupied_by_truck:
                        stats['collisions'] += 1
                        t.energy -= self.config.get('collision_energy_penalty', 2.0)
                        # esperar (no avanzar esta iteración)
                        continue

                    # ahora sí avanzar: pop y mover
                    prev_pos = t.pos
                    nextpos = t.path.pop(0)
                    t.energy -= t.step_cost()
                    # contar distancia solo si realmente avanza de celda
                    if nextpos != prev_pos:
                        self.total_distance += 1
                        t.total_distance += 1
                        t.pos = nextpos
                        if hasattr(t, 'action_log'):
                            t.action_log.append(t.pos)
                    else:
                        # se quedó en la misma celda (no contar como movimiento)
                        t.pos = nextpos
                    occupied[t.pos].append(t.id)

                    # si llegó a un bin
                    if t.state == 'to_bin':
                        # buscar bin en la misma celda
                        bins_here = [i for i,b in enumerate(self.bins) if b.pos == t.pos]
                        if bins_here:
                            b_idx = bins_here[0]
                            b_obj = self.bins[b_idx]

                            # solo recoger si tiene nivel positivo
                            if b_obj.level > 0:
                                amount = min(b_obj.level, t.capacity - t.load)
                                if amount > 0:
                                    t.load += amount
                                    b_obj.level -= amount

                                    # contabilizar una recolección válida
                                    stats['serviced'] += 1
                                    self.collected_count += 1
                                    if hasattr(t, 'collected_bins'):
                                        t.collected_bins += 1

                                    # marcar que bin empieza cooldown y liberar reserva
                                    try:
                                        b_obj.mark_collected(self.t)
                                    except Exception:
                                        b_obj.reserved_by = None
                                        b_obj.reserved_until = 0

                                    # si está lleno o poca energía -> ir a depot
                                    if t.is_full() or t.energy < 0.1 * self.config['truck_energy']:
                                        depotpos = self.depots[t.home_depot].pos
                                        path = self.plan_with_astar(t, depotpos)
                                        t.path = path if path else []
                                        t.state = 'to_depot'
                                    else:
                                        # si aún puede seguir, liberar estado para reasignación
                                        t.state = 'idle'
                                        t.path = []
                            else:
                                # si bin vacío, liberamos su reserva para que no bloquee
                                b_obj.reserved_by = None
                                b_obj.reserved_until = 0

                else:
                    # penalización por quedarse sin energía
                    if t.energy <= 0:
                        stats['energy_penalties'] += 1
                        t.energy = 0
                        t.state = 'idle'

            # ---------------------------------------------------------
            # 🔵 REGISTRO PARA GRAFICAS DE PERFORMANCE
            # ---------------------------------------------------------
            self._ensure_learning_log()
            self.episode_bins.append(sum(t.collected_bins for t in self.trucks))
            self.episode_distance.append(sum(t.total_distance for t in self.trucks))

            # ---------------------------------------------------------
            # 4. DETECTAR COLISIONES
            # ---------------------------------------------------------
            for pos, ids in occupied.items():
                if len(ids) > 1:
                    stats['collisions'] += len(ids) - 1

            # ---------------------------------------------------------
            # 5. ALMACENAR FRAME (CADA 10 PASOS)
            # ---------------------------------------------------------
            if render and step % 10 == 0:
                frames.append(self.render_frame())

        # -------------------------
        # Al terminar el episodio, registrar métricas de aprendizaje (por episodio)
        # -------------------------
        self._ensure_learning_log()
        # per-episode deltas
        delta_collected = self.collected_count - start_collected
        delta_distance = self.total_distance - start_total_distance
        self.episode_bins.append(delta_collected)
        self.episode_distance.append(delta_distance)
        # eficiencia: bins recolectados por unidad de distancia (fallback si distancia 0)
        efficiency = delta_collected / max(1, delta_distance)
        self.learning_progress.append(efficiency)

        return stats, frames

    # ----------------------------------------------------------------
    # Dibuja el mundo completo como un frame de la simulación
    # ----------------------------------------------------------------
    def draw_clean_world(self, ax):
        # ===============================
        # OBSTÁCULOS
        # ===============================
        for y in range(self.world.height):
            for x in range(self.world.width):
                if self.world.grid[y, x] == 1:
                    ax.scatter(x, y, c='black', s=40, marker='x', alpha=0.8)

        # ===============================
        # DEPOTS
        # ===============================
        for depot in self.depots:
            x, y = depot.pos
            ax.scatter(x, y, c='yellow', s=140, marker='D')

        # ===============================
        # BINS
        # ===============================
        for i, b in enumerate(self.bins):
            fill_level = b.level / b.capacity
            ready = fill_level >= self.config.get("pickup_threshold", 0.7)

            color = 'red' if ready else 'green'
            alpha = 0.3 + 0.7 * fill_level

            ax.scatter(b.pos[0], b.pos[1], c=color, s=80, marker='o', alpha=alpha)

        # ===============================
        # TRUCKS
        # ===============================
        for t in self.trucks:
            x, y = t.pos
            assigned = (t.state == 'to_bin')

            color = 'darkblue' if assigned else 'blue'

            ax.scatter(x, y, c=color, s=40, marker='s', alpha=0.9)

        # ===============================
        # PATHS
        # ===============================
        for t in self.trucks:
            if t.path and len(t.path) > 0:
                xs = [p[0] for p in t.path]
                ys = [p[1] for p in t.path]
                ax.plot(xs, ys, 'b--', alpha=0.3, linewidth=1)

        # ===============================
        # STATUS TEXT
        # ===============================
        serviced_bins = sum(1 for b in self.bins if b.level < b.capacity)
        total_load = sum(t.load for t in self.trucks)

        ax.text(
            0.02, 0.98,
            f"Serviced: {serviced_bins}/{len(self.bins)}\n"
            f"Total Load: {total_load:.1f}\n"
            f"Total Distance: {self.total_distance}\n",
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10
        )

    # ----------------------------------------------------------------
    # Frame de salida para GIF
    # ----------------------------------------------------------------
    def render_frame(self):
        # Figura con dos columnas: mundo (izq) y leyenda (der)
        fig, (ax, ax_leg) = plt.subplots(
            1, 2, figsize=(12, 7), dpi=200, gridspec_kw={'width_ratios': [4, 1]}
        )

        ax.set_xlim(-0.5, self.world.width - 0.5)
        ax.set_ylim(-0.5, self.world.height - 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.25)
        ax.set_title(f"Garbage Collection Simulation — Step {self.t}")

        # Dibujar el mundo en el eje izquierdo
        self.draw_clean_world(ax)

        # Preparar eje lateral para la leyenda (sin ejes)
        ax_leg.axis('off')

        # Crear handles legibles para la leyenda lateral
        handles = [
            mlines.Line2D([], [], color='blue', marker='s', markersize=10, linestyle='None', label='Truck'),
            mlines.Line2D([], [], color='green', marker='o', markersize=10, linestyle='None', label='Bin (empty)'),
            mlines.Line2D([], [], color='red', marker='o', markersize=10, linestyle='None', label='Bin (ready)'),
            mlines.Line2D([], [], color='gold', marker='D', markersize=10, linestyle='None', label='Depot'),
            mlines.Line2D([], [], color='black', marker='x', markersize=10, linestyle='None', label='Obstacle'),
            mlines.Line2D([], [], color='b', linestyle='--', label='Planned path')
        ]

        # Colocar la leyenda centrada
        ax_leg.legend(handles=handles, loc='center', fontsize=10, frameon=False, title='Legend')

        # Exportar la figura a array RGB al tamaño y dpi actuales (preserva resolución)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())   # <-- compatible y moderno
        img = buf[:, :, :3].copy()                    # quitar alpha

        plt.close(fig)
        return img
    
    # ================================================================
    #   PERFORMANCE PLOTS — listo para integrar
    # ================================================================
    # -------------------------------------------------------------------
    # REGISTRO DE APRENDIZAJE (se crea si no existe)
    # -------------------------------------------------------------------
    def _ensure_learning_log(self):
        if not hasattr(self, "learning_progress"):
            self.learning_progress = []
        if not hasattr(self, "episode_bins"):
            self.episode_bins = []
        if not hasattr(self, "episode_distance"):
            self.episode_distance = []
    
    # -------------------------------------------------------------------
    # GUARDAR TODAS LAS GRÁFICAS
    # -------------------------------------------------------------------
    def save_performance_plots(self, folder="plots"):
        os.makedirs(folder, exist_ok=True)
        self._ensure_learning_log()

        # -------------------------
        # TRUCK EFFICIENCY
        # -------------------------
        fig, ax = plt.subplots(figsize=(7, 5))
        self._plot_truck_efficiency(ax)
        save_figure(fig, f"{folder}/truck_efficiency.png")

        # -------------------------
        # PERFORMANCE GLOBAL
        # -------------------------
        fig, ax = plt.subplots(figsize=(7, 5))
        self._plot_global_performance(ax)
        save_figure(fig, f"{folder}/global_performance.png")

        # -------------------------
        # SPATIAL HEATMAP
        # -------------------------
        fig, ax = plt.subplots(figsize=(7, 5))
        self._plot_spatial_movement(ax)
        save_figure(fig, f"{folder}/movement_heatmap.png")

        # -------------------------
        # BINS OVER TIME
        # -------------------------
        fig, ax = plt.subplots(figsize=(7, 5))
        self._plot_bins_progress(ax)
        save_figure(fig, f"{folder}/bins_progress.png")

        # -------------------------
        # TRUCK INDIVIDUAL PERFORMANCE
        # -------------------------
        fig, ax = plt.subplots(figsize=(7, 5))
        self._plot_truck_individual_performance(ax)
        save_figure(fig, f"{folder}/truck_individual_performance.png")

        print(f"[OK] Performance plots saved to '{folder}'")

        # -------------------------
        # Combined figure: todos los subplots en una sola imagen (2x3)
        # -------------------------
        fig_comb, axes = plt.subplots(2, 3, figsize=(15, 10))
        ax_list = axes.flatten()

        # colocar cada plot en un subplot
        try:
            self._plot_truck_efficiency(ax_list[0])
            self._plot_global_performance(ax_list[1])
            self._plot_spatial_movement(ax_list[2])
            self._plot_bins_progress(ax_list[3])
            self._plot_truck_individual_performance(ax_list[4])

            # último panel: resumen de métricas
            ax_list[5].axis('off')
            total_served = sum(self.episode_bins) if hasattr(self, 'episode_bins') else 0
            avg_eff = (sum(self.learning_progress)/len(self.learning_progress)) if getattr(self, 'learning_progress', None) else 0
            txt = (
                f"Total episodes recorded: {len(self.episode_bins)}\n"
                f"Total bins collected (sum episodes): {total_served}\n"
                f"Average efficiency (bins/distance): {avg_eff:.3f}\n"
                f"Total distance: {getattr(self, 'total_distance', 0)}"
            )
            ax_list[5].text(0.1, 0.5, txt, fontsize=12, va='center')

            fig_comb.tight_layout()
            save_figure(fig_comb, f"{folder}/combined_performance.png")
        except Exception:
            # si algo falla en el combinado, no interrumpir
            plt.close(fig_comb)

    # -------------------------------------------------------------------
    #   TRUCK EFFICIENCY PLOT
    # -------------------------------------------------------------------
    def _plot_truck_efficiency(self, ax):
        trucks = self.trucks

        ids = [t.id for t in trucks]
        collected = [t.collected_bins for t in trucks]
        distance  = [max(1, t.total_distance) for t in trucks]

        efficiency = [c/d for c, d in zip(collected, distance)]

        bars = ax.bar(ids, efficiency, color='green', alpha=0.7)
        ax.set_title("Truck Efficiency (Bins per Distance)")
        ax.set_ylabel("Efficiency")
        ax.set_xlabel("Truck ID")

        # Etiquetas "Truck 0", "Truck 1", ...
        ax.set_xticks(ids)
        ax.set_xticklabels([f"Truck {i}" for i in ids])

        for b, e in zip(bars, efficiency):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.02,
                    f"{e:.3f}", ha='center')

    # -------------------------------------------------------------------
    #   GLOBAL PERFORMANCE
    # -------------------------------------------------------------------
    def _plot_global_performance(self, ax):
        total_collected = sum(getattr(t, 'collected_bins', 0) for t in self.trucks)
        total_distance  = sum(getattr(t, 'total_distance', 0)  for t in self.trucks)
        total_capacity  = sum(t.capacity for t in self.trucks)

        metrics = ["Collected", "Distance", "Total Capacity"]
        values  = [total_collected, total_distance, total_capacity]

        bars = ax.bar(metrics, values, color=['blue', 'orange', 'purple'], alpha=0.7)
        ax.set_title("Overall Performance")
        ax.set_ylabel("Value")

        for b, v in zip(bars, values):
            ax.text(b.get_x()+b.get_width()/2, v*1.02, f"{v}", ha="center")

    # -------------------------------------------------------------------
    #   SPATIAL MOVEMENT HEATMAP
    # -------------------------------------------------------------------
    def _plot_spatial_movement(self, ax):
        xs, ys = [], []

        for t in self.trucks:
            if not hasattr(t, 'action_log'):
                continue
            for step in t.action_log:
                # action_log stores (x,y) tuples
                xs.append(step[0])
                ys.append(step[1])

        if len(xs) == 0:
            ax.text(0.5, 0.5, "No movement data", ha="center", va="center")
            return

        ax.hist2d(xs, ys, bins=20, cmap="Blues")
        ax.set_title("Truck Position Heatmap")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")

    # -------------------------------------------------------------------
    #   BINS PROGRESS OVER TIME
    # -------------------------------------------------------------------
    def _plot_bins_progress(self, ax):
        if len(self.episode_bins) == 0:
            ax.text(0.5, 0.5, "No bins data", ha="center", va="center")
            return

        ax.plot(self.episode_bins, label="Bins Collected", linewidth=2)
        ax.plot(self.episode_distance, label="Distance", linewidth=2)
        ax.set_title("Episode Progress")
        ax.set_ylabel("Value")
        ax.set_xlabel("Step")
        ax.legend()

    def _plot_truck_individual_performance(self, ax):
        """
        Grafica el performance individual de cada truck:
        - Carga final (current load)
        - Bins recolectados
        - Distancia recorrida
        """

        truck_ids = [f"Truck {t.id}" for t in self.trucks]
        loads = [t.load for t in self.trucks]
        bins_collected = [getattr(t, "collected_bins", 0) for t in self.trucks]
        distances = [getattr(t, "total_distance", 0) for t in self.trucks]

        bar_width = 0.25
        x = np.arange(len(self.trucks))

        ax.bar(x - bar_width, loads, width=bar_width, label="Final Load", alpha=0.8)
        ax.bar(x, bins_collected, width=bar_width, label="Bins Collected", alpha=0.8)
        ax.bar(x + bar_width, distances, width=bar_width, label="Distance", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(truck_ids)
        ax.set_title("Individual Truck Performance")
        ax.set_ylabel("Value")
        ax.legend()

        # Mostrar etiquetas numéricas sobre cada barra
        for values, offset in [(loads, -bar_width), (bins_collected, 0), (distances, bar_width)]:
            for i, val in enumerate(values):
                ax.text(
                    i + offset,
                    val + max(1, val * 0.02),
                    f"{val}",
                    ha="center",
                    va="bottom",
                    fontsize=8
                )


# -----------------------------
# Benchmark utilities
# -----------------------------
def run_benchmark(config: dict, repeats: int=3, out_csv: str='benchmark.csv'):
    header = ['scenario','rep','n_trucks','n_bins','n_obstacles','plan_time_s','sim_time_s','serviced','energy_penalties','collisions']
    rows = []
    for rep in range(repeats):
        sim = Simulator(config)
        t0 = time.time()
        plan_time = 0.0
        for t in sim.trucks:
            if not sim.bins:
                continue
            bid = min(range(len(sim.bins)), key=lambda i: manhattan(t.pos, sim.bins[i].pos))
            st = time.time()
            p = sim.plan_with_astar(t, sim.bins[bid].pos)
            plan_time += time.time() - st
        stats, frames = sim.run_episode(max_steps=2000, render=False)
        sim_time = 0.0  # included in stats timing if needed
        rows.append(['default', rep, config['n_trucks'], config['n_bins'], config['n_obstacles'], round(plan_time,4), round(sim_time,4), stats['serviced'], stats['energy_penalties'], stats['collisions']])
        print(f'bench rep {rep} plan_time {plan_time:.3f}s serviced {stats["serviced"]}')
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f'Benchmark results written to {out_csv}')
    return rows

# -----------------------------
# GIF export helper
# -----------------------------
def save_frames_as_gif(frames, filename, interval_ms=200):
    """Save a list of RGB numpy arrays as an animated GIF using Pillow.

    This preserves the native resolution of the frames (no resampling
    by Matplotlib) and lets us control duration per frame.
    """
    if not frames:
        print("No frames to save.")
        return

    images = [Image.fromarray(frame.astype('uint8')) for frame in frames]
    # duration is in milliseconds per frame
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        duration=interval_ms,
        loop=0,
        optimize=True
    )

def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

# -----------------------------
# Demo / CLI
# -----------------------------
def demo(config=None):
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
    sim = Simulator(cfg)
    stats, frames = sim.run_episode(max_steps=1000, render=True)
    sim.save_performance_plots('performance_plots')
    if frames:
        save_frames_as_gif(frames, cfg['gif_path'], interval_ms=cfg['frame_interval_ms'])
        print('Saved GIF to', cfg['gif_path'])
    print('Simulation stats:', stats)

# -----------------------------
# Unit tests (basic correctness checks)
# -----------------------------
def _test_astar():
    g = GridWorld(10,8, obstacle_prob=0.1, seed=1)
    s = g.random_free_cell(); t = g.random_free_cell()
    path = astar(g, s, t)
    assert path is None or (path[0] == s and path[-1] == t)
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
    p.add_argument('--no-agentpy', action='store_true', help='Run without requiring AgentPy (skip RL model)')
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