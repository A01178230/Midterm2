"""
quickstart_multiagent_planner.py

Un script de inicio para tu proyecto: implementa
- GridGraph (mapa en cuadrícula)
- A* (ruta para un agente)
- Detección de conflictos (espacio-tiempo)
- CBS secuencial (esqueleto funcional para 2..N agentes)
- Simulador sencillo y visualización con matplotlib

Objetivo: tener un entregable ejecutable que genere simulaciones y sirva de base para
extender con particionado, paralelismo y RL.

Dependencias: numpy, matplotlib
Instalación (si no las tienes):
    pip install numpy matplotlib

Uso:
    python quickstart_multiagent_planner.py

El script genera un mapa aleatorio pequeño, planifica rutas para varios agentes
con CBS secuencial y muestra la animación/estadísticas.

"""

import random
import math
import heapq
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.neighbors import KDTree
from matplotlib.animation import PillowWriter


# --------------------------
# Utils: grid and helpers
# --------------------------

def make_grid(width, height, obstacle_prob=0.18, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    grid = np.zeros((height, width), dtype=np.int8)  # 0 free, 1 obstacle
    for y in range(height):
        for x in range(width):
            if random.random() < obstacle_prob:
                grid[y, x] = 1
    return grid


def in_bounds(grid, p):
    h, w = grid.shape
    x, y = p
    return 0 <= x < w and 0 <= y < h


def neighbors(grid, p):
    x, y = p
    cand = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
    for nx, ny in cand:
        if in_bounds(grid,(nx,ny)) and grid[ny, nx] == 0:
            yield (nx, ny)


def manhattan(a,b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

# --------------------------
# A* implementation
# --------------------------

def astar_grid(grid, start, goal, heuristic=manhattan):
    if start == goal:
        return [start]
    openq = []
    heapq.heappush(openq, (heuristic(start, goal), 0, start, None))
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
            # reconstruct
            path = []
            cur = node
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            return list(reversed(path))
        for neigh in neighbors(grid, node):
            ng = g + 1
            if neigh in closed:
                continue
            if neigh not in gscore or ng < gscore[neigh]:
                gscore[neigh] = ng
                heapq.heappush(openq, (ng + heuristic(neigh, goal), ng, neigh, node))
    return None

# --------------------------
# Time-expanded path helpers and conflict detection
# --------------------------

def to_time_expanded(path):
    # path: list of positions -> return list of (pos, t)
    return [(pos, t) for t, pos in enumerate(path)]


def detect_conflicts(paths):
    # paths: list of lists of positions (each a schedule)
    conflicts = []
    N = len(paths)
    maxT = max(len(p) for p in paths)
    for i in range(N):
        for j in range(i+1, N):
            for t in range(maxT):
                pos_i = paths[i][t] if t < len(paths[i]) else paths[i][-1]
                pos_j = paths[j][t] if t < len(paths[j]) else paths[j][-1]
                # vertex conflict
                if pos_i == pos_j:
                    conflicts.append({'type':'vertex','a1':i,'a2':j,'t':t,'pos':pos_i})
                # edge swap
                prev_i = paths[i][t-1] if t-1 >=0 and t-1 < len(paths[i]) else None
                prev_j = paths[j][t-1] if t-1 >=0 and t-1 < len(paths[j]) else None
                if prev_i is not None and prev_j is not None:
                    if prev_i == pos_j and prev_j == pos_i:
                        conflicts.append({'type':'edge','a1':i,'a2':j,'t':t,'pos':(prev_i,pos_i)})
    return conflicts

# --------------------------
# Simple CBS (sequential) implementation
# --------------------------
# Constraints: list of dicts: {agent, type('vertex'|'edge'), t, pos}


def path_cost(path):
    return len(path)


def satisfies_constraint(path, constraint, agent_id):
    # constraint applies only to specified agent
    if constraint.get('agent') is not None and constraint['agent'] != agent_id:
        return True
    t = constraint['t']
    if constraint['type'] == 'vertex':
        pos = path[t] if t < len(path) else path[-1]
        return pos != constraint['pos']
    elif constraint['type'] == 'edge':
        # pos is (from,to)
        from_pos, to_pos = constraint['pos']
        if t-1 < 0:
            return True
        prev = path[t-1] if t-1 < len(path) else path[-1]
        cur = path[t] if t < len(path) else path[-1]
        return not (prev == from_pos and cur == to_pos)
    return True


def replan_with_constraints(grid, agent_id, start, goal, constraints):
    # naive: forbid vertex/edge by temporarily marking them as blocked for the time step
    # We implement a time-aware A* that checks constraints at expansion time
    from collections import deque

    def heuristic(a,b):
        return manhattan(a,b)

    # state: (pos, t)
    start_state = (start, 0)
    openq = []
    heapq.heappush(openq, (heuristic(start, goal), 0, start_state, None))
    came_from = {}
    gscore = {start_state: 0}
    closed = set()
    max_expand_t = 200
    while openq:
        f, g, (node, t), parent = heapq.heappop(openq)
        if (node,t) in closed:
            continue
        closed.add((node,t))
        came_from[(node,t)] = parent
        # Goal test: reached goal; we allow waiting at goal
        if node == goal:
            # reconstruct path by following parents and then compressing time
            path_rev = []
            cur = (node,t)
            while cur is not None:
                pos, _t = cur
                path_rev.append(pos)
                cur = came_from[cur]
            path = list(reversed(path_rev))
            # compress: keep until stays the same; ensure path is non-empty
            # remove duplicate trailing identical positions to avoid overlong wait
            # But we keep as-is since conflicts expect time-expanded
            return path
        # expand neighbors and wait action
        # generate candidate moves
        cand_moves = list(neighbors(grid, node)) + [node]  # wait
        for nei in cand_moves:
            nt = t+1
            new_state = (nei, nt)
            if nt > max_expand_t:
                continue
            # build candidate path prefix by following parents (expensive) OR evaluate constraint locally
            # We'll check constraints that target this agent and time nt
            violated = False
            for c in constraints:
                if c.get('agent') is not None and c['agent'] != agent_id:
                    continue
                if c['t'] != nt:
                    continue
                if c['type'] == 'vertex':
                    if nei == c['pos']:
                        violated = True
                        break
                elif c['type'] == 'edge':
                    from_pos, to_pos = c['pos']
                    # prev position at time t is node
                    if node == from_pos and nei == to_pos:
                        violated = True
                        break
            if violated:
                continue
            ng = g + 1
            if new_state not in gscore or ng < gscore[new_state]:
                gscore[new_state] = ng
                heapq.heappush(openq, (ng + heuristic(nei, goal), ng, new_state, (node,t)))
    return None


def cbs_sequential(grid, starts, goals):
    # starts/goals: lists of positions
    num_agents = len(starts)
    # root node: compute individual shortest paths
    root = {'paths': [], 'constraints': []}
    for i in range(num_agents):
        p = astar_grid(grid, starts[i], goals[i])
        if p is None:
            raise RuntimeError(f"No path for agent {i}")
        root['paths'].append(p)
    import heapq
    openq = []
    # cost = sum of path lengths
    def cost(node):
        return sum(len(p) for p in node['paths'])
    heapq.heappush(openq, (cost(root), root))

    while openq:
        _, node = heapq.heappop(openq)
        conflicts = detect_conflicts(node['paths'])
        if not conflicts:
            return node['paths']
        # pick first conflict (naive)
        conf = conflicts[0]
        a1 = conf['a1']; a2 = conf['a2']; t = conf['t']
        # generate two child nodes with constraints for a1 and a2
        children = []
        if conf['type'] == 'vertex':
            pos = conf['pos']
            for agent in [a1, a2]:
                new_constraints = node['constraints'] + [{'agent': agent, 'type':'vertex', 't':t, 'pos':pos}]
                paths = node['paths'].copy()
                newpath = replan_with_constraints(grid, agent, starts[agent], goals[agent], new_constraints)
                if newpath is None:
                    continue
                paths[agent] = newpath
                children.append({'paths': paths, 'constraints': new_constraints})
        else:  # edge
            frompos, topos = conf['pos']
            for agent, (fpos, tpos) in zip([a1,a2], [conf['pos'], (conf['pos'][1], conf['pos'][0])]):
                new_constraints = node['constraints'] + [{'agent': agent, 'type':'edge', 't':t, 'pos':(fpos,tpos)}]
                paths = node['paths'].copy()
                newpath = replan_with_constraints(grid, agent, starts[agent], goals[agent], new_constraints)
                if newpath is None:
                    continue
                paths[agent] = newpath
                children.append({'paths': paths, 'constraints': new_constraints})
        for child in children:
            heapq.heappush(openq, (cost(child), child))
    return None

# --------------------------
# Simulation + visualization
# --------------------------

def sample_free_cell(grid):
    h,w = grid.shape
    while True:
        x = random.randrange(0,w)
        y = random.randrange(0,h)
        if grid[y,x] == 0:
            return (x,y)


def build_instance(width=20, height=20, n_agents=4, obstacle_prob=0.18, seed=None):
    grid = make_grid(width, height, obstacle_prob, seed)
    starts = []
    goals = []
    for _ in range(n_agents):
        s = sample_free_cell(grid)
        g = sample_free_cell(grid)
        while g == s:
            g = sample_free_cell(grid)
        starts.append(s)
        goals.append(g)
    return grid, starts, goals


def animate_solution(grid, paths, starts, goals, interval=500):
    h,w = grid.shape
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(-0.5, w-0.5)
    ax.set_ylim(-0.5, h-0.5)
    ax.set_xticks([]); ax.set_yticks([])
    # draw obstacles
    for y in range(h):
        for x in range(w):
            if grid[y,x] == 1:
                ax.add_patch(plt.Rectangle((x-0.5,h-1-y-0.5),1,1, color='black'))
    scat = ax.scatter([], [], s=200)
    colors = plt.cm.get_cmap('tab10')
    maxT = max(len(p) for p in paths)
    def frame(t):
        pts = []
        cols = []
        for i,p in enumerate(paths):
            pos = p[t] if t < len(p) else p[-1]
            # matplotlib y axis invert
            pts.append((pos[0], h-1-pos[1]))
            cols.append(colors(i))
        scat.set_offsets(pts)
        scat.set_color(cols)
        ax.set_title(f'Timestep {t}')
        return scat,
    ani = animation.FuncAnimation(fig, lambda i: frame(i), frames=range(maxT), interval=interval, blit=False, repeat=False)
    plt.show()

# ==== Regional A* Planner (integrated with spatial partitioning) ====
class RegionalAStar:
    def __init__(self, grid, partitioner):
        self.grid = grid
        self.partitioner = partitioner

    def plan_for_agents(self, agents, goals):
        regions = self.partitioner.partition_agents(agents)
        regional_paths = {}
        for region_id, agent_ids in regions.items():
            for aid in agent_ids:
                path = self.plan_single(agents[aid], goals[aid])
                regional_paths[aid] = path
        return regional_paths

    def plan_single(self, start, goal):
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break
            for nxt in neighbors(self.grid, current):
                new_cost = cost_so_far[current] + 1
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + self.heuristic(nxt, goal)
                    heapq.heappush(frontier, (priority, nxt))
                    came_from[nxt] = current
        return self.reconstruct_path(came_from, start, goal)

    def plan_single_with_constraints(self, agent_id, start, goal, constraints, max_expand_t=200):
        """
        Time-aware A* that respects vertex and edge constraints for a single agent.
        Constraints: list of dicts with keys: agent (optional), type('vertex'|'edge'), t, pos
        """
        # state: (pos, t)
        start_state = (start, 0)
        openq = []
        heapq.heappush(openq, (self.heuristic(start, goal), 0, start_state, None))
        came_from = {}
        gscore = {start_state: 0}
        closed = set()
        while openq:
            f, g, (node, t), parent = heapq.heappop(openq)
            if (node, t) in closed:
                continue
            closed.add((node, t))
            came_from[(node, t)] = parent
            if node == goal:
                # reconstruct path by following parents
                path_rev = []
                cur = (node, t)
                while cur is not None:
                    pos, _t = cur
                    path_rev.append(pos)
                    cur = came_from[cur]
                path = list(reversed(path_rev))
                return path
            # expand neighbors + wait
            cand_moves = list(neighbors(self.grid, node)) + [node]
            for nei in cand_moves:
                nt = t + 1
                if nt > max_expand_t:
                    continue
                violated = False
                for c in constraints:
                    if c.get('agent') is not None and c['agent'] != agent_id:
                        continue
                    if c['t'] != nt:
                        continue
                    if c['type'] == 'vertex':
                        if nei == c['pos']:
                            violated = True
                            break
                    elif c['type'] == 'edge':
                        from_pos, to_pos = c['pos']
                        if node == from_pos and nei == to_pos:
                            violated = True
                            break
                if violated:
                    continue
                new_state = (nei, nt)
                ng = g + 1
                if new_state not in gscore or ng < gscore[new_state]:
                    gscore[new_state] = ng
                    heapq.heappush(openq, (ng + self.heuristic(nei, goal), ng, new_state, (node, t)))
        return None

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def reconstruct_path(self, came_from, start, goal):
        # support both simple came_from (node->parent) and time-aware came_from ((node,t)->parent)
        if (goal in came_from):
            # simple case
            path = [goal]
            while path[-1] != start:
                path.append(came_from[path[-1]])
            path.reverse()
            return path
        # time-aware
        # find any key with node == goal
        goal_keys = [k for k in came_from.keys() if k[0] == goal and isinstance(k, tuple)]
        if not goal_keys:
            return []
        # pick one with smallest time
        goal_key = min(goal_keys, key=lambda x: x[1])
        path_rev = []
        cur = goal_key
        while cur is not None:
            pos = cur[0]
            path_rev.append(pos)
            cur = came_from[cur]
        path = list(reversed(path_rev))
        return path

# ==== Regional CBS integration ====

def cbs_regional(grid, starts, goals, partitioner, max_iterations=1000):
    """
    High-level regional CBS:
    1. Partition agents using partitioner
    2. Plan locally per region using RegionalAStar
    3. Detect conflicts across all agents
    4. Resolve conflicts using CBS where replanning for an agent uses RegionalAStar.plan_single_with_constraints
    """
    num_agents = len(starts)
    # build helpers
    regional_planner = RegionalAStar(grid, partitioner)

    # initial planning per region
    paths = regional_planner.plan_for_agents({i: starts[i] for i in range(num_agents)}, {i: goals[i] for i in range(num_agents)})
    # convert to list by agent id order
    paths_list = [paths[i] for i in range(num_agents)]

    # CBS node structure
    from heapq import heappush, heappop
    def node_cost(n):
        return sum(len(p) for p in n['paths'])

    root = {'paths': paths_list, 'constraints': []}
    openq = []
    heappush(openq, (node_cost(root), root))
    iterations = 0
    while openq and iterations < max_iterations:
        iterations += 1
        _, node = heappop(openq)
        conflicts = detect_conflicts(node['paths'])
        if not conflicts:
            return node['paths']
        conf = conflicts[0]
        a1 = conf['a1']; a2 = conf['a2']; t = conf['t']
        children = []
        if conf['type'] == 'vertex':
            pos = conf['pos']
            for agent in [a1, a2]:
                new_constraints = node['constraints'] + [{'agent': agent, 'type': 'vertex', 't': t, 'pos': pos}]
                paths_copy = [p.copy() for p in node['paths']]
                newpath = regional_planner.plan_single_with_constraints(agent, starts[agent], goals[agent], new_constraints)
                if newpath is None or len(newpath) == 0:
                    continue
                paths_copy[agent] = newpath
                children.append({'paths': paths_copy, 'constraints': new_constraints})
        else:  # edge conflict
            frompos, topos = conf['pos']
            # two constraints: forbid edge for each agent
            for agent, edge in zip([a1, a2], [conf['pos'], (conf['pos'][1], conf['pos'][0])]):
                new_constraints = node['constraints'] + [{'agent': agent, 'type': 'edge', 't': t, 'pos': edge}]
                paths_copy = [p.copy() for p in node['paths']]
                newpath = regional_planner.plan_single_with_constraints(agent, starts[agent], goals[agent], new_constraints)
                if newpath is None or len(newpath) == 0:
                    continue
                paths_copy[agent] = newpath
                children.append({'paths': paths_copy, 'constraints': new_constraints})
        for child in children:
            heappush(openq, (node_cost(child), child))
    # failed to find solution within iterations
    return None

# ==== Spatial Partitioning (KD-Tree + Voronoi) ====

class SpatialPartitioner:
    def __init__(self, grid, num_regions):
        self.grid = grid
        self.num_regions = num_regions
        self.centroids = self._init_centroids()
        self.kdtree = KDTree(self.centroids)

    def _init_centroids(self):
        free_cells = [(x, y) for x in range(self.grid.width)
                             for y in range(self.grid.height)
                             if self.grid.passable((x,y))]
        np.random.shuffle(free_cells)
        return free_cells[:self.num_regions]

    def assign_region(self, pos):
        dist, idx = self.kdtree.query([pos], k=1)
        return idx[0][0]

    def partition_agents(self, agents):
        regions = {i: [] for i in range(self.num_regions)}
        for aid, start in agents.items():
            r = self.assign_region(start)
            regions[r].append(aid)
        return regions


# ==== Parallel CBS, D* Lite (simplified) and RL Meta-controller ====
import concurrent.futures
from collections import defaultdict, deque

# --- Helper: conflict graph and connected components ---
def build_conflict_graph(conflicts):
    """Return adjacency list of agents that are in conflict."""
    adj = defaultdict(set)
    for c in conflicts:
        a1 = c['a1']; a2 = c['a2']
        adj[a1].add(a2)
        adj[a2].add(a1)
    return adj

def connected_components(adj, nodes):
    seen = set()
    comps = []
    for n in nodes:
        if n in seen:
            continue
        comp = []
        dq = deque([n])
        seen.add(n)
        while dq:
            u = dq.popleft()
            comp.append(u)
            for v in adj.get(u, []):
                if v not in seen:
                    seen.add(v)
                    dq.append(v)
        comps.append(comp)
    return comps

# --- Parallel CBS ---

def cbs_parallel(grid, starts, goals, partitioner, max_iterations=200, workers=None):
    """
    Parallel CBS: detect conflicts, build conflict graph, solve each connected component (subproblem)
    independently in parallel using regional CBS (which replans only affected agents). Iterate until
    no global conflicts or max_iterations exceeded.
    """
    num_agents = len(starts)
    regional_planner = RegionalAStar(grid, partitioner)

    # initial plan
    paths = regional_planner.plan_for_agents({i: starts[i] for i in range(num_agents)}, {i: goals[i] for i in range(num_agents)})
    paths_list = [paths[i] for i in range(num_agents)]

    it = 0
    while it < max_iterations:
        it += 1
        conflicts = detect_conflicts(paths_list)
        if not conflicts:
            return paths_list
        adj = build_conflict_graph(conflicts)
        nodes = list(range(num_agents))
        comps = connected_components(adj, nodes)
        # prepare subproblems: only components with >1 agent need resolution
        subprobs = [c for c in comps if len(c) > 1]
        if not subprobs:
            # conflicts exist but no edges? fallback sequential CBS
            res = cbs_regional(grid, starts, goals, partitioner, max_iterations=1000)
            return res
        # run each subproblem in parallel
        results = {}
        def solve_subproblem(agent_subset):
            # build local starts/goals and call cbs_regional limited to these agents
            local_idx = {i: starts[i] for i in agent_subset}
            local_goals = {i: goals[i] for i in agent_subset}
            # remap indices to 0..k-1 for internal planner
            remap = {old: new for new, old in enumerate(agent_subset)}
            inv_remap = {v:k for k,v in remap.items()}
            small_starts = [local_idx[old] for old in agent_subset]
            small_goals = [local_goals[old] for old in agent_subset]
            # Use cbs_regional but with remapped indices by creating adapted functions
            small_paths = cbs_regional(grid, small_starts, small_goals, partitioner, max_iterations=500)
            if small_paths is None:
                return None
            # map back
            mapped = {agent_subset[i]: small_paths[i] for i in range(len(agent_subset))}
            return mapped

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            future_to_comp = {ex.submit(solve_subproblem, comp): comp for comp in subprobs}
            for fut in concurrent.futures.as_completed(future_to_comp):
                comp = future_to_comp[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = None
                results[tuple(comp)] = res
        # integrate results
        updated = False
        for comp, sol in results.items():
            if sol is None:
                continue
            for aid, p in sol.items():
                if len(p) == 0:
                    continue
                if p != paths_list[aid]:
                    paths_list[aid] = p
                    updated = True
        if not updated:
            # unable to improve; break to avoid infinite loop
            break
    # final check
    if detect_conflicts(paths_list):
        return None
    return paths_list

# --- Simplified D* Lite implementation for grid replanning ---
# This is a compact version that supports dynamic cost changes and incremental replanning.

class DStarLite:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        # g and rhs values
        self.g = {}
        self.rhs = {}
        self.U = []  # priority queue
        self.km = 0
        self.parents = {}
        self._init()

    def _heuristic(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def _init(self):
        self.g = defaultdict(lambda: float('inf'))
        self.rhs = defaultdict(lambda: float('inf'))
        self.rhs[self.goal] = 0
        self._push(self.goal, self._calculate_key(self.goal))

    def _calculate_key(self, s):
        g_rhs = min(self.g.get(s, float('inf')), self.rhs.get(s, float('inf')))
        return (g_rhs + self._heuristic(self.start, s) + self.km, g_rhs)

    def _push(self, s, key):
        heapq.heappush(self.U, (key, s))

    def _update_vertex(self, u):
        if u != self.goal:
            # rhs = min over successors (cost(u,u') + g(u'))
            vals = []
            for v in neighbors(self.grid, u):
                vals.append(1 + self.g.get(v, float('inf')))
            self.rhs[u] = min(vals) if vals else float('inf')
        # remove u from U if present (lazy removal)
        self._push(u, self._calculate_key(u))

    def compute_shortest_path(self, max_iterations=10000):
        it = 0
        while self.U and it < max_iterations:
            it += 1
            k_old, u = heapq.heappop(self.U)
            k_old_val = k_old
            k_new = self._calculate_key(u)
            if k_old_val < k_new:
                self._push(u, k_new)
            elif self.g.get(u, float('inf')) > self.rhs.get(u, float('inf')):
                self.g[u] = self.rhs[u]
                for s in neighbors(self.grid, u):
                    self._update_vertex(s)
            else:
                g_old = self.g.get(u, float('inf'))
                self.g[u] = float('inf')
                for s in neighbors(self.grid, u) + [u]:
                    self._update_vertex(s)
        # after compute, build shortest path from start to goal using g-values
        return self._extract_path()

    def _extract_path(self):
        if self.g.get(self.start, float('inf')) == float('inf'):
            return []
        path = [self.start]
        cur = self.start
        visited = set([cur])
        while cur != self.goal:
            succs = list(neighbors(self.grid, cur))
            if not succs:
                return []
            # pick successor with minimal cost 1+g[v]
            best = min(succs, key=lambda v: 1 + self.g.get(v, float('inf')))
            if best in visited:
                # loop detected
                return []
            path.append(best)
            visited.add(best)
            cur = best
        return path

    def update_edge(self, u, v, new_cost):
        # For grid edges cost is 1 for passable, inf for blocked
        # Here we simply trigger vertex updates for affected nodes
        self._update_vertex(u)
        self._update_vertex(v)


# --- RL Integration: simple Gym env and PPO meta-controller example ---
try:
    import gym
    from gym import spaces
    from stable_baselines3 import PPO
except Exception as e:
    gym = None
    PPO = None

class MetaControllerEnv(gym.Env if gym else object):
    """
    Environment where the agent selects which agent should yield in the next conflict resolution step.
    Observation: flattened vector of (for a small fixed number of agents) distances-to-goal normalized and conflict flags.
    Action: index of agent to yield (0..n-1) or n meaning 'no-op'.
    Reward: negative total conflicts after applying action and replanning.
    NOTE: This is a simplified environment to demonstrate integration; you should adapt it to your
    full simulator and observation design.
    """
    metadata = {'render.modes': []}

    def __init__(self, grid, starts, goals, partitioner, max_agents=6):
        if gym is None:
            raise RuntimeError('gym or stable-baselines3 not available')
        super().__init__()
        self.grid = grid
        self.starts = starts
        self.goals = goals
        self.partitioner = partitioner
        self.max_agents = max_agents
        self.n_agents = len(starts)
        # observation: for each agent: dist_to_goal (scalar), conflict_flag (0/1)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.max_agents*2,), dtype=float)
        # action: choose agent index to yield or no-op
        self.action_space = spaces.Discrete(self.max_agents + 1)
        self.regional_planner = RegionalAStar(self.grid, self.partitioner)

    def reset(self):
        # initial planning
        self.paths = self.regional_planner.plan_for_agents({i: self.starts[i] for i in range(self.n_agents)}, {i: self.goals[i] for i in range(self.n_agents)})
        self.paths = [self.paths[i] for i in range(self.n_agents)]
        obs = self._build_obs()
        return obs

    def _build_obs(self):
        obs = np.zeros(self.max_agents*2, dtype=float)
        for i in range(min(self.n_agents, self.max_agents)):
            path = self.paths[i]
            dist = (len(path) - 1) if path else (self.grid.shape[0] + self.grid.shape[1])
            obs[2*i] = dist / (self.grid.shape[0] + self.grid.shape[1])
        conflicts = detect_conflicts(self.paths)
        conflicted_agents = set()
        for c in conflicts:
            conflicted_agents.add(c['a1']); conflicted_agents.add(c['a2'])
        for i in range(min(self.n_agents, self.max_agents)):
            obs[2*i+1] = 1.0 if i in conflicted_agents else 0.0
        return obs

    def step(self, action):
        # if action < n_agents: force that agent to wait by adding a vertex constraint at t=1 on its start
        if action < self.n_agents:
            agent = action
            constraint = {'agent': agent, 'type': 'vertex', 't': 1, 'pos': self.starts[agent]}
            # replan that agent
            newp = self.regional_planner.plan_single_with_constraints(agent, self.starts[agent], self.goals[agent], [constraint])
            if newp and len(newp) > 0:
                self.paths[agent] = newp
        # else no-op
        # compute reward
        conflicts = detect_conflicts(self.paths)
        reward = -len(conflicts)
        done = True  # one-step episode for simplicity
        obs = self._build_obs()
        info = {'conflicts': len(conflicts)}
        return obs, reward, done, info

# Example training stub (won't run if stable-baselines3 is not installed)
def train_meta_controller(grid, starts, goals, partitioner):
    if PPO is None:
        print('stable-baselines3 not available; skipping training stub')
        return None
    env = MetaControllerEnv(grid, starts, goals, partitioner)
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)
    return model

# === End of augmented features ===

# --------------------------
# RUNNABLE SIMULATION ENTRYPOINTS + GIF EXPORT
# --------------------------

# Configuration dictionary (modifiable)
DEFAULT_CONFIG = {
    'width': 24,
    'height': 16,
    'n_agents': 6,
    'obstacle_prob': 0.18,
    'seed': 123,
    'gif_path': 'simulation_output.gif',
    'frame_interval_ms': 250,
    'workers': 4,
}

class SpatialPartitionerSimple:
    """
    Particiona la cuadrícula en bloques iguales (grid-based partitioning).
    Solo sirve para acelerar búsquedas locales o CBS regional.
    """
    def __init__(self, grid, num_regions=4):
        self.grid = grid
        self.h, self.w = grid.shape
        self.num_regions = num_regions

        # dividir en regiones tipo rejilla 2x2, 3x3, etc.
        self.cols = int(np.sqrt(num_regions))
        self.rows = max(1, num_regions // self.cols)

        self.region_w = self.w / self.cols
        self.region_h = self.h / self.rows

    def region_of(self, pos):
        x, y = pos
        c = min(int(x // self.region_w), self.cols - 1)
        r = min(int(y // self.region_h), self.rows - 1)
        return r * self.cols + c

class SpatialPartitionerCustom:
    """
    Particionador más avanzado: agrupa celdas libres usando K-means.
    Produce regiones más compactas que la versión simple.
    """
    def __init__(self, grid, num_regions=4):
        self.grid = grid
        self.h, self.w = grid.shape
        self.num_regions = num_regions

        free_cells = [(x, y) for y in range(self.h) for x in range(self.w) if grid[y, x] == 0]
        free_cells = np.array(free_cells)

        if len(free_cells) == 0:
            raise ValueError("Grid has no free cells")

        # inicializar centros aleatorios
        centers_idx = np.random.choice(len(free_cells), num_regions, replace=False)
        centers = free_cells[centers_idx]

        # simple K-means (3 iteraciones)
        for _ in range(3):
            # asignación
            d = np.sum((free_cells[:, None, :] - centers[None, :, :])**2, axis=2)
            labels = np.argmin(d, axis=1)

            # actualización
            for k in range(num_regions):
                pts = free_cells[labels == k]
                if len(pts) > 0:
                    centers[k] = pts.mean(axis=0)

        self.centers = centers

    def region_of(self, pos):
        d = np.sum((self.centers - np.array(pos))**2, axis=1)
        return int(np.argmin(d))


def prepare_trash_instance(width, height, n_agents, obstacle_prob=0.18, seed=None):
    """
    Build a trash-collection instance with:
      - agents (starts)
      - one pickup location per agent
      - a small set of depots (drop-off zones)
    """
    grid = make_grid(width, height, obstacle_prob, seed)
    starts = []
    pickups = []
    for _ in range(n_agents):
        s = sample_free_cell(grid)
        p = sample_free_cell(grid)
        while p == s:
            p = sample_free_cell(grid)
        starts.append(s)
        pickups.append(p)
    # create 1 or 2 depots
    depots = []
    for _ in range(max(1, n_agents//3)):
        depots.append(sample_free_cell(grid))
    return grid, starts, pickups, depots


def simulate_trash_collection(config=None, save_gif=True):
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
    width = cfg['width']; height = cfg['height']; n_agents = cfg['n_agents']
    grid, starts, pickups, depots = prepare_trash_instance(width, height, n_agents, cfg['obstacle_prob'], cfg['seed'])

    # assign each agent a pickup and a nearest depot
    goals_phase1 = {i: pickups[i] for i in range(n_agents)}
    # compute nearest depot for each pickup
    def nearest_depot(pos):
        dists = [manhattan(pos, d) for d in depots]
        return depots[int(np.argmin(dists))]
    goals_phase2 = {i: nearest_depot(pickups[i]) for i in range(n_agents)}

    partitioner = SpatialPartitionerSimple(grid, num_regions=max(2, n_agents//2)) if 'SpatialPartitionerSimple' in globals() else SpatialPartitionerCustom(grid, max(2, n_agents//2))
    # NOTE: in case your canvas contains a different partitioner name, replace above with SpatialPartitioner(grid, num_regions)

    # Phase 1: plan starts -> pickups
    print('Phase 1 planning: agents -> pickups')
    paths_phase1 = cbs_parallel(grid, starts, goals_phase1, partitioner, workers=cfg['workers'])
    if paths_phase1 is None:
        print('Phase 1 failed to find conflict-free plans. Falling back to sequential CBS.')
        paths_phase1 = cbs_regional(grid, starts, [goals_phase1[i] for i in range(n_agents)], partitioner)
        if paths_phase1 is None:
            raise RuntimeError('Unable to plan phase 1')
        paths_phase1 = {i: paths_phase1[i] for i in range(n_agents)}

    # animate phase 1
    frames = []
    frames += render_frames_for_paths(grid, paths_phase1, starts, pickups, depots)

    # Phase 2: after pickup, plan pickups -> depots
    print('Phase 2 planning: pickups -> depots')
    starts_phase2 = [pickups[i] for i in range(n_agents)]
    paths_phase2 = cbs_parallel(grid, starts_phase2, goals_phase2, partitioner, workers=cfg['workers'])
    if paths_phase2 is None:
        print('Phase 2 failed to find conflict-free plans. Falling back to sequential CBS.')
        paths_phase2 = cbs_regional(grid, starts_phase2, [goals_phase2[i] for i in range(n_agents)], partitioner)
        if paths_phase2 is None:
            raise RuntimeError('Unable to plan phase 2')
        paths_phase2 = {i: paths_phase2[i] for i in range(n_agents)}

    frames += render_frames_for_paths(grid, paths_phase2, starts_phase2, depots, depots)

    # Save GIF
    gif_path = cfg['gif_path']
    save_frames_as_gif(frames, gif_path, interval_ms=cfg['frame_interval_ms'])
    print(f'Saved simulation GIF to: {gif_path}')
    return gif_path

# Utility: render frames given paths (returns list of RGB images)
def render_frames_for_paths(grid, paths_map, starts, goals, depots, max_frames_per_phase=200):
    # paths_map: dict agent->path (list of positions)
    num_agents = len(paths_map)
    paths = [paths_map[i] for i in range(num_agents)]
    maxT = min(max(len(p) for p in paths), max_frames_per_phase)
    h, w = grid.shape
    frames = []
    for t in range(maxT):
        fig, ax = plt.subplots(figsize=(6,4))
        ax.set_xlim(-0.5, w-0.5); ax.set_ylim(-0.5, h-0.5)
        ax.set_xticks([]); ax.set_yticks([])
        # draw obstacles
        for y in range(h):
            for x in range(w):
                if grid[y,x] == 1:
                    ax.add_patch(plt.Rectangle((x-0.5,h-1-y-0.5),1,1,color='black'))
        # draw depots
        for d in depots:
            ax.add_patch(plt.Circle((d[0], h-1-d[1]), 0.3, color='gold'))
        # draw goals (pickup or depot)
        for i in range(len(goals)):
            g = goals[i]
            ax.add_patch(plt.Rectangle((g[0]-0.2,h-1-g[1]-0.2),0.4,0.4, color='green', alpha=0.6))
        # agents
        colors = plt.cm.get_cmap('tab10')
        for i,p in enumerate(paths):
            pos = p[t] if t < len(p) else p[-1]
            ax.scatter([pos[0]], [h-1-pos[1]], s=120, color=colors(i), zorder=5)
            # trail: show previous up to 5 steps
            trail = [p[k] if k < len(p) else p[-1] for k in range(max(0, t-5), t+1)]
            trail_coords = [(pp[0], h-1-pp[1]) for pp in trail]
            xs = [c[0] for c in trail_coords]; ys=[c[1] for c in trail_coords]
            ax.plot(xs, ys, linewidth=2, color=colors(i), alpha=0.7)
        ax.set_title(f'Timestep {t}')
        fig.canvas.draw()
        # convert to RGB array
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(img)
        plt.close(fig)
    return frames


def save_frames_as_gif(frames, filename, interval_ms=250):
    if not frames:
        raise RuntimeError('No frames to save')
    # Use PillowWriter via matplotlib animation
    fig = plt.figure(figsize=(6,4))
    plt.axis('off')
    im = plt.imshow(frames[0])
    def update(i):
        im.set_data(frames[i])
        return [im]
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=interval_ms, blit=True)
    writer = PillowWriter(fps=1000/interval_ms)
    ani.save(filename, writer=writer)
    plt.close(fig)

# Convenience: run with default config when executed as script
if __name__ == '__main__':
    cfg = DEFAULT_CONFIG.copy()
    cfg['gif_path'] = os.path.join(os.getcwd(), 'trash_simulation.gif')
    try:
        simulate_trash_collection(cfg, save_gif=True)
    except Exception as e:
        print('Simulation failed:', e)
        raise

# --------------------------
# Main demo
# --------------------------
def main_demo():
    grid, starts, goals = build_instance(width=18, height=12, n_agents=5, obstacle_prob=0.18, seed=42)
    print("Starts:", starts)
    print("Goals: ", goals)
    try:
        paths = cbs_sequential(grid, starts, goals)
    except RuntimeError as e:
        print("Falló la planificación inicial:", e)
        return
    print("Paths found. Lengths:", [len(p) for p in paths])
    conflicts = detect_conflicts(paths)
    print("Conflicts detected after CBS (should be none):", conflicts)
    animate_solution(grid, paths, starts, goals, interval=400)

if __name__ == '__main__':
    main_demo()
