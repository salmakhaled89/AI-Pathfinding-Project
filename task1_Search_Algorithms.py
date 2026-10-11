"""
How to run the code:
1. Open the terminal.
2. Type: python task1_Search_Algorithms.py and press Enter.

How to change the map:
type 'a', 'b', or 'c' in the menu when the program runs.
"""

import time
import pandas as pd
import heapq
import itertools

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", None)

MAP_A = [
"S..#....",
"##.#.##.",
"....~~..",
".####.#.",
"...~..#G"
]

MAP_B = [
"S~~~~~~~G",
".#######.",
".........",
".#.#.#.#.",
"........."
]

MAP_C = [
"S..#....",
".#.#.##.",
"...#..#.",
".###.##.",
"....#..G"
]

GRID = MAP_A


# Breadth-First Search (BFS) implementation with performance metrics

def BFS(map,s,g):
    start_time=time.perf_counter()
    #find map dimisins
    rows,cols=len(map),len(map[0])
    start,goal=None,None

    #find start and goal indices
    for row in range(rows):
        for col in range(cols):
            if map[row][col]==s:
                start=(row,col)
            elif map[row][col]==g:
                goal=(row,col)

    #initialize search variables
    queue=[(start,[start],0)] #[current node,path,cost]
    visited={start}           #set of the visited nodes
    visited_nodes=0           # number of nods visited
    max_frontier_size=1

    #while loop for searching
    while queue:
        max_frontier_size = max(max_frontier_size, len(queue))
        current,path,cost=queue.pop(0)
        visited_nodes+=1
        if current==goal:
            end_time=time.perf_counter()
            run_time=(end_time-start_time)*1000
            path_length=len(path)
            return path,path_length,cost,visited_nodes,max_frontier_size,run_time
        r,c=current

        for dr,dc in[(-1,0),(1,0),(0,-1),(0,1)]: #check for neighbors [up,down,right,left]
            nr,nc=r+dr,c+dc
            if 0<= nr <rows and 0<= nc < cols:
                cell=map[nr][nc]
                if cell !="#":
                    next_pos=(nr,nc)
                    if next_pos not in visited:
                        visited.add(next_pos)
                        step_cost=3 if cell=="~" else 1
                        queue.append((next_pos,path+[next_pos],cost+step_cost))
    end_time=time.perf_counter()
    run_time=(end_time-start_time)*1000
    return None,0,float("inf"),visited_nodes,max_frontier_size,run_time


# -------------------- A* SEARCH  --------------------
def find_start_goal(map_grid, start_symbol="S", goal_symbol="G"):
    start = goal = None
    for r, row in enumerate(map_grid):
        for c, cell in enumerate(row):
            if cell == start_symbol:
                start = (r, c)
            elif cell == goal_symbol:
                goal = (r, c)
    return start, goal


def heuristic(position, goal):
    # Manhattan distance: number of horizontal + vertical steps to the goal.
    return abs(position[0] - goal[0]) + abs(position[1] - goal[1])


def A_star(map_grid, s, g):
    """A* returns the same six-item result structure as BFS."""
    start_time = time.perf_counter()
    rows, cols = len(map_grid), len(map_grid[0])
    start, goal = find_start_goal(map_grid, s, g)

    if start is None or goal is None:
        run_time = (time.perf_counter() - start_time) * 1000
        return None, 0, float("inf"), 0, 0, run_time

    # Entries are (f-score, tie-breaker, position). The counter resolves ties.
    counter = itertools.count()
    frontier = [(heuristic(start, goal), next(counter), start)]
    came_from = {start: None}
    g_cost = {start: 0}
    nodes_expanded = 0
    max_frontier_size = 1
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        current_f, _, current = heapq.heappop(frontier)

        # Skip stale entries when a better route to this cell was found.
        if current_f != g_cost[current] + heuristic(current, goal):
            continue

        if current == goal:
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            run_time = (time.perf_counter() - start_time) * 1000
            # Match BFS's convention: path length counts coordinates S through G.
            return path, len(path), g_cost[goal], nodes_expanded, max_frontier_size, run_time

        nodes_expanded += 1
        r, c = current
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                cell = map_grid[nr][nc]
                if cell == "#":
                    continue
                neighbor = (nr, nc)
                step_cost = 3 if cell == "~" else 1
                tentative_g = g_cost[current] + step_cost
                if tentative_g < g_cost.get(neighbor, float("inf")):
                    g_cost[neighbor] = tentative_g
                    came_from[neighbor] = current
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(frontier, (f_score, next(counter), neighbor))

    run_time = (time.perf_counter() - start_time) * 1000
    return None, 0, float("inf"), nodes_expanded, max_frontier_size, run_time


# Helper function to visualize the path using '*'
def print_map_with_path(map_grid, path):
    grid_copy = [list(row) for row in map_grid]

    if path:
        for r, c in path[1:-1]:
            grid_copy[r][c] = '*'

    for row in grid_copy:
        print("".join(row))


# -------------------- INTERACTIVE MENU --------------------
while True:
    choice = input("\nEnter map name (a, b, c), press Enter for default (MAP_A), or type 'e' to quit: ").strip().lower()

    if choice == 'e':
        print("Exiting interactive menu.")
        break
    elif choice == 'a' or choice == '':
        GRID = MAP_A
        map_name = "MAP_A"
        if choice == "":
            print("Using default: MAP_A")
    elif choice == 'b':
        GRID = MAP_B
        map_name = "MAP_B"
    elif choice == 'c':
        GRID = MAP_C
        map_name = "MAP_C"
    else:
        print("Invalid choice, please try again.")
        continue

    choice_alg = input("Choose algorithm - Type (bfs or astar): ").strip().lower()

    if choice_alg == 'bfs':
        alg_name = "BFS"
        path, length, cost, nodes_visited, max_front, r_time = BFS(GRID, "S", "G")
    elif choice_alg in ('astar', 'a*'):
        alg_name = "A*"
        path, length, cost, nodes_visited, max_front, r_time = A_star(GRID, "S", "G")
    else:
        print("Invalid algorithm. Please type 'bfs' or 'astar'.")
        continue

    print(f"\nResults for {map_name} using {alg_name}:")
    print(f"Algorithm Name: {alg_name}")
    if path:
        print(f"Path: {path}")
        print(f"Path Length (coordinates including S and G): {length}")
        print(f"Total Path Cost: {cost}")
        print_map_with_path(GRID, path)
    else:
        print("No path found")
    print(f"Nodes Expanded: {nodes_visited}")
    print(f"Max Frontier Size: {max_front}")
    print(f"Run Time (ms): {r_time:.4f}")


# -------------------- FINAL COMPARISON TABLE --------------------
# Both algorithms run on all three maps after the user exits the menu.
maps_list = [("MAP_A", MAP_A), ("MAP_B", MAP_B), ("MAP_C", MAP_C)]
algorithms = [("BFS", BFS), ("A*", A_star)]

results_dict = {
    "map name": [],
    "algorithm": [],
    "path": [],
    "path length": [],
    "total cost": [],
    "nodes expanded": [],
    "max frontier size": [],
    "run time in ms": []
}

for name, current_map in maps_list:
    for alg_name, algorithm in algorithms:
        path, length, cost, nodes_visited, max_front, r_time = algorithm(current_map, "S", "G")
        results_dict["map name"].append(name)
        results_dict["algorithm"].append(alg_name)
        results_dict["path"].append(path if path else "No Path")
        results_dict["path length"].append(length if path else 0)
        results_dict["total cost"].append(cost)
        # BFS's existing metric counts dequeued nodes, including the goal.
        results_dict["nodes expanded"].append(nodes_visited)
        results_dict["max frontier size"].append(max_front)
        results_dict["run time in ms"].append(round(r_time, 4))

df_results = pd.DataFrame(results_dict)
print("\nFinal comparison: BFS vs A* on all maps")
print(df_results.to_string(index=False))
