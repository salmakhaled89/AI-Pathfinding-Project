# AI Pathfinding Project
Python implementation of BFS and A* search algorithms on grid maps with performance evaluation metrics.

## Maps Description
The project evaluates the algorithms across three distinct grid environments:
- **MAP_A:** A standard balanced map with a mix of clear paths (`.`), rough terrain/water (`~`), and walls (`#`).
- **MAP_B:** A map containing a trap where the shortest path is not the cheapest, testing the algorithms' ability to handle terrain costs.
- **MAP_C:** A challenging layout designed to have no valid solution (no path from start `S` to goal `G`).
