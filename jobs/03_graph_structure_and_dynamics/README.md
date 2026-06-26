# 03 Graph Structure And Dynamics

This section covers the structural and dynamic RPG graph analysis runs:

- `jobs/reproduction/rpg/graph_analysis/run_static_*.sh`
- `jobs/reproduction/rpg/graph_analysis/run_dynamic_*.sh`
- `jobs/reproduction/rpg/graph_analysis/run_scoring_*.sh`
- `jobs/reproduction/rpg/graph_analysis/run_frontier_memory_*.sh`
- `jobs/reproduction/rpg/graph_analysis/run_novelty_*.sh`
- `jobs/reproduction/rpg/graph_analysis/run_pruning_*.sh`
- `jobs/reproduction/rpg/graph_analysis/run_pool_rerank_*.sh`

These jobs support the graph-structure and graph-dynamics sections of the paper.

Convenience wrapper:

```bash
cd jobs/03_graph_structure_and_dynamics
DATASET=sports_and_outdoors ANALYSIS=static bash ./submit_dataset.sh
```
