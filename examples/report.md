# Harbor Concourse Drill — Crowd Flow Report

A deterministic evacuation drill for a divided transport concourse with two exits, three mobility groups, queue-sensitive routing, and a slowly expanding smoke source.

Deterministic seed: `20260716`

## Outcome

| Metric | Value |
|---|---:|
| Agents | 17 |
| Evacuated | 17 |
| Trapped | 0 |
| Completion rate | 100.0% |
| Simulation duration | 42 ticks |
| Mean evacuation time | 28.82 ticks |
| P95 evacuation time | 41.0 ticks |
| Maximum exit queue | 4 agents |
| Mean route replans | 3.0 |
| Hazard exposure | 0 agent-ticks |

## Exit utilization

| Exit | Evacuated agents | Share |
|---|---:|---:|
| North Gate | 10 | 58.8% |
| East Ramp | 7 | 41.2% |

## Busiest traversable cells

| Coordinate | Agent visits |
|---|---:|
| `6,4` | 11 |
| `7,4` | 11 |
| `18,1` | 10 |
| `8,4` | 10 |
| `14,1` | 8 |
| `15,1` | 8 |
| `16,1` | 8 |
| `17,1` | 8 |
| `13,1` | 7 |
| `22,9` | 7 |

## Interpretation

The simulator replans routes using accumulated traffic heat and current exit queues. High-visit cells identify structural bottlenecks; a high P95 relative to the mean suggests that a minority of agents experienced severe queuing or hazard detours.
