"""Deterministic multi-agent evacuation simulation with congestion-aware A* routing."""

from __future__ import annotations

import argparse
import heapq
import html
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


class ScenarioError(ValueError):
    """Raised when a simulation scenario is invalid."""


@dataclass(frozen=True, order=True)
class Point:
    x: int
    y: int

    @classmethod
    def from_value(cls, value: Iterable[int]) -> "Point":
        x, y = value
        return cls(int(x), int(y))

    def manhattan(self, other: "Point") -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)

    def key(self) -> str:
        return f"{self.x},{self.y}"


@dataclass(frozen=True)
class Exit:
    position: Point
    capacity: int
    name: str


@dataclass(frozen=True)
class SpawnGroup:
    name: str
    count: int
    cells: tuple[Point, ...]
    mobility: float
    reaction_delay: tuple[int, int]


@dataclass(frozen=True)
class GridMap:
    rows: tuple[str, ...]

    @property
    def width(self) -> int:
        return len(self.rows[0])

    @property
    def height(self) -> int:
        return len(self.rows)

    def in_bounds(self, point: Point) -> bool:
        return 0 <= point.x < self.width and 0 <= point.y < self.height

    def tile(self, point: Point) -> str:
        return self.rows[point.y][point.x]

    def traversable(self, point: Point) -> bool:
        return self.in_bounds(point) and self.tile(point) != "#"

    def neighbors(self, point: Point) -> tuple[Point, ...]:
        candidates = (
            Point(point.x + 1, point.y),
            Point(point.x - 1, point.y),
            Point(point.x, point.y + 1),
            Point(point.x, point.y - 1),
        )
        return tuple(candidate for candidate in candidates if self.traversable(candidate))

    def points_with(self, marker: str) -> tuple[Point, ...]:
        return tuple(
            Point(x, y)
            for y, row in enumerate(self.rows)
            for x, value in enumerate(row)
            if value == marker
        )

    def validate(self) -> None:
        if not self.rows:
            raise ScenarioError("map must contain rows")
        if len({len(row) for row in self.rows}) != 1:
            raise ScenarioError("map rows must have equal width")
        if self.width < 5 or self.height < 5:
            raise ScenarioError("map must be at least 5 x 5")
        allowed = {"#", ".", "S", "E", "H"}
        unknown = sorted({value for row in self.rows for value in row} - allowed)
        if unknown:
            raise ScenarioError(f"unknown map markers: {', '.join(unknown)}")
        if not self.points_with("E"):
            raise ScenarioError("map must contain at least one exit marker E")


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    grid: GridMap
    exits: tuple[Exit, ...]
    spawn_groups: tuple[SpawnGroup, ...]
    max_steps: int = 160
    reroute_interval: int = 5
    congestion_weight: float = 1.8
    hazard_weight: float = 18.0
    hazard_growth_interval: int = 8
    hazard_max_radius: int = 3

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Scenario":
        grid = GridMap(tuple(str(row) for row in raw["map"]))
        exits = tuple(
            Exit(
                position=Point.from_value(item["position"]),
                capacity=int(item["capacity"]),
                name=str(item["name"]),
            )
            for item in raw["exits"]
        )
        groups = tuple(
            SpawnGroup(
                name=str(item["name"]),
                count=int(item["count"]),
                cells=tuple(Point.from_value(value) for value in item["cells"]),
                mobility=float(item.get("mobility", 1.0)),
                reaction_delay=tuple(int(value) for value in item.get("reaction_delay", [0, 0])),
            )
            for item in raw["spawn_groups"]
        )
        simulation = raw.get("simulation", {})
        scenario = cls(
            name=str(raw["name"]).strip(),
            description=str(raw.get("description", "")).strip(),
            grid=grid,
            exits=exits,
            spawn_groups=groups,
            max_steps=int(simulation.get("max_steps", 160)),
            reroute_interval=int(simulation.get("reroute_interval", 5)),
            congestion_weight=float(simulation.get("congestion_weight", 1.8)),
            hazard_weight=float(simulation.get("hazard_weight", 18.0)),
            hazard_growth_interval=int(simulation.get("hazard_growth_interval", 8)),
            hazard_max_radius=int(simulation.get("hazard_max_radius", 3)),
        )
        scenario.validate()
        return scenario

    def validate(self) -> None:
        self.grid.validate()
        if not self.name:
            raise ScenarioError("scenario name must not be empty")
        if self.max_steps < 1 or self.reroute_interval < 1:
            raise ScenarioError("max_steps and reroute_interval must be positive")
        if self.hazard_growth_interval < 1 or self.hazard_max_radius < 0:
            raise ScenarioError("hazard settings are invalid")

        map_exits = set(self.grid.points_with("E"))
        configured_positions: set[Point] = set()
        for exit_ in self.exits:
            if exit_.position not in map_exits:
                raise ScenarioError(f"configured exit {exit_.name} is not on an E tile")
            if exit_.capacity < 1:
                raise ScenarioError("exit capacity must be positive")
            if exit_.position in configured_positions:
                raise ScenarioError("exit positions must be unique")
            configured_positions.add(exit_.position)
        if configured_positions != map_exits:
            raise ScenarioError("every E tile must have exactly one exit configuration")

        occupied_spawn_cells: set[Point] = set()
        for group in self.spawn_groups:
            if group.count < 1 or group.count > len(group.cells):
                raise ScenarioError(f"spawn group {group.name} count exceeds available cells")
            if not 0 < group.mobility <= 1:
                raise ScenarioError("mobility must be in (0, 1]")
            low, high = group.reaction_delay
            if low < 0 or high < low:
                raise ScenarioError("reaction_delay must be [minimum, maximum]")
            for cell in group.cells:
                if not self.grid.traversable(cell) or self.grid.tile(cell) == "E":
                    raise ScenarioError(f"invalid spawn cell {cell.key()}")
                if cell in occupied_spawn_cells:
                    raise ScenarioError(f"duplicate spawn cell {cell.key()}")
                occupied_spawn_cells.add(cell)


@dataclass
class Agent:
    agent_id: str
    group: str
    position: Point
    mobility: float
    reaction_delay: int
    status: str = "active"
    target_exit: Point | None = None
    path: list[Point] = field(default_factory=list)
    path_index: int = 0
    evacuated_at: int | None = None
    wait_ticks: int = 0
    distance_walked: int = 0
    replans: int = 0
    hazard_exposure: int = 0

    @property
    def next_point(self) -> Point | None:
        next_index = self.path_index + 1
        return self.path[next_index] if next_index < len(self.path) else None

    @property
    def remaining_steps(self) -> int:
        return max(0, len(self.path) - self.path_index - 1)


@dataclass(frozen=True)
class SimulationMetrics:
    total_agents: int
    evacuated: int
    trapped: int
    completion_rate: float
    duration: int
    mean_evacuation_time: float | None
    p95_evacuation_time: float | None
    max_exit_queue: int
    mean_replans: float
    total_hazard_exposure: int
    exit_usage: dict[str, int]


@dataclass(frozen=True)
class SimulationResult:
    scenario: str
    seed: int
    metrics: SimulationMetrics
    agents: tuple[dict[str, Any], ...]
    visit_heat: dict[str, int]
    queue_history: dict[str, tuple[int, ...]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def astar(
    grid: GridMap,
    start: Point,
    goal: Point,
    *,
    congestion: dict[Point, float] | None = None,
    hazard_cells: set[Point] | None = None,
    congestion_weight: float = 0.0,
    hazard_weight: float = 0.0,
) -> list[Point]:
    """Find a least-cost path, including both *start* and *goal*."""

    if not grid.traversable(start) or not grid.traversable(goal):
        return []
    congestion = congestion or {}
    hazard_cells = hazard_cells or set()
    frontier: list[tuple[float, int, Point]] = []
    serial = 0
    heapq.heappush(frontier, (start.manhattan(goal), serial, start))
    came_from: dict[Point, Point | None] = {start: None}
    cost_so_far: dict[Point, float] = {start: 0.0}

    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for neighbor in grid.neighbors(current):
            move_cost = 1.0 + congestion.get(neighbor, 0.0) * congestion_weight
            if neighbor in hazard_cells:
                move_cost += hazard_weight
            new_cost = cost_so_far[current] + move_cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                serial += 1
                priority = new_cost + neighbor.manhattan(goal)
                heapq.heappush(frontier, (priority, serial, neighbor))
                came_from[neighbor] = current

    if goal not in came_from:
        return []
    path = [goal]
    cursor = goal
    while came_from[cursor] is not None:
        cursor = came_from[cursor]  # type: ignore[assignment]
        path.append(cursor)
    path.reverse()
    return path


class CrowdSimulator:
    """Advance agents through the map with dynamic route and queue decisions."""

    def __init__(self, scenario: Scenario, seed: int = 20260716) -> None:
        self.scenario = scenario
        self.seed = int(seed)
        self._exit_by_point = {exit_.position: exit_ for exit_ in scenario.exits}

    def run(self) -> SimulationResult:
        rng = random.Random(self.seed)
        agents = self._build_agents(rng)
        visits: Counter[Point] = Counter(agent.position for agent in agents)
        queue_history: dict[Point, list[int]] = {exit_.position: [] for exit_ in self.scenario.exits}
        exit_usage: Counter[str] = Counter()
        duration = 0

        for tick in range(self.scenario.max_steps):
            duration = tick + 1
            active = [agent for agent in agents if agent.status == "active"]
            if not active:
                break
            hazards = self._hazard_cells(tick)
            queue_counts = self._queue_counts(active)
            for exit_ in self.scenario.exits:
                queue_history[exit_.position].append(queue_counts[exit_.position])
            self._step(
                tick=tick,
                active=active,
                visits=visits,
                hazards=hazards,
                queue_counts=queue_counts,
                exit_usage=exit_usage,
                rng=rng,
            )

        for agent in agents:
            if agent.status == "active":
                agent.status = "trapped"

        metrics = self._metrics(agents, duration, queue_history, exit_usage)
        agent_records = tuple(
            {
                "agent_id": agent.agent_id,
                "group": agent.group,
                "status": agent.status,
                "evacuated_at": agent.evacuated_at,
                "distance_walked": agent.distance_walked,
                "wait_ticks": agent.wait_ticks,
                "replans": agent.replans,
                "hazard_exposure": agent.hazard_exposure,
                "final_position": agent.position.key(),
            }
            for agent in sorted(agents, key=lambda item: item.agent_id)
        )
        return SimulationResult(
            scenario=self.scenario.name,
            seed=self.seed,
            metrics=metrics,
            agents=agent_records,
            visit_heat={point.key(): count for point, count in sorted(visits.items())},
            queue_history={
                self._exit_by_point[point].name: tuple(values)
                for point, values in queue_history.items()
            },
        )

    def _build_agents(self, rng: random.Random) -> list[Agent]:
        agents: list[Agent] = []
        for group in self.scenario.spawn_groups:
            cells = list(group.cells)
            rng.shuffle(cells)
            for index, cell in enumerate(cells[: group.count], start=1):
                delay = rng.randint(group.reaction_delay[0], group.reaction_delay[1])
                agents.append(
                    Agent(
                        agent_id=f"{group.name}-{index:02d}",
                        group=group.name,
                        position=cell,
                        mobility=group.mobility,
                        reaction_delay=delay,
                    )
                )
        return sorted(agents, key=lambda agent: agent.agent_id)

    def _hazard_cells(self, tick: int) -> set[Point]:
        origins = self.scenario.grid.points_with("H")
        radius = min(self.scenario.hazard_max_radius, tick // self.scenario.hazard_growth_interval)
        return {
            Point(x, y)
            for y in range(self.scenario.grid.height)
            for x in range(self.scenario.grid.width)
            if self.scenario.grid.traversable(Point(x, y))
            and any(Point(x, y).manhattan(origin) <= radius for origin in origins)
        }

    def _queue_counts(self, active: list[Agent]) -> dict[Point, int]:
        counts = {exit_.position: 0 for exit_ in self.scenario.exits}
        for agent in active:
            if agent.target_exit is not None and agent.position.manhattan(agent.target_exit) <= 3:
                counts[agent.target_exit] += 1
        return counts

    def _step(
        self,
        *,
        tick: int,
        active: list[Agent],
        visits: Counter[Point],
        hazards: set[Point],
        queue_counts: dict[Point, int],
        exit_usage: Counter[str],
        rng: random.Random,
    ) -> None:
        occupied = {agent.position: agent.agent_id for agent in active}
        exit_intents: dict[Point, list[Agent]] = defaultdict(list)
        move_intents: dict[Point, list[Agent]] = defaultdict(list)
        heat_scale = max(1, tick + 1)
        congestion = {point: count / heat_scale for point, count in visits.items()}

        for agent in active:
            if agent.position in hazards:
                agent.hazard_exposure += 1
            if tick < agent.reaction_delay:
                agent.wait_ticks += 1
                continue
            if rng.random() > agent.mobility:
                agent.wait_ticks += 1
                continue

            needs_plan = (
                not agent.path
                or agent.next_point is None
                or (agent.wait_ticks > 0 and tick % self.scenario.reroute_interval == 0)
            )
            if needs_plan:
                self._plan_agent(agent, congestion, hazards, queue_counts)
            destination = agent.next_point
            if destination is None:
                agent.wait_ticks += 1
                continue
            if destination in self._exit_by_point:
                exit_intents[destination].append(agent)
            else:
                move_intents[destination].append(agent)

        for exit_point, candidates in sorted(exit_intents.items()):
            exit_ = self._exit_by_point[exit_point]
            ordered = sorted(candidates, key=lambda agent: (-agent.wait_ticks, agent.remaining_steps, agent.agent_id))
            for agent in ordered[: exit_.capacity]:
                occupied.pop(agent.position, None)
                agent.position = exit_point
                agent.status = "evacuated"
                agent.evacuated_at = tick + 1
                agent.distance_walked += 1
                agent.path_index += 1
                visits[exit_point] += 1
                exit_usage[exit_.name] += 1
            for agent in ordered[exit_.capacity :]:
                agent.wait_ticks += 1

        winners: list[tuple[Point, Agent]] = []
        for destination, candidates in move_intents.items():
            winner = sorted(candidates, key=lambda agent: (-agent.wait_ticks, agent.remaining_steps, agent.agent_id))[0]
            winners.append((destination, winner))
            for loser in candidates:
                if loser is not winner:
                    loser.wait_ticks += 1

        for destination, agent in sorted(winners, key=lambda item: (item[1].remaining_steps, item[1].agent_id)):
            if agent.status != "active":
                continue
            if destination in occupied:
                agent.wait_ticks += 1
                continue
            occupied.pop(agent.position, None)
            agent.position = destination
            occupied[destination] = agent.agent_id
            agent.path_index += 1
            agent.distance_walked += 1
            agent.wait_ticks = max(0, agent.wait_ticks - 1)
            visits[destination] += 1

    def _plan_agent(
        self,
        agent: Agent,
        congestion: dict[Point, float],
        hazards: set[Point],
        queue_counts: dict[Point, int],
    ) -> None:
        candidates: list[tuple[float, str, Exit, list[Point]]] = []
        for exit_ in self.scenario.exits:
            path = astar(
                self.scenario.grid,
                agent.position,
                exit_.position,
                congestion=congestion,
                hazard_cells=hazards,
                congestion_weight=self.scenario.congestion_weight,
                hazard_weight=self.scenario.hazard_weight,
            )
            if not path:
                continue
            queue_penalty = queue_counts[exit_.position] / exit_.capacity * 2.5
            score = len(path) - 1 + queue_penalty
            candidates.append((score, exit_.name, exit_, path))
        if not candidates:
            agent.path = []
            agent.target_exit = None
            return
        _, _, chosen_exit, chosen_path = min(candidates, key=lambda item: (item[0], item[1]))
        agent.target_exit = chosen_exit.position
        agent.path = chosen_path
        agent.path_index = 0
        agent.replans += 1

    def _metrics(
        self,
        agents: list[Agent],
        duration: int,
        queue_history: dict[Point, list[int]],
        exit_usage: Counter[str],
    ) -> SimulationMetrics:
        times = sorted(agent.evacuated_at for agent in agents if agent.evacuated_at is not None)
        evacuated = len(times)
        trapped = len(agents) - evacuated
        p95 = None
        if times:
            p95 = float(times[max(0, math.ceil(len(times) * 0.95) - 1)])
        max_queue = max((max(values, default=0) for values in queue_history.values()), default=0)
        return SimulationMetrics(
            total_agents=len(agents),
            evacuated=evacuated,
            trapped=trapped,
            completion_rate=round(evacuated / len(agents), 4) if agents else 1.0,
            duration=duration,
            mean_evacuation_time=round(statistics.fmean(times), 2) if times else None,
            p95_evacuation_time=p95,
            max_exit_queue=max_queue,
            mean_replans=round(statistics.fmean(agent.replans for agent in agents), 2) if agents else 0.0,
            total_hazard_exposure=sum(agent.hazard_exposure for agent in agents),
            exit_usage={exit_.name: exit_usage[exit_.name] for exit_ in self.scenario.exits},
        )


def load_scenario(path: Path) -> Scenario:
    return Scenario.from_dict(json.loads(path.read_text(encoding="utf-8")))


def render_report(scenario: Scenario, result: SimulationResult) -> str:
    metrics = result.metrics
    lines = [
        f"# {scenario.name} — Crowd Flow Report",
        "",
        scenario.description,
        "",
        f"Deterministic seed: `{result.seed}`",
        "",
        "## Outcome",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Agents | {metrics.total_agents} |",
        f"| Evacuated | {metrics.evacuated} |",
        f"| Trapped | {metrics.trapped} |",
        f"| Completion rate | {metrics.completion_rate:.1%} |",
        f"| Simulation duration | {metrics.duration} ticks |",
        f"| Mean evacuation time | {metrics.mean_evacuation_time} ticks |",
        f"| P95 evacuation time | {metrics.p95_evacuation_time} ticks |",
        f"| Maximum exit queue | {metrics.max_exit_queue} agents |",
        f"| Mean route replans | {metrics.mean_replans} |",
        f"| Hazard exposure | {metrics.total_hazard_exposure} agent-ticks |",
        "",
        "## Exit utilization",
        "",
        "| Exit | Evacuated agents | Share |",
        "|---|---:|---:|",
    ]
    for name, count in metrics.exit_usage.items():
        share = count / metrics.evacuated if metrics.evacuated else 0.0
        lines.append(f"| {name} | {count} | {share:.1%} |")

    heat = sorted(result.visit_heat.items(), key=lambda item: (-item[1], item[0]))[:10]
    lines.extend(["", "## Busiest traversable cells", "", "| Coordinate | Agent visits |", "|---|---:|"])
    lines.extend(f"| `{coordinate}` | {count} |" for coordinate, count in heat)
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The simulator replans routes using accumulated traffic heat and current exit queues. "
            "High-visit cells identify structural bottlenecks; a high P95 relative to the mean "
            "suggests that a minority of agents experienced severe queuing or hazard detours.",
            "",
        ]
    )
    return "\n".join(lines)


def render_heatmap_svg(scenario: Scenario, result: SimulationResult, cell_size: int = 34) -> str:
    margin = 74
    width = margin * 2 + scenario.grid.width * cell_size
    height = margin * 2 + scenario.grid.height * cell_size
    heat = {Point.from_value(map(int, key.split(","))): value for key, value in result.visit_heat.items()}
    max_heat = max(heat.values(), default=1)
    safe_name = html.escape(scenario.name)
    elements = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f"<title>{safe_name} evacuation heatmap</title>",
        '<rect width="100%" height="100%" fill="#0c1220"/>',
        f'<text x="{margin}" y="38" fill="#f4f7ff" font-family="system-ui" font-size="22">{safe_name} — visit heat</text>',
    ]
    for y, row in enumerate(scenario.grid.rows):
        for x, tile in enumerate(row):
            point = Point(x, y)
            px, py = margin + x * cell_size, margin + y * cell_size
            if tile == "#":
                color = "#202b3d"
            elif tile == "E":
                color = "#39d98a"
            elif tile == "H":
                color = "#ff5c70"
            else:
                intensity = heat.get(point, 0) / max_heat
                red = round(38 + 214 * intensity)
                green = round(63 + 82 * (1 - intensity))
                blue = round(112 + 90 * (1 - intensity))
                color = f"rgb({red},{green},{blue})"
            elements.append(
                f'<rect x="{px}" y="{py}" width="{cell_size-1}" height="{cell_size-1}" fill="{color}" rx="3"/>'
            )
            if tile in {"E", "H"}:
                elements.append(
                    f'<text x="{px+cell_size/2:.1f}" y="{py+cell_size*0.68:.1f}" text-anchor="middle" fill="#071018" font-family="monospace" font-weight="bold">{tile}</text>'
                )
    legend_y = height - 32
    elements.extend(
        [
            f'<circle cx="{margin}" cy="{legend_y}" r="7" fill="#39d98a"/><text x="{margin+14}" y="{legend_y+5}" fill="#cbd5e8" font-family="system-ui" font-size="13">exit</text>',
            f'<circle cx="{margin+88}" cy="{legend_y}" r="7" fill="#ff5c70"/><text x="{margin+102}" y="{legend_y+5}" fill="#cbd5e8" font-family="system-ui" font-size="13">hazard origin</text>',
            f'<text x="{width-margin}" y="{legend_y+5}" text-anchor="end" fill="#91a2bd" font-family="monospace" font-size="13">max visits: {max_heat}</text>',
            "</svg>",
        ]
    )
    return "\n".join(elements)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path, help="JSON scenario file")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("examples"))
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()

    scenario = load_scenario(args.scenario)
    result = CrowdSimulator(scenario, seed=args.seed).run()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(render_report(scenario, result), encoding="utf-8")
    (args.output_dir / "heatmap.svg").write_text(render_heatmap_svg(scenario, result), encoding="utf-8")
    print(
        f"Evacuated {result.metrics.evacuated}/{result.metrics.total_agents} agents "
        f"in {result.metrics.duration} ticks; outputs written to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
