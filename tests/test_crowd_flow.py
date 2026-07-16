import json
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from crowd_flow import (
    CrowdSimulator,
    GridMap,
    Point,
    Scenario,
    ScenarioError,
    astar,
    load_scenario,
    render_heatmap_svg,
    render_report,
)


ROOT = Path(__file__).resolve().parents[1]


class CrowdFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scenario = load_scenario(ROOT / "scenario.json")

    def test_sample_map_is_rectangular_and_configured(self) -> None:
        self.assertEqual(self.scenario.grid.width, 24)
        self.assertEqual(self.scenario.grid.height, 12)
        self.assertEqual(len(self.scenario.exits), 2)
        self.assertEqual(sum(group.count for group in self.scenario.spawn_groups), 17)

    def test_astar_routes_around_walls(self) -> None:
        grid = GridMap(("#######", "#..#..#", "#..#..#", "#.....#", "#######"))
        path = astar(grid, Point(1, 1), Point(5, 1))
        self.assertTrue(path)
        self.assertEqual(path[0], Point(1, 1))
        self.assertEqual(path[-1], Point(5, 1))
        self.assertTrue(all(grid.traversable(point) for point in path))

    def test_hazard_cost_changes_route(self) -> None:
        grid = GridMap(("#######", "#.....#", "#.....#", "#.....#", "#######"))
        direct = astar(grid, Point(1, 2), Point(5, 2))
        detour = astar(
            grid,
            Point(1, 2),
            Point(5, 2),
            hazard_cells={Point(3, 2)},
            hazard_weight=20,
        )
        self.assertIn(Point(3, 2), direct)
        self.assertNotIn(Point(3, 2), detour)

    def test_simulation_is_deterministic(self) -> None:
        first = CrowdSimulator(self.scenario, seed=7).run().to_dict()
        second = CrowdSimulator(self.scenario, seed=7).run().to_dict()
        self.assertEqual(first, second)
        self.assertNotEqual(first, CrowdSimulator(self.scenario, seed=8).run().to_dict())

    def test_sample_evacuation_completes(self) -> None:
        result = CrowdSimulator(self.scenario).run()
        self.assertEqual(result.metrics.total_agents, 17)
        self.assertEqual(result.metrics.evacuated, 17)
        self.assertEqual(result.metrics.trapped, 0)
        self.assertEqual(sum(result.metrics.exit_usage.values()), 17)
        self.assertLess(result.metrics.duration, self.scenario.max_steps)

    def test_agent_positions_remain_serializable(self) -> None:
        result = CrowdSimulator(self.scenario).run()
        encoded = json.dumps(result.to_dict())
        self.assertIn('"visit_heat"', encoded)
        self.assertTrue(all("," in key for key in result.visit_heat))

    def test_report_and_svg_are_valid(self) -> None:
        result = CrowdSimulator(self.scenario).run()
        report = render_report(self.scenario, result)
        self.assertIn("# Harbor Concourse Drill — Crowd Flow Report", report)
        self.assertIn("Completion rate", report)
        svg = render_heatmap_svg(self.scenario, result)
        root = ET.fromstring(svg)
        self.assertEqual(root.tag, "{http://www.w3.org/2000/svg}svg")

    def test_rejects_non_rectangular_map(self) -> None:
        raw = json.loads((ROOT / "scenario.json").read_text(encoding="utf-8"))
        raw["map"][2] = raw["map"][2][:-1]
        with self.assertRaisesRegex(ScenarioError, "equal width"):
            Scenario.from_dict(raw)

    def test_rejects_exit_not_on_marker(self) -> None:
        raw = json.loads((ROOT / "scenario.json").read_text(encoding="utf-8"))
        raw["exits"][0]["position"] = [2, 2]
        with self.assertRaisesRegex(ScenarioError, "not on an E tile"):
            Scenario.from_dict(raw)


if __name__ == "__main__":
    unittest.main()
