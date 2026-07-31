import json
from pathlib import Path

import pytest

from eval.gui_grounding.analyze_planner_suitability import build_report, evaluate_one


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def target(uid: str, action: str, bbox: list[int], value: str = "") -> dict:
    return {
        "benchmark": "mind2web",
        "sample_id": f"mind2web:test:{uid}",
        "split": "test",
        "image": f"images/{uid}.jpg",
        "prompt": f"Direct {uid}",
        "provenance": {"action_uid": uid},
        "target_action": action,
        "target_bbox_1000": bbox,
        "target_value": value,
    }


def planner_target(row: dict) -> dict:
    result = dict(row)
    result["benchmark"] = "mind2web_task_history"
    result["sample_id"] = result["sample_id"].replace("mind2web:", "mind2web_task_history:")
    result["prompt"] = result["prompt"].replace("Direct", "Plan")
    return result


def prediction(row: dict, text: str, latency: float = 1.0) -> dict:
    return {
        "sample_id": row["sample_id"],
        "prediction": text,
        "latency_seconds": latency,
    }


def build_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    benchmark_root = tmp_path / "benchmark"
    direct_dir = tmp_path / "direct"
    planner_dir = tmp_path / "planner"
    direct = [
        target("a", "lclick", [100, 100, 200, 200]),
        target("b", "type_in", [200, 200, 300, 300], "Hello World"),
        target("c", "hover", [300, 300, 400, 400]),
        target("d", "lclick", [400, 400, 500, 500]),
    ]
    planner = [planner_target(row) for row in direct]
    write_jsonl(benchmark_root / "samples/direct.jsonl", direct)
    write_jsonl(benchmark_root / "samples/planner.jsonl", planner)
    (benchmark_root / "manifest.json").write_text(
        json.dumps(
            {
                "benchmarks": {
                    "mind2web": {"path": "samples/direct.jsonl"},
                    "mind2web_task_history": {"path": "samples/planner.jsonl"},
                }
            }
        )
    )
    direct_text = [
        "lclick [120,120,180,180]",
        "type_in [220,220,280,280] hello   world",
        "hover [320,320,380,380]",
        "lclick [10,10,20,20]",
    ]
    planner_text = [
        "lclick [120,120,180,180]",
        "type_in [220,220,280,280] goodbye",
        "hover [10,10,20,20]",
        "lclick [420,420,480,480]",
    ]
    write_jsonl(
        direct_dir / "mind2web/part-00000.jsonl",
        [prediction(row, text) for row, text in zip(direct, direct_text)],
    )
    write_jsonl(
        planner_dir / "mind2web_task_history/part-00000.jsonl",
        [prediction(row, text) for row, text in zip(planner, planner_text)],
    )
    return benchmark_root, direct_dir, planner_dir


def test_build_report_scores_strict_values_and_paired_outcomes(tmp_path: Path) -> None:
    benchmark_root, direct_dir, planner_dir = build_fixture(tmp_path)
    report, table = build_report(
        benchmark_root=benchmark_root,
        direct_predictions_dir=direct_dir,
        planner_predictions_dir=planner_dir,
    )

    direct = report["arms"]["direct_target_grounding"]["overall"]
    planner = report["arms"]["task_history_planner"]["overall"]
    paired = report["paired_comparison"]
    assert direct["strict_next_action_success"] == pytest.approx(0.75)
    assert planner["point_success"] == pytest.approx(0.75)
    assert planner["strict_next_action_success"] == pytest.approx(0.5)
    assert planner["type_value_accuracy"] == 0.0
    assert paired == {
        "both_success": 1,
        "direct_only_success": 2,
        "planner_only_success": 1,
        "neither_success": 0,
        "planner_minus_direct_percentage_points": pytest.approx(-25.0),
        "planner_retention_of_direct_success": pytest.approx(2 / 3),
        "mcnemar_exact_two_sided_p": 1.0,
    }
    assert len(report["failure_examples"]) == 2
    assert len(table) == 10


def test_build_report_rejects_incomplete_prediction_coverage(tmp_path: Path) -> None:
    benchmark_root, direct_dir, planner_dir = build_fixture(tmp_path)
    planner_path = planner_dir / "mind2web_task_history/part-00000.jsonl"
    planner_path.write_text(planner_path.read_text().splitlines()[0] + "\n")

    with pytest.raises(RuntimeError, match="coverage does not exactly match"):
        build_report(
            benchmark_root=benchmark_root,
            direct_predictions_dir=direct_dir,
            planner_predictions_dir=planner_dir,
        )


def test_action_accuracy_is_independent_of_box_validity() -> None:
    row = evaluate_one(
        target("a", "lclick", [100, 100, 200, 200]),
        {"prediction": "lclick [200,200,100,100]"},
    )

    assert row["parsed"] is False
    assert row["action_hit"] is True
    assert row["point_hit"] is False
    assert row["strict_hit"] is False
