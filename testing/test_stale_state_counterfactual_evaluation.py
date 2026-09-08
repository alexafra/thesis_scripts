from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).parents[1] / "stale_state_counterfactual_evaluation.py"
SPEC = spec_from_file_location("stale_state_counterfactual_evaluation", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
counterfactual = module_from_spec(SPEC)
sys.modules[SPEC.name] = counterfactual
SPEC.loader.exec_module(counterfactual)


def _event(*, end=34):
    return counterfactual.PlateauEvent(
        split="validation",
        dataset_path="/dataset/validation",
        loader_position=0,
        episode_index=9,
        event_index=0,
        fresh_proxy_frame=5,
        plateau_end_frame=end,
        plateau_frames=end - 4,
        stale_candidate_frames=end - 5,
        fps=30.0,
        duration_s=(end - 5) / 30,
        left_action_range=0.0,
        right_action_range=0.2,
        strong=True,
        moving_hand="right",
        task="stack the three red cups.",
    )


def test_detector_keeps_one_handed_motion_when_other_hand_is_stationary():
    frames = 50
    left_state = np.zeros((frames, 2), dtype=np.float32)
    right_state = np.column_stack(
        (
            np.arange(frames, dtype=np.float32) * 0.001,
            np.arange(frames, dtype=np.float32) * 0.002,
        )
    )
    right_state[5:35] = right_state[5]
    left_action = np.zeros((frames, 2), dtype=np.float32)
    right_action = np.zeros((frames, 2), dtype=np.float32)
    right_action[5:35, 0] = np.linspace(0.0, 0.2, 30)
    frame = pd.DataFrame(
        {
            "state.left_hand": list(left_state),
            "state.right_hand": list(right_state),
            "action.left_hand": list(left_action),
            "action.right_hand": list(right_action),
        }
    )

    events = counterfactual.detect_plateaus(
        frame,
        split="validation",
        dataset_path=Path("/dataset/validation"),
        loader_position=0,
        episode_index=9,
        task="stack the three red cups.",
        fps=30.0,
        minimum_frames=25,
        strong_action_range=0.1,
    )

    assert len(events) == 1
    event = events[0]
    assert event.fresh_proxy_frame == 5
    assert event.plateau_end_frame == 34
    assert event.plateau_frames == 30
    assert event.strong is True
    assert event.moving_hand == "right"


def test_offset_eight_aligns_all_three_strategies_to_same_expert_window():
    decisions, skipped = counterfactual.plan_decisions(
        _event(),
        episode_length=50,
        action_horizon=32,
        execution_horizon=8,
        offsets=[8],
    )
    assert skipped == []
    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.anchor_frame == 13
    assert decision.target_start_frame == 13
    assert decision.target_end_frame == 20
    assert decision.window_all_stale is True

    fresh_chunk = np.arange(64, dtype=np.float64).reshape(32, 2)
    stale_chunk = 1000 + np.arange(64, dtype=np.float64).reshape(32, 2)
    expert_actions = 2000 + np.arange(100, dtype=np.float64).reshape(50, 2)
    previous_command = expert_actions[5]
    expert = expert_actions[13:21]
    strategies = counterfactual.build_strategy_arrays(
        stale_chunk=stale_chunk,
        fresh_chunk=fresh_chunk,
        pre_dropout_command=previous_command,
        expert=expert,
        decision_offset=8,
        execution_horizon=8,
    )

    np.testing.assert_array_equal(strategies["stale_state_inference"], stale_chunk[0:8])
    np.testing.assert_array_equal(strategies["continue_previous_chunk"], fresh_chunk[8:16])
    np.testing.assert_array_equal(
        strategies["hold_pre_dropout"],
        np.repeat(previous_command[None, :], 8, axis=0),
    )
    assert all(values.shape == expert.shape for values in strategies.values())


def test_decision_boundary_is_flagged_and_never_padded():
    full, skipped = counterfactual.plan_decisions(
        _event(end=20),
        episode_length=50,
        action_horizon=32,
        execution_horizon=8,
        offsets=[8, 25],
    )
    assert len(full) == 1
    assert full[0].window_all_stale is True
    assert skipped[0]["reason"] == "decision_anchor_after_plateau"

    partial, skipped = counterfactual.plan_decisions(
        _event(end=19),
        episode_length=50,
        action_horizon=32,
        execution_horizon=8,
        offsets=[8],
    )
    assert skipped == []
    assert partial[0].window_all_stale is False

    invalid, skipped = counterfactual.plan_decisions(
        _event(end=40),
        episode_length=50,
        action_horizon=32,
        execution_horizon=8,
        offsets=[25],
    )
    assert invalid == []
    assert skipped[0]["reason"] == "previous_chunk_slice_exceeds_action_horizon"


def test_metric_row_uses_physical_errors_and_strategy_specific_history():
    expert = np.array([[1.0, 2.0], [2.0, 4.0]])
    prediction = np.array([[2.0, 2.0], [4.0, 2.0]])
    previous = np.array([1.0, 1.0])
    metrics = counterfactual._metric_row(
        values=prediction,
        expert=expert,
        fresh_proxy_command=previous,
        strategy_previous_command=np.array([2.0, 1.5]),
        group_indices=np.array([0, 1]),
    )

    assert metrics["sample_count"] == 4
    assert metrics["mae"] == 1.25
    assert metrics["mse"] == 2.25
    assert metrics["jump_from_fresh_proxy_command_mae"] == 1.0
    assert metrics["entry_jump_from_strategy_history_mae"] == 0.25
    assert metrics["within_window_mean_absolute_slew"] == 1.0


def test_plateau_starting_at_episode_start_has_no_fresh_proxy():
    event = counterfactual.PlateauEvent(
        **{
            **_event().__dict__,
            "fresh_proxy_frame": 0,
            "plateau_end_frame": 29,
            "plateau_frames": 30,
            "stale_candidate_frames": 29,
        }
    )
    decisions, skipped = counterfactual.plan_decisions(
        event,
        episode_length=50,
        action_horizon=32,
        execution_horizon=8,
        offsets=[8],
    )

    assert decisions == []
    assert skipped[0]["reason"] == "plateau_starts_at_episode_start"


def test_noise_seed_is_stable_and_repeat_specific():
    kwargs = {
        "split": "validation",
        "episode_index": 42,
        "event_index": 3,
        "repeat": 0,
    }
    first = counterfactual._stable_noise_seed(42, **kwargs)
    assert first == counterfactual._stable_noise_seed(42, **kwargs)
    assert first != counterfactual._stable_noise_seed(42, **{**kwargs, "repeat": 1})
