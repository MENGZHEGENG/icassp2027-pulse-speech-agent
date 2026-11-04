from pulse.simulate_policy import PulsePolicy, make_trial, score_policy
import random


def test_pulse_simulation_runs() -> None:
    rng = random.Random(7)
    trials = [make_trial(rng, 0.5) for _ in range(20)]
    result = score_policy("pulse", trials, PulsePolicy(entity_risk_threshold=0.1))
    assert result.trials == 20
    assert 0 <= result.task_success <= 1
