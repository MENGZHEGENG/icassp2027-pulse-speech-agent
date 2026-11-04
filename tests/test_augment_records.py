import math

import numpy as np

from pulse.augment_records import apply_gain, augmentation_suffix, mix_white_noise, resample_linear, rms, safe_stem


def test_mix_white_noise_approximately_matches_snr() -> None:
    rng = np.random.default_rng(0)
    clean = np.sin(np.linspace(0.0, 20.0 * math.pi, 16000, dtype=np.float32))
    mixed = mix_white_noise(clean, 10.0, rng)
    noise = mixed - clean
    measured_snr = 20.0 * math.log10(rms(clean) / rms(noise))
    assert 9.5 <= measured_snr <= 10.5


def test_resample_linear_speed_changes_length() -> None:
    audio = np.linspace(-1.0, 1.0, 100, dtype=np.float32)
    faster = resample_linear(audio, 1.25)
    slower = resample_linear(audio, 0.8)
    assert len(faster) == 80
    assert len(slower) == 125
    assert faster.dtype == np.float32


def test_apply_gain_scales_amplitude() -> None:
    audio = np.array([0.25, -0.25], dtype=np.float32)
    gained = apply_gain(audio, 6.0)
    assert np.allclose(gained, audio * (10 ** (6.0 / 20.0)))


def test_safe_stem_and_suffix_are_stable() -> None:
    assert safe_stem("speaker/id 001") == "speaker-id-001"
    assert augmentation_suffix(10.0, 1.0, 0.0) == "snr10"
    assert augmentation_suffix(None, 1.1, -3.0) == "speed1.1-gain-3"
