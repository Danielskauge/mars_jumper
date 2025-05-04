from __future__ import annotations
import numpy as np
from numpy.random import Generator
import matplotlib.pyplot as plt

"""
random_tone_burst.py
====================

Target trajectory generator for actuator‑net data collection
-----------------------------------------------------------

Signal = slow drift sine  +  single fast tone, with automatic
alternation between **bursts** (fast tone ON) and **holds** (flat).

Key simplifications
-------------------
* *Fast‑tone amplitude* is drawn as a **fraction of half‑range**;
  if the sum overshoots the joint limits we simply **clip**.
* No explicit safety margin needed in the math.

Why it's useful
---------------
* **Representative**  – drift explores the full angle range; bursts add
  rich dynamics; holds give steady‑state samples.
* **Persistently exciting**  – each burst lasts ≥ min_cycles cycles of its
  fast tone, so short history windows are well‑conditioned.
* **Simple**  – one class, two public methods: ``step(dt)`` and
  ``generate_buffer(duration, fs)``.
"""




class RandomToneBurstGenerator:
    """
    Parameters
    ----------
    theta_min, theta_max : float
        Joint limits [rad].  Output is clipped to these bounds.
    drift_freq : float
        Frequency of the slow drift sine [Hz].
    drift_amp_ratio : float
        Drift amplitude as fraction of half‑range (0–1).
    f_range : (float, float)
        Log‑uniform bounds for fast‑tone frequency [Hz].
    a_ratio : (float, float)
        Uniform bounds for **fast‑tone amplitude / half‑range**.
        Example (0.1, 1.0) lets the fast sine reach the limit.
    burst_dur, hold_dur : (float, float)
        Min/max duration for bursts and holds [s].
    ramp_time : float
        Linear edge‑blend time [s] for burst ↔ hold transitions.
    min_cycles : int
        Fast tone is held for at least this many cycles before redraw.
    rng : np.random.Generator | None
        Custom RNG for reproducible sequences.
    """

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        *,
        theta_min: float = -0.9,
        theta_max: float = 0.9,
        drift_freq: float = 0.05,
        drift_amp_ratio: float = 0.45,
        f_range: tuple[float, float] = (0.4, 6.0),
        a_ratio: tuple[float, float] = (0.1, 1.0),
        burst_dur: tuple[float, float] = (0.5, 3.0),
        hold_dur: tuple[float, float] = (0.3, 2.0),
        ramp_time: float = 0.015,
        min_cycles: int = 6,
        rng: Generator | None = None,
    ) -> None:
        # ---- limits & drift amplitude ----------------------------------
        self.th_min = theta_min
        self.th_max = theta_max
        self.half_range = 0.5 * (theta_max - theta_min)

        self.drift_freq = drift_freq
        self.drift_amp = drift_amp_ratio * self.half_range

        # ---- fast‑tone parameter ranges --------------------------------
        self.fmin, self.fmax = f_range
        self.ar_min, self.ar_max = a_ratio

        self.burst_min, self.burst_max = burst_dur
        self.hold_min, self.hold_max = hold_dur
        self.ramp_time = ramp_time
        self.min_cycles = min_cycles
        self.rng = rng or np.random.default_rng()

        # ---- initial generator state -----------------------------------
        self.t = 0.0                       # global time
        self.drift_phase = 0.0

        self.in_burst = True
        self.segment_t = 0.0
        self.segment_len = self.rng.uniform(*burst_dur)
        self.blend_timer = ramp_time

        self._draw_fast_tone(keep_val=0.0)

        # hold bookkeeping
        self.holding = False
        self.hold_value = 0.0

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #
    def _draw_fast_tone(self, *, keep_val: float) -> None:
        """Resample fast‑tone frequency, amplitude, phase (continuous)."""
        self.fast_freq = float(
            np.exp(self.rng.uniform(np.log(self.fmin), np.log(self.fmax)))
        )
        amp_ratio = self.rng.uniform(self.ar_min, self.ar_max)
        self.fast_amp = amp_ratio * self.half_range

        # continuity: choose phase so A*sin(phi) = keep_val
        ratio = np.clip(keep_val / self.fast_amp, -1.0, 1.0)
        self.fast_phase = float(np.arcsin(ratio))

        self.cycles = 0.0          # reset cycle counter
        self.blend_timer = self.ramp_time

    def _next_segment(self) -> None:
        """Switch between burst and hold, sample new segment length."""
        self.in_burst = not self.in_burst
        self.segment_t = 0.0

        if self.in_burst:
            self.segment_len = self.rng.uniform(self.burst_min, self.burst_max)
            self._draw_fast_tone(keep_val=0.0)
        else:
            self.segment_len = self.rng.uniform(self.hold_min, self.hold_max)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def step(self, dt: float) -> float:
        """
        Advance the generator by `dt` seconds and return the next target.

        Parameters
        ----------
        dt : float
            Time step [s].

        Returns
        -------
        theta_cmd : float
            Target joint angle at the new time step [rad],
            **clipped** to [theta_min, theta_max].
        """
        # --- segment timing --------------------------------------------
        self.t += dt
        self.segment_t += dt
        if self.segment_t >= self.segment_len:
            self._next_segment()

        # --- gate value with linear ramps ------------------------------
        g = 1.0 if self.in_burst else 0.0
        if self.blend_timer > 0.0:
            g *= 1.0 - self.blend_timer / self.ramp_time
            self.blend_timer -= dt

        # --- drift component -------------------------------------------
        if g > 0.999:
            self.drift_phase += 2 * np.pi * self.drift_freq * dt
        theta_drift = self.drift_amp * np.sin(self.drift_phase)

        # --- fast tone -------------------------------------------------
        theta_fast = self.fast_amp * np.sin(
            2 * np.pi * self.fast_freq * self.t + self.fast_phase
        )
        self.cycles += self.fast_freq * dt

        # resample fast tone within burst (once min_cycles satisfied)
        if self.in_burst and self.cycles >= self.min_cycles and self.segment_t >= 0.5 * self.segment_len:
            self._draw_fast_tone(keep_val=theta_fast)
            theta_fast = self.fast_amp * np.sin(
                2 * np.pi * self.fast_freq * self.t + self.fast_phase
            )

        # --- combine with hold logic -----------------------------------
        if g > 0.999:                          # dynamic burst
            theta_cmd = theta_drift + theta_fast
            self.hold_value = theta_cmd        # remember for next hold
            self.holding = False
        else:                                  # flat hold
            if not self.holding:
                self.holding = True
            theta_cmd = self.hold_value

        # --- final clipping to hard limits -----------------------------
        return float(np.clip(theta_cmd, self.th_min, self.th_max))

    def generate_buffer(self, duration: float, fs: int) -> np.ndarray:
        """
        Pre‑compute a full trajectory.

        Parameters
        ----------
        duration : float
            Length in seconds.
        fs : int
            Sampling rate [Hz].

        Returns
        -------
        ndarray
            Array of shape `(duration*fs,)` with the target angle.
        """
        N = int(duration * fs)
        dt = 1.0 / fs
        buf = np.empty(N)
        for i in range(N):
            buf[i] = self.step(dt)
        return buf

# --- Main execution block for plotting ---
if __name__ == "__main__":
    # Simulation parameters
    SIM_DURATION = 20.0  # seconds
    SAMPLE_RATE = 100   # Hz

    # Instantiate the generator with default parameters
    generator = RandomToneBurstGenerator()

    # Generate the trajectory
    theta_target = generator.generate_buffer(duration=SIM_DURATION, fs=SAMPLE_RATE)

    # Create time vector
    time_vec = np.linspace(0, SIM_DURATION, len(theta_target), endpoint=False)

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(time_vec, theta_target)
    plt.title("Generated Random Tone Burst Trajectory")
    plt.xlabel("Time (s)")
    plt.ylabel("Target Angle (rad)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("random_tone_burst_trajectory.png")
