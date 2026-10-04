Cycle-by-Cycle Waveform Features
=================================

``cycle_features`` uses `ByCycle compute_features
<https://bycycle-tools.github.io/bycycle/generated/bycycle.features.compute_features.html>`__
to locate trough-to-trough cycles and characterize their waveform. Install
``eegtable[cycles]``. The project's declared pandas dependency is ``>=2.0,<3.0``.
The function signature is in :doc:`/api/dynamics`.

Detection and Window Support
----------------------------

Cycle detection runs once on the complete epoch for each channel and requested
band. A three-cycle narrowband filter locates extrema; amplitudes and waveform
shape are measured on the original broadband signal. Finite, nonconstant
``Signal`` inputs and epochs longer than this filter are required. Band limits
must lie inside the recording passband and strictly between zero and Nyquist.
Do not supply a narrowband-filtered waveform to study its original asymmetry.

For each analysis window, only cycles whose preceding and following troughs
both lie inside its bounds count. Detection and consistency use the full epoch,
including neighboring cycles outside the window. ByCycle's first and last
cycles cannot be bursting under consistency detection. The trough-duration
term for peak/trough symmetry uses the preceding zero crossing, so its support
can extend before the cycle's first trough.

Outputs and Aggregation
-----------------------

Waveform summaries average only burst-labelled cycles:

* ``cycle_period``: trough-to-trough period, in seconds.
* ``cycle_rise_time`` and ``cycle_decay_time``: trough-to-peak and
  peak-to-trough durations, in seconds.
* ``cycle_rise_decay_symmetry``: fraction of a cycle spent rising.
* ``cycle_peak_trough_symmetry``: peak duration divided by peak plus preceding
  trough duration, from waveform zero crossings.
* ``cycle_amplitude``: mean of rise and decay voltage excursions. For a sine
  of amplitude :math:`a`, this is approximately :math:`2a`. Input amplitude
  must be in volts for the reported unit.

``cycle_count`` counts all complete cycles and ``cycle_burst_count`` counts
those labelled bursting. ``cycle_burst_fraction`` is their quotient. It is a
fraction of cycles, distinct from the fraction of time spent bursting.

Burst Criteria
--------------

Consistency thresholds default to amplitude fraction 0, amplitude consistency
0.5, period consistency 0.5, monotonicity 0.8, and a run of at least three cycles.
Override the named ``burst_thresholds`` after inspecting the data; suitable
thresholds depend on the signal and hypothesis. See the `official burst
threshold definitions
<https://bycycle-tools.github.io/bycycle/generated/bycycle.burst.detect_bursts_cycles.html>`__.
All settings are recorded in the feature identity.

Undefined Results and Limitations
---------------------------------

A window without complete cycles has count zero, undefined burst fraction,
and ``cycle_no_complete_cycles``. A window without bursts has burst count zero,
undefined shape summaries, and ``cycle_no_burst``. Its burst fraction is zero
when complete cycles exist. Group flags mark any affected member channel;
ROI/global values follow the library's arithmetic channel aggregation.
Coverage measures the original samples in the window and does not quantify
burst detection quality. Upstream errors surface.

ROI/global counts are arithmetic means of channel counts and can therefore
be noninteger. These summaries do not detect a single shared multichannel
cycle. Waveform shape remains sensitive to noise, preprocessing, and the
presence of overlapping activity in the broadband trace; a burst label is
an algorithmic classification under the chosen criteria.
