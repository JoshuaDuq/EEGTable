PhysioNet motor movement
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 26 30 14

   * - Claim
     - Criterion
     - Observed
     - Result
   * - With a per-trial baseline-calibrated threshold, beta burst rate falls during movement (``burst_rate``)
     - lower in at least 90 percent of subjects; p below 1e-6
     - –
     - pass
   * - Burst rate is the count divided by the window length (``burst_count``, ``burst_rate``)
     - relative error below 1e-9
     - –
     - pass
   * - Mu bursts that survive the rest-median threshold are smaller during movement (``burst_amplitude``)
     - lower during movement in at least 90 percent of 20 subjects; p below 1e-4
     - lower during movement in 20 of 20
     - pass
   * - Movement suppresses mu and beta bursts over the hand areas (rest-median threshold) (``burst_rate``, ``fraction_above_threshold``, ``burst_duration``)
     - lower during movement in at least 80 percent of 20 subjects; p below 1e-3
     - burst_rate mu: lower during movement in 20 of 20 subjects, mean difference -0.355; burst_rate beta: lower during movement in 20 of 20 subjects, mean difference -0.250; fraction_above_threshold mu: lower during movement in 20 of 20 subjects, mean difference -0.088; fraction_above_threshold beta: lower during movement in 20 of 20 subjects, mean difference -0.090; burst_duration mu: lower during movement in 20 of 20 subjects, mean difference -0.025; burst_duration beta: lower during movement in 17 of 20 subjects, mean difference -0.007
     - pass
   * - eegtable check names a channel the recordings lack (``eegtable command``)
     - exit 1 and the channel named
     - –
     - pass
   * - eegtable check validates a recipe against real files and writes nothing (``eegtable command``)
     - exit 0; recordings listed; no output directory
     - –
     - pass
   * - eegtable run writes per-epoch and cross-trial bundles with a JSON progress stream (``eegtable command``, ``runner``)
     - two bundles of each kind, as the recipe says; a second run refused
     - –
     - pass
   * - Occipital coherence rises at whichever frequency is being flickered (``spectral_connectivity``)
     - driven condition above the other by 0.03 at 12 and at 15 Hz
     - –
     - pass
   * - Plain wPLI rises as trials fall; the debiased form does not (``spectral_connectivity``, ``wpli``)
     - wPLI at 10 trials above wPLI at 42 by 0.1; debiased within 0.05
     - wPLI 0.28 at 42 trials, 0.42 at 10; debiased 0.06 and 0.07
     - pass
   * - Every spectral connectivity method reproduces mne_connectivity (``spectral_connectivity``)
     - relative error below 1e-9 for all eight methods
     - –
     - pass
   * - Graph summaries of a real network lie in their defined ranges (``global_efficiency``, ``clustering_coefficient``)
     - efficiency in (0, 1], clustering in [0, 1]
     - –
     - pass
   * - Raw envelope correlation is the Fisher-z mean of per-trial envelope correlations (``envelope_correlation``)
     - relative error below 1e-9 on the first fifty pairs
     - –
     - pass
   * - Neighbouring electrodes correlate far more than distant ones until orthogonalized (``envelope_correlation``)
     - raw near minus far above 0.2; orthogonalized near below half of raw
     - raw near 0.73 against far 0.30; orthogonalized near 0.17 against far 0.16
     - pass
   * - wPLI reproduces a direct mne_connectivity call on the same epochs (``wpli``)
     - relative error below 1e-9 on every pair
     - –
     - pass
   * - Cross-fitted CSP components separate movement from rest in every subject (``csp_features``)
     - first component AUC below 0.5, second above; best above 0.6 in every subject
     - mu: best component AUC 0.67 to 0.96, mean 0.84 over 20 subjects; beta: best component AUC 0.66 to 0.99, mean 0.86 over 20 subjects
     - pass
   * - CSP filters and patterns are mutually inverse (``csp_features``)
     - filters times patterns transposed is the identity to 1e-8
     - –
     - pass
   * - Deep sleep against wake decodes across subjects (``loso_folds``, ``cross_fit_classification``, ``classification_metrics``)
     - balanced accuracy above 0.9 (chance 0.5)
     - balanced accuracy 0.96, per subject 0.91 to 0.98, 893 epochs
     - pass
   * - Movement against rest decodes across subjects, leave-one-subject-out (``loso_folds``, ``cross_fit_classification``, ``classification_metrics``)
     - balanced accuracy above 0.55 (chance 0.5)
     - balanced accuracy 0.68, per subject 0.52 to 0.86, 1740 epochs
     - pass
   * - Movement against rest decodes within subject from sensorimotor power (``stack_rows``, ``build_design``, ``within_subject_folds``, ``cross_fit_classification``, ``classification_metrics``)
     - balanced accuracy above 0.65 (chance 0.5)
     - balanced accuracy 0.79, per subject 0.60 to 0.95, 1740 epochs
     - pass
   * - Band power is the trapezoid integral of the Welch PSD; mean PSD is that per Hz (``integrated_band_power``, ``mean_psd``, ``Spectra.from_spectrum``)
     - relative error below 1e-12
     - –
     - pass
   * - The loaded epochs match the published recording (``BandSignal.from_epochs``)
     - 160 Hz, 64 channels, 45 movement trials
     - –
     - pass
   * - Hilbert ERDS ranks channels and trials like MNE's multitaper time-frequency ERDS (``erds_mean``, ``BandSignal.from_epochs``)
     - correlation above 0.75 by channel and by trial-channel cell, every subject
     - r by channel 0.79 to 0.99; by cell 0.79 to 0.98 over 20 subjects
     - pass
   * - Hand movement desynchronizes mu and beta over the hand areas (``erds_mean``)
     - per-subject mean dB below zero; one-sided t-test p below 0.05 across 20 subjects
     - mu: mean -3.5 dB, range -5.1 to -2.4, negative in 20 of 20, p = 3e-15; beta: mean -3.6 dB, range -4.4 to -2.9, negative in 20 of 20, p = 3e-18
     - pass
   * - Single-trial percent ERDS is right-skewed, which is why dB is the default (``erds_mean``)
     - trial mean above trial median in at least 75 percent of subjects
     - mean above median in 20 of 20 subjects
     - pass
   * - The runner's TSV bundle holds the API's numbers (``runner``, ``erds_mean``, ``integrated_band_power``)
     - relative error below 1e-9
     - –
     - pass
   * - With every correction off, the peak is the argmax bin (``peak_frequency``)
     - exact equality
     - –
     - pass
   * - Variance is NumPy's over the window (``variance``, ``Signal.from_epochs``)
     - relative error below 1e-12
     - –
     - pass
   * - The ERDS measures are the documented trace's mean, balance, slope and argmaxes (``erds_mean``, ``erd_magnitude``, ``erd_duration``, ``ers_magnitude``, ``ers_duration``, ``erds_slope``, ``erds_peak_latency``, ``erds_rebound_latency``)
     - relative error below 1e-9; durations tile the window
     - –
     - pass
   * - Movement deepens and lengthens desynchronization and shortens synchronization (``erd_magnitude``, ``erd_duration``, ``ers_magnitude``)
     - ERD larger and ERS smaller during movement in 90 percent of subjects; p below 1e-5
     - –
     - pass
   * - The default six-cycle persistence keeps the onset's false-positive rate low (``erds_onset_latency``)
     - above 0.9 at a fixed 100 ms; below 0.15 at the default; lower still at 1500 ms
     - beta rest trials with an onset: 1.00 at 100 ms, 0.05 at the six-cycle default, 0.00 at 1500 ms
     - pass
   * - Four resting microstate templates explain most of the topographic variance (``segment``)
     - global explained variance above 0.5; no two templates correlate above 0.9
     - global explained variance 0.63
     - pass
   * - Coverage sums to one and equals duration times occurrence; transitions sum to one (``microstate_coverage``, ``microstate_duration``, ``microstate_occurrence``, ``microstate_transitions``)
     - relative error below 1e-9
     - –
     - pass
   * - Microstates last tens of milliseconds (``microstate_duration``)
     - median duration of every state between 30 and 150 ms
     - median durations 68 to 96 ms
     - pass
   * - In MNE's CSP decoding pipeline, eegtable's CSP separates movement from rest about as well as mne.decoding.CSP (``CommonSpatialPattern``)
     - mean accuracy over 20 subjects within 0.05 of MNE's, on the same splits
     - mean accuracy over 20 subjects: MNE 0.757, eegtable 0.731
     - pass
   * - Orthogonalized envelope correlation is the Fisher-z mean of mne-connectivity's per-trial estimate (``envelope_correlation``)
     - relative error below 1e-9 on every pair
     - 2016 pairs over 42 trials, maximum relative error 7.3e-16
     - pass
   * - Support is partial where wavelets overrun the window while coverage stays full (``Spectra.from_tfr``)
     - support at most 1 and below 1 somewhere; coverage exactly 1
     - –
     - pass
   * - The wavelet density does not change when the recording is resampled (``Spectra.from_tfr``, ``mean_tfr_power``)
     - median ratio at 80 against 160 Hz within 2 percent of one
     - median ratio 80 Hz against 160 Hz: 0.999
     - pass
   * - Wavelet power divided by the sampling rate matches a Welch PSD of the same window (``Spectra.from_tfr``, ``mean_tfr_power``)
     - median ratio within 0.9 to 1.1; log correlation above 0.95
     - mu: median ratio 0.98, log correlation 0.98; beta: median ratio 1.02, log correlation 0.98
     - pass
   * - A multitaper spectrum without full normalization is refused (``Spectra.from_spectrum``)
     - ValueError naming the normalization
     - –
     - pass
   * - Multitaper and Welch band power agree (``integrated_band_power``)
     - median ratio within 0.9 to 1.1; log correlation above 0.9
     - mu: median ratio 0.98, log correlation 0.92; beta: median ratio 1.00, log correlation 0.94
     - pass
   * - Multitaper band power integrates edge to edge on a grid that misses the band edges (``integrated_band_power``, ``Spectra.from_spectrum``)
     - relative error below 1e-9 against an edge-interpolated trapezoid
     - –
     - pass
   * - The passband helpers read the recording's filter edges (``passband_fraction``, ``check_passband``)
     - fraction 1 inside and 0.5 straddling; refusal below the high-pass edge
     - –
     - pass
   * - detect-bads, reject, and interpolate are PyPREP NoisyChannels, autoreject fit/transform, and interpolate_bads with the same seeds (``preprocessing``)
     - identical candidate lists, retained epochs, repair labels, and samples
     - –
     - pass
   * - Motor-task IRASA signed component integrals agree with NeuroDSP and SciPy quadrature (``irasa``)
     - rtol=1e-10, atol=1e-24 including off-grid edges near both fit-range boundaries
     - low: max absolute component integral error 6.46e-27 V^2; mu: max absolute component integral error 3.23e-27 V^2; beta: max absolute component integral error 6.46e-27 V^2; high: max absolute component integral error 2.02e-28 V^2
     - pass
   * - Signed IRASA components sum to an independently computed SciPy Welch band integral (``irasa``)
     - rtol=1e-10, atol=1e-24 in four bands on all selected motor trials/channels
     - four bands, six trials, two channels; max relative total-power error 6.56e-16
     - pass
   * - Motor-task IRASA aperiodic offset and slope agree with direct NeuroDSP fits (``irasa``)
     - rtol=1e-10, atol=1e-12 on six trials at C3 and C4
     - six trials, two channels; max absolute offset/slope error 0.00e+00
     - pass

ERP CORE Flankers
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 26 30 14

   * - Claim
     - Criterion
     - Observed
     - Result
   * - Wrong presses produce a negativity at FCz in the following 100 ms (``mean_amplitude``, ``peak_amplitude``)
     - error trials below correct by more than 3 microvolts; p below 1e-6
     - mean 0 to 100 ms at FCz: +3.9 microvolts on 346 correct trials, -7.1 on 54 errors
     - pass
   * - Single-trial N1 latency over lateral occipital cortex clusters near 180 ms (``peak_latency``)
     - median in 150 to 220 ms; interquartile range below 50 ms; 60 percent within 40 ms
     - median 183 ms, interquartile range 165 to 198 ms on a 150 ms window
     - pass
   * - The P3 area over Pz and CPz is positive on most trials (``area_under_curve``)
     - positive on more than 70 percent of trials; pre-stimulus area near zero
     - positive on 82 percent of 400 trials
     - pass
   * - Peak amplitude, peak latency and area are their documented formulas (``peak_amplitude``, ``peak_latency``, ``area_under_curve``)
     - exact for peaks; relative error below 1e-9 for area
     - –
     - pass
   * - The loaded epochs match the published task (``Signal.from_epochs``)
     - 1024 Hz, 400 stimuli, more than 300 correct and 30 error trials
     - –
     - pass
   * - Peak latency and amplitude are what MNE's Evoked.get_peak reports (``peak_latency``, ``peak_amplitude``)
     - identical on every channel for negative, positive and absolute peaks
     - 30 channels, three polarities, all identical
     - pass
   * - reject, reference, resample, crop-epochs, baseline are drop_bad, set_eeg_reference, decimate, crop, apply_baseline (``preprocessing``)
     - identical events and drop log; samples within 1e-15 V
     - –
     - pass
   * - epoch is mne.Epochs on the acquisition grid with baseline, proj, and decim off (``preprocessing``)
     - identical events, drop log, and samples; events past the crop dropped, not shifted
     - –
     - pass
   * - notch and filter are Raw.notch_filter and Raw.filter on physiology channels only (``preprocessing``)
     - samples within 1e-15 V; 60 Hz bin reduced, 20 Hz bin within 0.5%
     - –
     - pass
   * - prepare and crop-raw are set_bipolar_reference and Raw.crop (``preprocessing``)
     - identical samples, first_samp, and channel order
     - –
     - pass
   * - apply-artifact is ICA.apply of the saved fit with the reviewed exclusion (``preprocessing``)
     - samples within 1e-15 V; mean \|corr(FP1, VEOG)\| across epochs down by a quarter
     - –
     - pass

Sleep-EDF
^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 26 30 14

   * - Claim
     - Criterion
     - Observed
     - Result
   * - A broadband gain changes band power but not the power above the 1/f fit (``periodic_power``)
     - under a tenfold gain, periodic power within 1e-9 of unchanged, mean PSD tenfold
     - –
     - pass
   * - Dividing by the fitted line leaves a spectrum centred on one (``aperiodic_ratio``)
     - median log10 ratio within 0.05 of zero
     - –
     - pass
   * - The robust slope tracks plain least squares (``aperiodic``)
     - correlation above 0.98; median absolute difference below 0.1
     - –
     - pass
   * - The aperiodic slope steepens wake, N2, N3 (``aperiodic``)
     - median ordering holds; wake-vs-N3 AUC below 0.1, both derivations, both subjects
     - median slope SC4001 Pz-Oz: wake -1.42, N2 -2.77, N3 -3.24; SC4011 Pz-Oz: wake -0.47, N2 -2.63, N3 -3.23
     - pass
   * - With peak rejection off, the aperiodic fit is least squares in log-log space (``aperiodic``)
     - relative error below 1e-9
     - –
     - pass
   * - Sample entropy and Higuchi dimension are lower in N3 than in wake (``sample_entropy``, ``higuchi_fractal_dimension``)
     - wake-vs-N3 AUC below 0.2 on both derivations
     - sample_entropy: AUC Fpz-Cz 0.07, Pz-Oz 0.06; higuchi_fractal_dimension: AUC Fpz-Cz 0.00, Pz-Oz 0.00
     - pass
   * - Multiscale entropy at scale one is sample entropy (``multiscale_entropy``, ``sample_entropy``)
     - relative error below 1e-9
     - –
     - pass
   * - Sleep-EDF DFA exponents agree with direct AntroPy calls on contiguous windows (``detrended_fluctuation``)
     - rtol=1e-12, atol=1e-12; finite values in both derivations and windows
     - 80 estimates across two nights and five stages; max absolute error 0.00e+00
     - pass
   * - Sleep-EDF binary complexity agrees with AntroPy after explicit symbolization (``lempel_ziv_complexity``)
     - rtol=1e-12, atol=1e-12 for median and mean thresholds; finite values
     - median: 80 estimates across two nights and five stages; max absolute error 0.00e+00; mean: 80 estimates across two nights and five stages; max absolute error 0.00e+00
     - pass
   * - Sleep-EDF ordinal entropy agrees with direct AntroPy calls in each channel/window (``permutation_entropy``)
     - rtol=1e-12, atol=1e-12 for order/delay (3, 1) and (4, 2); finite values
     - order=3, delay=1: 80 estimates across two nights and five stages; max absolute error 0.00e+00; order=4, delay=2: 80 estimates across two nights and five stages; max absolute error 0.00e+00
     - pass
   * - Band power reproduces the staging accuracy of MNE's sleep-classification tutorial (``mean_psd``)
     - random forest trained on one night scores the other within 0.03 of the tutorial's own features
     - accuracy on SC4011: tutorial 0.684, eegtable 0.683
     - pass
   * - Held-out permutation importance points at the derivation that carries sleep depth (``permutation_importance_over_folds``)
     - Pz-Oz carries more than 60 percent of the total; every value finite
     - Pz-Oz carries 0.81 of the total importance
     - pass
   * - Conformal intervals reach nominal coverage on exchangeable epochs (``prediction_intervals``)
     - split and CV+ within 0.85 to 0.95 at 90 percent nominal; quantile at least 0.9
     - split: coverage 0.894 at 90 percent nominal; cv_plus: coverage 0.909 at 90 percent nominal; quantile: coverage 0.998 at 90 percent nominal
     - pass
   * - Phase-amplitude coupling is the documented mean vector length (``pac``)
     - relative error below 1e-9
     - –
     - pass
   * - Raw mean vector length stays well above zero with the coupling destroyed (``pac``)
     - surrogate median above 0.03, all values in [0, 1]
     - –
     - pass
   * - Refitting under within-subject permuted targets puts the null at zero (``permutation_test``)
     - null within 0.2 of zero; p below 0.05; observed statistic recomputed identically
     - null within 0.06 of zero over 50 permutations, p = 0.02
     - pass
   * - Sleep depth 0 to 3 is recovered across subjects by leave-one-subject-out ridge (``stack_rows``, ``build_design``, ``loso_folds``, ``cross_fit_regression``, ``regression_metrics``, ``subject_level_r``)
     - subject-level r above 0.7, and in every subject; R squared above 0.5
     - subject-level r 0.88 (lowest subject 0.86), R squared 0.66, 2511 epochs
     - pass
   * - Relative delta rises and relative alpha and beta fall from wake to N3 at Pz-Oz (``integrated_band_power``, ``band_ratio``)
     - wake-vs-N3 AUC above 0.8 (delta), below 0.2 (alpha), below 0.1 (beta)
     - AUC SC4001: delta 0.94, alpha 0.01, beta 0.00; SC4011: delta 0.91, alpha 0.02, beta 0.00
     - pass
   * - The loaded epochs match the published recording (``Signal.from_epochs``)
     - 100 Hz, Fpz-Cz and Pz-Oz, at least 50 wake and 50 N3 epochs
     - –
     - pass
   * - Complexity is higher in N3 than in wake at Pz-Oz (``hjorth_complexity``)
     - wake-vs-N3 AUC above 0.8, both subjects
     - –
     - pass
   * - Mobility is the documented formula in hertz (``hjorth_mobility``)
     - relative error below 1e-9
     - –
     - pass
   * - Mobility falls wake, N2, N3 at Pz-Oz (``hjorth_mobility``)
     - median ordering holds; wake-vs-N3 AUC below 0.1
     - –
     - pass
   * - The slow-wave fraction increases monotonically N1, N2, N3 (``integrated_band_power``, ``band_ratio``)
     - median ordering holds on both derivations, both subjects
     - –
     - pass
   * - Edge, centroid, entropy and bandwidth are lower in N3 than in wake at Pz-Oz (``spectral_edge``, ``spectral_centroid``, ``spectral_entropy``, ``spectral_bandwidth``)
     - medians ordered; wake-vs-N3 AUC below 0.25
     - –
     - pass
   * - Asymmetry is right minus left on log power (``asymmetry``)
     - relative error below 1e-12
     - –
     - pass
   * - Slow waves are larger and smoother than wake EEG at Pz-Oz (``root_mean_square``, ``peak_to_peak``, ``line_length``)
     - wake-vs-N3 AUC above 0.7 for RMS and peak-to-peak, below 0.3 for line length
     - –
     - pass
   * - The time-domain measures are their NumPy and SciPy formulas (``root_mean_square``, ``peak_to_peak``, ``mean_amplitude``, ``amplitude_quantile``, ``skewness``, ``kurtosis``, ``line_length``)
     - relative error below 1e-7
     - –
     - pass

MNE SSVEP
^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 26 30 14

   * - Claim
     - Criterion
     - Observed
     - Result
   * - ITPC is the documented formula, trials first then time (``itpc``)
     - relative error below 1e-9
     - –
     - pass
   * - Phase locks across trials at the flicker frequency only in the driven condition (``itpc``, ``ppc``)
     - driven above undriven by 0.1 for ITPC and PPC, at 12 and at 15 Hz
     - 12 Hz band: itpc 0.49 driven, 0.31 not; ppc 0.19 driven, 0.02 not; 15 Hz band: itpc 0.53 driven, 0.40 not; ppc 0.24 driven, 0.10 not
     - pass
   * - Pairwise phase consistency sits near zero without a locked response (``ppc``)
     - absolute value below 0.1
     - –
     - pass
   * - On a narrow band the zero-crossing rate is twice the Hjorth mobility (Rice) (``zero_crossing_rate``, ``hjorth_mobility``)
     - within 6 percent per cell and 1 percent on average
     - –
     - pass
   * - SSVEP fixed/knee model parameters, diagnostics and selected peaks match specparam (``spectral_parameterization``)
     - rtol=1e-10, atol=1e-12 on every Oz trial; absent peaks are NaN and flagged
     - fixed: 20 Oz trials, 208 finite parameters, 12 absent peak parameters; max absolute error 0.00e+00; knee: 20 Oz trials, 234 finite parameters, 6 absent peak parameters; max absolute error 0.00e+00
     - pass
   * - With smoothing off and alpha excluded, the peak lands on the flicker frequency (``peak_frequency``)
     - condition medians within 0.3 Hz of 12 and 15 Hz; 70 percent of trials within
     - medians 12.00 and 15.00 Hz; 18 of 20 trials within 0.3 Hz
     - pass
   * - Occipital power at 12 Hz against 15 Hz separates the flicker conditions on every trial (``integrated_band_power``, ``band_ratio``)
     - smallest 12 Hz-trial ratio above largest 15 Hz-trial ratio
     - 12 Hz trials at least 0.87, 15 Hz trials at most 0.02 (log10 ratio)
     - pass
   * - The line at the flicker frequency stands above its 3 Hz neighbourhood only when driven (``mean_psd``, ``band_ratio``)
     - driven trials' minimum SNR above undriven trials' maximum, at 12 and at 15 Hz
     - –
     - pass
   * - The second harmonic (24 against 30 Hz) separates the conditions on every trial (``integrated_band_power``, ``band_ratio``)
     - smallest 12 Hz-trial ratio above largest 15 Hz-trial ratio
     - –
     - pass
