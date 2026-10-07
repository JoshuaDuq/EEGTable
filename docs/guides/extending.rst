Your Own Measures
=================

.. rst-class:: hero-lede

   Write the computation; EEGTable supplies bands, windows, ROIs, coverage,
   support and provenance.

A kernel reduces arrays. :func:`~eegtable.spectral_measure` and
:func:`~eegtable.signal_measure` run it through the machinery the built-in
measures use. Bands and windows become columns, and channels are averaged into
ROIs and the global mean. Non-finite input lowers coverage, Morlet support is
carried, and every column is named from its complete definition.

Spectral kernels
----------------

.. code-block:: python

   import numpy as np
   import eegtable as ef

   def peak_power(power, freqs, weights):
       # power: (epochs, channels, windows, bins of one band), NaN where not finite.
       return np.nanmax(power, axis=-1)

   spectra = ef.Spectra.welch(epochs, recording="sub-01", fmin=1.0, fmax=45.0)
   table = ef.spectral_measure(
       spectra,
       peak_power,
       measure="peak_power",
       unit="V^2/Hz",
       bands=[ef.Band("alpha", 8.0, 13.0)],
       groups={"central": ["C3", "Cz", "C4"]},
   )

The kernel receives a band's bins and their trapezoid-rule weights in Hz, half a
spacing at each end bin, so ``np.nansum(power * weights, axis=-1)`` integrates
from the first bin to the last. It returns values
shaped ``(epochs, channels, windows)``, or ``(values, flags)`` with boolean flags
of that shape. ``baseline`` and ``normalize`` work as they do for
:func:`~eegtable.integrated_band_power`.

Time-domain kernels
-------------------

.. code-block:: python

   def largest_deflection(trace, times):
       # trace: (epochs, channels, samples of one window), NaN where not finite.
       return np.nanmax(np.abs(trace), axis=-1)

   signal = ef.Signal.from_epochs(epochs, recording="sub-01")
   table = ef.signal_measure(
       [signal],
       largest_deflection,
       measure="largest_deflection",
       unit="V",
       windows=[ef.Window("stimulus", 0.0, 0.5)],
   )

A :class:`~eegtable.BandSignal` is measured on its envelope, and its band
becomes a field of each column.

Identity
--------

The measure label is yours to choose. The kernel's qualified name and the
``parameters`` you pass enter every column's identity. Two different kernels, or
one kernel with two settings, therefore never share a column. Pass a named
function rather than a lambda; a ``functools.partial`` or callable object is
refused because it has no stable name. Put every setting that changes a value in
``parameters``. Settings that only change how a computation runs must stay out;
otherwise one feature splits into a column per setting.

Measures in recipes
-------------------

A package can offer its measures to recipes, and so to ``eegtable run`` and
:func:`~eegtable.extract`. It registers them under the ``eegtable.measures``
entry-point group:

.. code-block:: toml

   # pyproject.toml of your package
   [project.entry-points."eegtable.measures"]
   largest_deflection = "my_package.measures:largest_deflection"

.. code-block:: python

   # my_package/measures.py
   import numpy as np
   import eegtable as ef

   def _largest(trace, times):
       return np.nanmax(np.abs(trace), axis=-1)

   def largest_deflection(series, *, windows, groups=None, include_global=True):
       return ef.signal_measure(
           series,
           _largest,
           measure="largest_deflection",
           unit="V",
           windows=windows,
           groups=groups,
           include_global=include_global,
       )

The function follows the built-in measures' conventions:

- Its first parameter names its input: ``spectra``, ``series``, ``signals``,
  ``signal``, ``phase_signal`` or ``segmentation``.
- Its other parameters are keyword-only. The runner supplies ``bands``,
  ``windows``, ``groups``, ``include_global`` and ``baseline`` from the recipe.
- Any other annotated keyword parameter becomes a recipe setting.

Once the package is installed, a recipe names the measure like a built-in one:

.. code-block:: toml

   [[features]]
   measure = "largest_deflection"
   windows = ["stimulus"]

A registered measure is looked up only when a recipe names it, so a broken
plugin affects only the recipes that use it.

A measure contributed to EEGTable itself follows the same conventions;
`CONTRIBUTING <https://github.com/JoshuaDuq/EEGTable/blob/main/CONTRIBUTING.md>`__
lists the remaining steps: exporting, documenting and validating it.
