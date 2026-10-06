Installation
============

.. raw:: html

   <p class="hero-lede">
     Python 3.11 or newer. Install from source. Core dependencies are
     <code>numpy</code>, <code>scipy</code>, <code>pandas</code>, and <code>mne</code>.
   </p>

Requirements
------------

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: nav-cards

   .. grid-item-card:: Python ≥ 3.11
      :link: https://www.python.org/downloads/
      :link-type: url

      Python 3.11 or newer.

   .. grid-item-card:: Core Scientific Stack
      :link: https://mne.tools/stable/
      :link-type: url

      ``numpy>=1.26``, ``scipy>=1.11``, ``pandas>=2.0``, ``mne>=1.10``.

Setup
-----

Choose your operating system. Run these commands in a terminal to create
an environment and install EEGTable from source.

.. tab-set::
   :sync-group: operating-system

   .. tab-item:: macOS / Linux
      :sync: unix
      :name: macos-linux

      .. code-block:: bash

         git clone https://github.com/JoshuaDuq/EEGTable.git
         cd EEGTable
         python3 -m venv .venv
         source .venv/bin/activate
         python -m pip install --upgrade pip
         python -m pip install -e .

   .. tab-item:: Windows PowerShell
      :sync: windows
      :name: windows-powershell

      .. code-block:: powershell

         git clone https://github.com/JoshuaDuq/EEGTable.git
         cd EEGTable
         py -3.11 -m venv .venv
         .\.venv\Scripts\Activate.ps1
         python -m pip install --upgrade pip
         pip install -e .

Optional dependencies
---------------------

Install only the extras used by your analysis. Extras add dependencies; they do
not alter the core feature definitions. Combine related extras, for example
``python -m pip install -e ".[preprocessing,preprocessing-auto,model]"``.

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Extra
     - Packages
     - Used For
   * - ``[connectivity]``
     - ``mne-connectivity>=0.7``
     - Epoch-averaged and per-epoch time-frequency spectral connectivity,
       including :func:`~eegtable.wpli`.
   * - ``[microstates]``
     - ``scikit-learn>=1.3``
     - GFP peak clustering and microstate segmentation (``eegtable.microstates``).
   * - ``[model]``
     - ``scikit-learn>=1.3``, ``PyYAML>=6.0``, ``filelock>=3.0``
     - Design matrices, grouped cross-fitting, metrics, nulls, uncertainty, and
       model selection (``eegtable.model``, :doc:`/api/model`).
   * - ``[importance]``
     - ``scikit-learn>=1.3``, ``shap>=0.45``
     - SHAP explanations (``eegtable.model.importance``). Permutation importance
       is in the ``model`` extra.
   * - ``[preprocessing]``
     - ``mne>=1.13.2``, ``PyYAML``, ``scikit-learn``, ``h5io``, ``h5py``, ``filelock``
     - Raw-to-epochs preprocessing workflow (:doc:`/guides/preprocessing`).
   * - ``[preprocessing-auto]``
     - ``pyprep``, ``autoreject``, ``mne-icalabel``, ``python-picard``, ``onnxruntime``
     - PyPREP candidates, ICLabel, the Picard ICA solver, and autoreject.
   * - ``[bids]``
     - ``mne-bids>=0.19``, ``pybv``
     - Native EEG BIDS discovery, loading and preprocessing.
   * - ``[spectral-model]``
     - ``specparam>=2.0.0rc7,<2.1``
     - Full fixed/knee spectral parameterization and band peaks.
   * - ``[irasa]``
     - ``neurodsp>=2.3.0``
     - IRASA aperiodic and oscillatory separation.
   * - ``[cycles]``
     - ``bycycle>=1.2.0``, ``pandas<3.0``
     - Cycle waveform and burst features. ByCycle 1.2.0 fails under pandas 3,
       so this extra keeps pandas 2.
   * - ``[complexity]``
     - ``antropy>=0.2.2``
     - Permutation entropy, Lempel–Ziv complexity and DFA.
   * - ``[pac]``
     - ``tensorpac>=0.6.5``
     - Phase-amplitude coupling surrogate inference.
   * - ``[riemann]``
     - ``pyriemann>=0.11``, ``scikit-learn``
     - Training-fitted covariance and tangent-space decoding.
   * - ``[preprocessing-gui]``
     - ``mne-qt-browser``, ``PyQt6``
     - MNE viewers for ``preprocess review`` and ``preprocess inspect``.
   * - ``[dev]``
     - ``pytest``, ``ruff``, ``black``, ``mypy``, type stubs
     - Tests, type checking, and linting.
   * - ``[docs]``
     - ``sphinx``, ``furo``, ``myst-parser``, ``sphinx-copybutton``,
       ``sphinx-design``, ``sphinx-notfound-page``, ``scikit-learn``
     - This documentation, including the modeling API pages.

For development with the scientific integrations and documentation:

.. code-block:: bash

   python -m pip install -e ".[dev,docs,model,connectivity,microstates,importance,preprocessing,preprocessing-auto,bids,spectral-model,irasa,cycles,complexity,pac,riemann]"

The dependency bounds above follow ``pyproject.toml``. The ``cycles`` extra
keeps pandas below 3, and ``spectral-model`` accepts specparam's 2.0 releases
from ``2.0.0rc7``. Each output's provenance records the versions it was computed
with; reinstall those to reproduce it exactly. ``preprocessing-auto`` and
``preprocessing-gui`` supplement the ``preprocessing`` extra and should be
installed with it.

Modeling needs the ``model`` extra. SHAP needs the ``importance`` extra.
Permutation importance is included in ``model``.

The preprocessing terminal front end is a Go program, not a Python extra.
Build it with ``cd tui && go build -o eegtable-tui .`` (Go 1.24 or newer).
Keys are in :doc:`/guides/preprocessing`.

Verify the installation
-----------------------

.. code-block:: bash

   python -c "import eegtable; print(eegtable.__version__)"
   eegtable --version

For a development checkout, install the ``dev`` and ``model`` extras before
running ``python -m pytest``. Additional integration tests require the
corresponding extras. Public-dataset checks are opt-in; see
:doc:`/guides/validation`.

Build the documentation
-----------------------

From the repository root:

.. code-block:: bash

   python -m pip install -e ".[docs]"
   python -m sphinx -b html -W --keep-going docs docs/_build/html

Open ``docs/_build/html/index.html`` to inspect the generated site. The build
imports the current source for API signatures and docstrings. Internet access
is needed to fetch external reference inventories; the CI documentation job
also installs optional scientific integrations. A documentation build does not
rerun dataset validation or refresh its saved evidence.
