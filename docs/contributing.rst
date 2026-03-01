Contributing
============

Pull requests are welcome. This project follows standard GitHub flow.

Getting started
---------------

1. Fork the repository on GitHub
2. Clone your fork and create a feature branch::

    git clone https://github.com/your-username/visual_behavior_analysis.git
    cd visual_behavior_analysis
    git checkout -b my-feature

3. Install in editable mode with development dependencies::

    pip install --upgrade pip setuptools
    pip install -e ".[DEV]"

4. Make your changes, then run the test suite before opening a PR (see below)

Running tests
-------------

External users (no LIMS access)::

    pytest -m "not onprem"

Allen Institute internal users::

    pytest

Linting
-------

All code must pass flake8 before merging::

    flake8 visual_behavior

CircleCI runs both linting and tests automatically on every pull request.

Tagging reviewers
-----------------

Please tag ``@dougollerenshaw`` and ``@matchings`` on pull requests for review.

Contributors
------------

- Nicholas Cain - `@nicain <https://github.com/nicain>`_
- Marina Garrett - marinag@alleninstitute.org, `@matchings <https://github.com/matchings>`_
- Nile Graddis - nileg@alleninstitute.org, `@nilegraddis <https://github.com/nilegraddis>`_
- Justin Kiggins - `@neuromusic <https://github.com/neuromusic>`_
- Jerome Lecoq - jeromel@alleninstitute.org, `@jeromelecoq <https://github.com/jeromelecoq>`_
- Sahar Manavi - saharm@alleninstitute.org, `@saharmanavi <https://github.com/saharmanavi>`_
- Nicholas Mei - nicholas.mei@alleninstitute.org, `@njmei <https://github.com/njmei>`_
- Christopher Mochizuki - chrism@alleninstitute.org, `@mochic <https://github.com/mochic>`_
- Doug Ollerenshaw - dougo@alleninstitute.org, `@dougollerenshaw <https://github.com/dougollerenshaw>`_
- Natalia Orlova - nataliao@alleninstitute.org, `@nataliaorlova <https://github.com/nataliaorlova>`_
- Jed Perkins - `@jfperkins <https://github.com/jfperkins>`_
- Alex Piet - alex.piet@alleninstitute.org, `@alexpiet <https://github.com/alexpiet>`_
- Nick Ponvert - `@nickponvert <https://github.com/nickponvert>`_
- Kate Roll - kater@alleninstitute.org, `@downtoncrabby <https://github.com/downtoncrabby>`_
- Ryan Valenza - `@ryval <https://github.com/ryval>`_
- Farzaneh Najafi - farzaneh.najafi@alleninstitute.org
- Iryna Yavorska - iryna.yavorska@alleninstitute.org
