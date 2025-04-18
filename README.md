![CatSmoothing](https://raw.githubusercontent.com/cheginit/catsmoothing/refs/heads/main/docs/assets/logo_small.png)

# CatSmoothing: Smoothing Shapely Geometries with Catmull-Rom Splines

[![PyPI](https://img.shields.io/pypi/v/catsmoothing)](https://pypi.org/project/catsmoothing/)
[![Conda](https://img.shields.io/conda/vn/conda-forge/catsmoothing)](https://anaconda.org/conda-forge/catsmoothing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cheginit/catsmoothing/HEAD?labpath=docs%2Fexamples)
[![Downloads](https://img.shields.io/conda/dn/conda-forge/catsmoothing.svg)](https://anaconda.org/conda-forge/catsmoothing)

[![codecov](https://codecov.io/gh/cheginit/catsmoothing/graph/badge.svg?token=U2638J9WKM)](https://codecov.io/gh/cheginit/catsmoothing)
[![CI](https://github.com/cheginit/catsmoothing/actions/workflows/test.yml/badge.svg)](https://github.com/cheginit/catsmoothing/actions/workflows/test.yml)
[![Documentation Status](https://readthedocs.org/projects/catsmoothing/badge/?version=latest)](https://catsmoothing.readthedocs.io/latest/?badge=latest)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![CodSpeed Badge](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/cheginit/catsmoothing)

## Overview

**CatSmoothing** smooths [Shapely](https://shapely.readthedocs.io)
geometries, `LineString` and `(Multi)Polygon`, using the Catmull-Rom spline algorithm and can compute tangent angles at each vertex of a list of lines.
The implementation is based on the
[Splines](https://github.com/AudioSceneDescriptionFormat/splines)
library, but offers performance improvements and additional features.

You can try CatSmoothing directly in your browser by clicking the Binder badge above.

## Key Features

- Written in Rust for performance and runs in parallel using Rayon
- Creating splines from 2D/3D vertices of a line that allows computing n-th derivatives
- Smoothing geometries with Centripetal Catmull-Rom splines with **uniform spacing**
    (spacing is determined iteratively based on the arc length of the input geometry)
- Computing tangent vectors at each vertex of a line
- Optional Gaussian filtering to reduce noise in input geometries before smoothing

## Installation

Install CatSmoothing via `pip` or `micromamba`:

Using `pip`:

```bash
pip install catsmoothing
```

Using `micromamba` (or `conda`/`mamba`):

```bash
micromamba install -c conda-forge catsmoothing
```

## Quick Start

CatSmoothing provide one class called `CatmullRom` that is general purpose,
Catmull-Rom spline interpolation class. You can tweak the `alpha` parameter of
the class to interpolate with different versions of the Catmull-Rom spline
from 2D/3D vertices of a line and compute n-th derivatives.
For smoothing geometries, CatSmoothing uses the centripetal Catmull-Rom spline
algorithm, i.e., `alpha=0.5`. There are two functions that can be used
for smoothing geometries: `smooth_linestrings` and `smooth_polygon`. There is also
a function for computing tangent angles (in radians) at each vertex of a line.

### Basic Usage

For fitting a Catmull-Rom spline to a line, we can use the following code:

```python
from catsmoothing import CatmullRom


verts = [(0, 0), (0, 0.5), (1.5, 1.5), (1.6, 1.5), (3, 0.2), (3, 0)]
n_pts = 15
smoothed = {}
for alpha in (0, 0.5, 1):
    s = CatmullRom(verts, alpha=alpha, bc_type="closed")
    dots = int((s.grid[-1] - s.grid[0]) * n_pts) + 1
    distances = s.grid[0] + np.arange(dots) / n_pts
    smoothed[alpha] = s.evaluate(distances)
```

![Catmull-Rom Splines](https://raw.githubusercontent.com/cheginit/catsmoothing/main/docs/examples/images/alpha.png)

For smoothing a geometry, we can use the following code:

```python
from shapely import Polygon
import catsmoothing as cs


poly = Polygon(verts)
ploy_smoothed = cs.smooth_polygon(poly, n_pts=50)
```

![Polygon Smoothing](https://raw.githubusercontent.com/cheginit/catsmoothing/main/docs/examples/images/poly.png)

For smoothing a noisy line, we can use the following code:

```python
import numpy as np
from shapely import LineString
import catsmoothing as cs


rng = np.random.default_rng(123)
x = np.linspace(-3, 2.5, 50)
y = np.exp(-(x**2)) + 0.1 * rng.standard_normal(50)
line = LineString(np.c_[x, y])
line_smoothed = cs.smooth_linestrings(line, n_pts=30, gaussian_sigmas=2)
```

![Line Smoothing](https://raw.githubusercontent.com/cheginit/catsmoothing/main/docs/examples/images/line.png)

We can then compute the tangent angles in radians at each vertex of the smoothed line:

```python
tangents = cs.linestrings_tangent_angles(line_smoothed)
```

![Tangent Angles](https://raw.githubusercontent.com/cheginit/catsmoothing/main/docs/examples/images/tangents.png)

For more examples, visit the [documentation](https://catsmoothing.readthedocs.io).

## Contributing

We welcome contributions! For guidelines, please refer to the [CONTRIBUTING.md](https://catsmoothing.readthedocs.io/latest/CONTRIBUTING) and [CODE_OF_CONDUCT.md](https://github.com/cheginit/catsmoothing/blob/main/CODE_OF_CONDUCT.md).

## License

CatSmoothing is licensed under the MIT License. See the [LICENSE](https://github.com/cheginit/catsmoothing/blob/main/LICENSE) file for details.
