# CatSmoothing: Python Wrapper for WhiteboxTools

[![PyPI](https://img.shields.io/pypi/v/catsmoothing)](https://pypi.org/project/catsmoothing/)
[![Conda](https://img.shields.io/conda/vn/conda-forge/catsmoothing)](https://anaconda.org/conda-forge/catsmoothing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cheginit/catsmoothing/HEAD?labpath=docs%2Fexamples)

[![codecov](https://codecov.io/gh/cheginit/catsmoothing/graph/badge.svg?token=U2638J9WKM)](https://codecov.io/gh/cheginit/catsmoothing)
[![CI](https://github.com/cheginit/catsmoothing/actions/workflows/test.yml/badge.svg)](https://github.com/cheginit/catsmoothing/actions/workflows/test.yml)
[![Documentation Status](https://readthedocs.org/projects/catsmoothing/badge/?version=latest)](https://catsmoothing.readthedocs.io/latest/?badge=latest)

## Overview

**CatSmoothing**

You can try CatSmoothing directly in your browser by clicking the Binder badge above.

## Key Features



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



### Basic Usage



Example usage:

```python
import catsmoothing
from pathlib import Path

src_dir = Path("path/to/input_files/")
wbt_args = {
    "BreachDepressions": ["-i=dem.tif", "--fill_pits", "-o=dem_corr.tif"],
    # Additional tools...
}

catsmoothing.whitebox_tools(src_dir, wbt_args)
```

![Strahler Stream Order](https://raw.githubusercontent.com/cheginit/catsmoothing/main/docs/examples/images/stream_order.png)

For more examples, visit the [documentation](https://catsmoothing.readthedocs.io).

## Contributing

We welcome contributions! For guidelines, please refer to the [CONTRIBUTING.md](https://catsmoothing.readthedocs.io/latest/CONTRIBUTING) and [CODE_OF_CONDUCT.md](https://github.com/cheginit/catsmoothing/blob/main/CODE_OF_CONDUCT.md).

## License

CatSmoothing is licensed under the MIT License. See the [LICENSE](https://github.com/cheginit/catsmoothing/blob/main/LICENSE) file for details.
