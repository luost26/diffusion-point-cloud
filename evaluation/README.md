# Evaluation

From [https://github.com/stevenygd/PointFlow/tree/master/metrics](https://github.com/stevenygd/PointFlow/tree/master/metrics)

Modifications:

| Position          | Original                       | Modified                            |
| ----------------- | ------------------------------ | ----------------------------------- |
| Makefile:9        | `/usr/local/cuda/bin/nvcc`     | `/usr/local/cuda-10.0/bin/nvcc`     |
| Makefile:69,70    | `c++11`                        | `c++14`                             |
| Makefile:74,75    | `lib.linux-x86_64-3.6`         | `lib.linux-x86_64-3.7`              |
| Pybind/bind.cpp:5 | `#include "pybind/extern.hpp"` | `#include "extern.hpp"`             |
| \__init__.py      |                                | `from .evaluation_metrics import *` |

