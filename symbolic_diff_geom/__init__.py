"""symbolic_diff_geom/ # root package
├── __init__.py # imports and version info
├── charts.py # Chart and SymbolicManifold classes
├── metrics.py # MetricTensor + connection
├── vector_fields.py # VectorField, Lie bracket, flows
├── connections.py # LeviCivitaConnection, other connections
├── visualization.py # all plotting & interactive demos
├── latex_exporter.py # LaTeX export utilities
└── examples/
    ├── __init__.py
    └── sphere.py # S² example combining vector field & flow on manifold"""

# symbolic_diff_geom/__init__.py
"""
symbolic_diff_geom: A Python package for symbolic differential geometry.

Modules:
  charts            - Chart and SymbolicManifold classes for manifold definitions
  Riemannian_metric - RiemannianMetric for metrics, connections, curvature
  vector_fields     - VectorField, ParallelTransport, GeodesicDeviation, numerics
  connections       - Connection base, Levi-Civita, custom examples
  visualization     - Chart/Manifold/Field plotting utilities
  latex_exporter    - Utilities to export symbolic results to LaTeX

Usage:
  from symbolic_diff_geom import Chart, RiemannianMetric, VectorField, LeviCivitaConnection
"""
__version__ = "0.1.0"

# core imports
from .charts import Chart, SymbolicManifold, Domain, Embedding
from .Riemannian_metric import RiemannianMetric
from .vector_fields import VectorField, ParallelTransport, GeodesicDeviation, VectorFieldNumerics
from .connections import LeviCivitaConnection, MetricConnection, CustomConnection
from .visualization import ChartPlotter, ManifoldPlotter, VectorFieldPlotter, FlowPlotter, ConnectionPlotter
from .latex_exporter import LaTeXExporter

# package-level shortcuts
__all__ = [
    "Chart", "SymbolicManifold", "Domain", "Embedding",
    "RiemannianMetric",
    "VectorField", "ParallelTransport", "GeodesicDeviation", "VectorFieldNumerics",
    "LeviCivitaConnection", "MetricConnection", "CustomConnection",
    "ChartPlotter", "ManifoldPlotter", "VectorFieldPlotter", "FlowPlotter", "ConnectionPlotter",
    "LaTeXExporter",
]
