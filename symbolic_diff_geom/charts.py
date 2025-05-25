from __future__ import annotations
import sympy as sp
from sympy import Expr, Symbol, sympify, lambdify
from typing import Callable, Dict, List, Tuple, Optional, Union
import numpy as np

# -------------------------- Domain Definitions --------------------------
class Domain:
    """
    Abstract base class for chart domains in R^n.
    """
    def contains(self, point: Tuple[float, ...]) -> bool:
        """
        Determine if a numeric point lies in this domain.
        """
        raise NotImplementedError("Domain.contains must be implemented by subclasses.")

class PredicateDomain(Domain):
    """
    Domain defined by a Python predicate f: R^n -> bool.
    """
    def __init__(self, predicate: Callable[..., bool]):
        self.predicate = predicate
    def contains(self, point: Tuple[float, ...]) -> bool:
        return bool(self.predicate(*point))

class BoxDomain(Domain):
    """
    Axis-aligned box: each coordinate x_i in [lo_i, hi_i].
    """
    def __init__(self, bounds: List[Tuple[float, float]]):
        # bounds: list of (min, max) pairs for each coordinate
        self.bounds = bounds
    def contains(self, point: Tuple[float, ...]) -> bool:
        return all(lo <= x <= hi for x, (lo, hi) in zip(point, self.bounds))

class InequalityDomain(Domain):
    """
    Domain defined by a Sympy relational expression, e.g. x**2 + y**2 < 1.
    """
    def __init__(self, expr: Union[Expr, str], coords: List[Symbol]):
        self.expr = sympify(expr)
        self.coords = coords
        # lambdify to a numeric function
        self._func = lambdify(coords, self.expr, "numpy")
    def contains(self, point: Tuple[float, ...]) -> bool:
        return bool(self._func(*point))

class UnionDomain(Domain):
    """
    Union of multiple domains: point is in any one.
    """
    def __init__(self, domains: List[Domain]):
        self.domains = domains
    def contains(self, point: Tuple[float, ...]) -> bool:
        return any(domain.contains(point) for domain in self.domains)

# ------------------------ Embedding Definitions ------------------------
class Embedding:
    """
        Base class for embedding chart coordinates into R^m.
        By default behaves like ParametricEmbedding: store coords and map_exprs, compile with lambdify.
        """

    def __init__(self, coords: List[Symbol], map_exprs: List[Expr]):
        # store symbolic definitions
        self.coords = coords
        # sympify all expressions
        self.map_exprs = [sympify(expr) for expr in map_exprs]
        # compile to numeric
        self._func = lambdify(self.coords, self.map_exprs, "numpy")

    def evaluate(self, point: Tuple[float, ...]) -> Tuple[float, ...]:
        """
        Map a chart coordinate tuple to an embedded point in R^m.
        """
        arr = np.array(self._func(*point), dtype=float)
        return tuple(arr.flatten())

        # ------------------------ Embedding Definitions ------------------------(self, point: Tuple[float, ...]) -> Tuple[float, ...]:
        """
        Map a chart coordinate tuple to an embedded point in R^m.
        """
        raise NotImplementedError("Embedding.evaluate must be implemented by subclasses.")


class ParametricEmbedding(Embedding):
    """
    Embedding defined by Sympy expressions map_exprs in terms of coords.
    """
    def __init__(self, coords: List[Symbol], map_exprs: List[Expr]):
        self.coords = coords
        self.map_exprs = [sympify(expr) for expr in map_exprs]
        # Precompile with lambdify for performance
        self._func = lambdify(coords, self.map_exprs, "numpy")
    def evaluate(self, point: Tuple[float, ...]) -> Tuple[float, ...]:
        arr = np.array(self._func(*point), dtype=float)
        return tuple(arr.flatten())

# ---------------------- Coordinate Transition ----------------------
class TransitionMap:
    """
    Represents a transition map between two chart coordinate systems.
    """
    def __init__(
        self,
        source_coords: List[Symbol],
        target_coords: List[Symbol],
        forward_map: Dict[Symbol, Expr],
        inverse_map: Optional[Dict[Symbol, Expr]] = None
    ):
        # Validate keys
        if set(forward_map.keys()) != set(source_coords):
            raise ValueError("Forward map must define images of all source coords.")
        self.source_coords = source_coords
        self.target_coords = target_coords
        # Sympify expressions
        self.forward_map = {sym: sympify(expr) for sym, expr in forward_map.items()}
        self.inverse_map = {sym: sympify(expr) for sym, expr in (inverse_map or {}).items()}
        # Lambdify numeric functions
        self._fwd_func = lambdify(source_coords, list(self.forward_map.values()), "numpy")
        self._inv_func = (
            lambdify(target_coords, list(self.inverse_map.values()), "numpy")
            if inverse_map
            else None
        )

    def to_target(self, point: Tuple[float, ...]) -> Tuple[float, ...]:
        """
        Map a numeric point in source chart to coordinates in target chart.
        """
        arr = np.array(self._fwd_func(*point), dtype=float)
        return tuple(arr.flatten())

    def to_source(self, point: Tuple[float, ...]) -> Tuple[float, ...]:
        """
        Map a numeric point in target chart back to source chart.
        Requires inverse_map to be provided.
        """
        if not self._inv_func:
            raise ValueError("Inverse transition is not defined.")
        arr = np.array(self._inv_func(*point), dtype=float)
        return tuple(arr.flatten())

# -------------------------- Chart Definition --------------------------
class Chart:
    """
    Coordinate chart on a manifold.

    Attributes:
      name: unique identifier
      coords: symbols for local R^n coordinates
      domain: Domain instance specifying valid region
      embedding: optional Embedding into R^m
      transitions: mapping other_chart_name -> TransitionMap
    """
    def __init__(
        self,
        name: str,
        coords: List[Symbol],
        domain: Domain = PredicateDomain(lambda *args: True),
        embedding: Optional[Embedding] = None
    ):
        self.name = name
        self.coords = list(coords)
        self.dim = len(coords)
        self.domain = domain
        self.embedding = embedding
        # Transition maps to other charts in the same manifold
        self.transitions: Dict[str, TransitionMap] = {}

    def add_transition(
        self,
        other: Chart,
        forward: Dict[Symbol, Expr],
        inverse: Optional[Dict[Symbol, Expr]] = None
    ) -> None:
        """
        Define a transition between this chart and another.
        """
        tm = TransitionMap(self.coords, other.coords, forward, inverse)
        self.transitions[other.name] = tm

    def contains(self, point: Tuple[float, ...]) -> bool:
        """
        Check if a numeric point lies within the chart's domain.
        """
        return self.domain.contains(point)

    def to_chart(self, point: Tuple[float, ...], other: Chart) -> Tuple[float, ...]:
        """
        Transform a numeric point from this chart to another chart.
        """
        if other.name not in self.transitions:
            raise KeyError(f"No transition from {self.name} to {other.name}.")
        return self.transitions[other.name].to_target(point)

    def sample_grid(
        self,
        bounds: Optional[List[Tuple[float, float]]] = None,
        num: int = 20
    ) -> List[Tuple[float, ...]]:
        """
        Generate a grid of points within the chart domain or given bounds.
        """
        rng = bounds or getattr(self.domain, 'bounds', None)
        if not rng:
            raise ValueError("Bounds must be provided or domain must be BoxDomain.")
        # Create grid axes
        axes = [np.linspace(lo, hi, num) for lo, hi in rng]
        mesh = np.meshgrid(*axes)
        # Flatten mesh to list of tuples
        pts = [tuple(mesh[d].flat[i] for d in range(self.dim)) for i in range(mesh[0].size)]
        # Filter by domain
        return [pt for pt in pts if self.contains(pt)]

# End of Part 1: core definitions up to Chart.sample_grid

# ----------------------- Manifold Definition -----------------------
class SymbolicManifold:
    """
    Abstract n-dimensional manifold represented by an atlas of charts.
    """
    def __init__(self, name: str):
        self.name = name
        self.charts: Dict[str, Chart] = {}
        self.default: Optional[Chart] = None

    def add_chart(self, chart: Chart, default: bool = False) -> None:
        """
        Add a new Chart to the manifold.
        If default=True or no default set, mark this as default chart.
        """
        if chart.name in self.charts:
            raise KeyError(f"Chart '{chart.name}' already exists in manifold '{self.name}'.")
        self.charts[chart.name] = chart
        if default or self.default is None:
            self.default = chart

    def get_chart(self, name: str) -> Chart:
        """
        Get a chart by name.
        """
        if name not in self.charts:
            raise KeyError(f"Chart '{name}' not found in manifold '{self.name}'.")
        return self.charts[name]

    def find_chart(self, point: Tuple[float, ...]) -> Chart:
        """
        Find a chart whose domain contains the numeric point.
        Returns default chart if multiple match or fallback.
        """
        for chart in self.charts.values():
            try:
                if chart.contains(point):
                    return chart
            except Exception:
                continue
        if self.default and self.default.contains(point):
            return self.default
        raise ValueError(f"No chart contains point {point} in manifold '{self.name}'.")

    def transition(
        self,
        point: Tuple[float, ...],
        source: Optional[str] = None,
        target: Optional[str] = None
    ) -> Tuple[float, ...]:
        """
        Transition a numeric point from one chart to another.
        If source omitted, auto-detect via find_chart.
        If target omitted, use default chart.
        """
        src_chart = self.charts[source] if source else self.find_chart(point)
        tgt_chart = self.charts[target] if target else self.default
        if not src_chart or not tgt_chart:
            raise ValueError("Source or target chart undefined.")
        return src_chart.to_chart(point, tgt_chart)

# -------------------- Future Extension Hooks --------------------
class DomainParser:
    """
    Parse domain specification strings or objects into Domain instances.
    """
    @staticmethod
    def from_string(spec: str, coords: List[Symbol]) -> Domain:
        """
        Convert string like 'x**2+y**2<1' to an InequalityDomain.
        TODO: handle logical AND/OR, multiple relations.
        """
        expr = sympify(spec)
        if isinstance(expr, sp.Relational):
            return InequalityDomain(expr, coords)
        if isinstance(expr, sp.And) or isinstance(expr, sp.Or):
            # Decompose into subdomains
            sub = [InequalityDomain(e, coords) for e in expr.args]
            return UnionDomain(sub)
        raise ValueError(f"Cannot parse domain spec: {spec}")

class ChartValidator:
    """
    Validate chart and transition consistency.
    """
    @staticmethod
    def validate_transition(
        chart_a: Chart,
        chart_b: Chart,
        tol: float = 1e-6,
        samples: int = 5
    ) -> bool:
        """
        Numerically test forward+inverse mapping consistency.
        Samples points in overlapping domain.
        """
        tm = chart_a.transitions.get(chart_b.name)
        if not tm or not tm._inv_func:
            return False
        # sample grid in intersection of box domains if available
        bd_a = getattr(chart_a.domain, 'bounds', None)
        bd_b = getattr(chart_b.domain, 'bounds', None)
        if not bd_a or not bd_b:
            return False  # cannot sample
        samples_pts = []
        for i in range(samples):
            pt = tuple(
                np.random.uniform(max(bd_a[d][0], bd_b[d][0]), min(bd_a[d][1], bd_b[d][1]))
                for d in range(chart_a.dim)
            )
            samples_pts.append(pt)
        for pt in samples_pts:
            try:
                tgt = tm.to_target(pt)
                src = tm.to_source(tgt)
            except Exception:
                return False
            if any(abs(x-y) > tol for x,y in zip(pt,src)):
                return False
        return True

# End of Part 2: SymbolicManifold, DomainParser, ChartValidator

# ---------------------- Visualization Utilities ----------------------
class AtlasVisualizer:
    """
    Utilities for plotting chart domains and embedded manifolds.
    """
    @staticmethod
    def plot_domains(
        atlas: SymbolicManifold,
        chart_names: Optional[List[str]] = None,
        resolution: int = 100
    ) -> None:
        """
        Plot abstract chart domains in parameter space using Matplotlib.
        Supports only BoxDomain or InequalityDomain for now.
        """
        import matplotlib.pyplot as plt
        names = chart_names or list(atlas.charts.keys())
        plt.figure()
        for name in names:
            chart = atlas.get_chart(name)
            domain = chart.domain
            if isinstance(domain, BoxDomain):
                xs = np.linspace(domain.bounds[0][0], domain.bounds[0][1], resolution)
                ys = np.linspace(domain.bounds[1][0], domain.bounds[1][1], resolution) if chart.dim > 1 else [0]
                X, Y = np.meshgrid(xs, ys)
                mask = np.array([[domain.contains((x, y)) for x in xs] for y in ys])
                plt.contourf(X, Y, mask, alpha=0.3, label=name)
            elif isinstance(domain, InequalityDomain):
                xs = np.linspace(-1, 1, resolution)
                ys = np.linspace(-1, 1, resolution)
                X, Y = np.meshgrid(xs, ys)
                pts = np.vstack([X.flatten(), Y.flatten()]).T
                mask = np.array([domain.contains(tuple(pt)) for pt in pts]).reshape(X.shape)
                plt.contourf(X, Y, mask, alpha=0.3, label=name)
            else:
                continue
        plt.title(f"Chart Domains: {atlas.name}")
        plt.xlabel(str(atlas.default.coords[0]))
        if atlas.default.dim > 1:
            plt.ylabel(str(atlas.default.coords[1]))
        plt.legend()
        plt.show()

    @staticmethod
    def plot_embedding(
        chart: Chart,
        points: List[Tuple[float, ...]],
        vectors: Optional[List[Tuple[float, ...]]] = None
    ) -> None:
        """
        Plot a set of points (and optional tangent vectors) on the embedded manifold.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        if chart.embedding is None:
            raise ValueError("Chart has no embedding for visualization.")
        emb = chart.embedding
        # Evaluate points
        pts_emb = np.array([emb.evaluate(pt) for pt in points])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d' if pts_emb.shape[1] == 3 else None)
        if pts_emb.shape[1] == 3:
            ax.scatter(pts_emb[:, 0], pts_emb[:, 1], pts_emb[:, 2], s=20)
        else:
            plt.scatter(pts_emb[:, 0], pts_emb[:, 1], s=20)
        # Plot vectors if provided
        if vectors:
            jac = sp.Matrix(emb.map_exprs).jacobian(chart.coords)
            jac_func = lambdify(chart.coords, jac, 'numpy')
            for pt, vec in zip(points, vectors):
                Jp = np.array(jac_func(*pt), dtype=float)
                vec_emb = Jp.dot(np.array(vec, dtype=float))
                if pts_emb.shape[1] == 3:
                    ax.quiver(
                        *emb.evaluate(pt), *vec_emb, length=0.2, color='r'
                    )
                else:
                    plt.quiver(
                        pts_emb[:, 0], pts_emb[:, 1], vec_emb[0], vec_emb[1]
                    )
        plt.title(f"Embedding of Chart {chart.name}")
        plt.show()



