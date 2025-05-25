import sympy as sp
from sympy import Expr, Function, symbols, latex
from typing import List, Tuple, Any, Callable, TYPE_CHECKING

from charts import Chart
from Riemannian_metric import RiemannianMetric
if TYPE_CHECKING:
    from vector_fields import VectorField

# ======================= connections.py =======================
# Modular, extensible connection implementations

# ---------------- Part 1: Abstract Base ----------------
class Connection:
    """
    Abstract base for affine connections on a Chart.

    Must implement:
      - covariant_derivative
      - parallel_transport_equations
      - geodesic_equations
    """
    def __init__(self, chart: Chart):
        self.chart = chart
        self.coords = chart.coords

    def covariant_derivative(
        self,
        vec_components: List[Expr],
        direction_index: int
    ) -> List[Expr]:
        """
        ∇_{∂_{direction}} V components.
        """
        raise NotImplementedError

    def parallel_transport_equations(
        self,
        curve_funcs: List[Function],
        vec_funcs: List[Function]
    ) -> List[sp.Eq]:
        """
        ODEs for parallel transport: dV^i/dt + connection^i_{jk} x'^j V^k = 0.
        """
        raise NotImplementedError

    def geodesic_equations(self) -> Tuple[List[sp.Eq], List[Function]]:
        """
        Equations x''^i + connection^i_{jk} x'^j x'^k = 0.
        """
        raise NotImplementedError

# ---------------- Part 2: Levi-Civita Connection ----------------
class LeviCivitaConnection(Connection):
    """
    Metric-compatible, torsion-free connection derived from a RiemannianMetric.
    """
    def __init__(self, metric: RiemannianMetric, chart: Chart):
        super().__init__(chart)
        self.metric = metric
        self._Gamma: List[List[List[Expr]]] = None

    @property
    def Gamma(self) -> List[List[List[Expr]]]:
        """ Christoffel symbols Γ^i_{jk}. """
        if self._Gamma is None:
            self._Gamma = self.metric.christoffel_symbols()
        return self._Gamma

    def covariant_derivative(
        self,
        vec_components: List[Expr],
        direction_index: int
    ) -> List[Expr]:
        n = len(self.coords)
        result = [0]*n
        for i in range(n):
            term = sp.diff(vec_components[i], self.coords[direction_index])
            for j in range(n):
                term += self.Gamma[i][direction_index][j] * vec_components[j]
            result[i] = sp.simplify(term)
        return result

    def parallel_transport_equations(
        self,
        curve_funcs: List[Function],
        vec_funcs: List[Function]
    ) -> List[sp.Eq]:
        t = symbols('t')
        eqs = []
        for i in range(len(vec_funcs)):
            expr = sp.diff(vec_funcs[i], t)
            for j in range(len(curve_funcs)):
                for k in range(len(vec_funcs)):
                    expr += self.Gamma[i][j][k] * sp.diff(curve_funcs[j], t) * vec_funcs[k]
            eqs.append(sp.Eq(expr, 0))
        return eqs

    def geodesic_equations(self) -> Tuple[List[sp.Eq], List[Function]]:
        t = symbols('t')
        funcs = [Function(str(c))(t) for c in self.coords]
        eqs = []
        for i in range(len(funcs)):
            expr = sp.diff(funcs[i], (t,2))
            for j in range(len(funcs)):
                for k in range(len(funcs)):
                    expr += self.Gamma[i][j][k] * sp.diff(funcs[j], t) * sp.diff(funcs[k], t)
            eqs.append(sp.Eq(expr, 0))
        return eqs, funcs

    def to_latex(self) -> str:
        lines = []
        n = len(self.coords)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if self.Gamma[i][j][k] != 0:
                        lines.append(f"\\Gamma^{{{i}}}_{{{j}{k}}} = {latex(self.Gamma[i][j][k])}")
        return "\\\\n".join(lines)

# ---------------- Part 3: Metric Connection Alias ----------------
MetricConnection = LeviCivitaConnection

# ---------------- Part 4: Custom Connection Example ----------------
class CustomConnection(Connection):
    """
    Example: flat connection (all Γ^i_{jk} = 0).
    """
    def __init__(self, chart: Chart):
        super().__init__(chart)

    def covariant_derivative(self, vec_components: List[Expr], direction_index: int) -> List[Expr]:
        return [sp.diff(vec_components[i], self.coords[direction_index]) for i in range(len(self.coords))]

    def parallel_transport_equations(self, curve_funcs: List[Function], vec_funcs: List[Function]) -> List[sp.Eq]:
        t = symbols('t')
        return [sp.Eq(sp.diff(Vi, t), 0) for Vi in vec_funcs]

    def geodesic_equations(self) -> Tuple[List[sp.Eq], List[Function]]:
        t = symbols('t')
        funcs = [Function(str(c))(t) for c in self.coords]
        return [sp.Eq(sp.diff(fi, (t,2)), 0) for fi in funcs], funcs

# ---------------- Part 5: Utility and Validation ----------------
def validate_connection(conn: Connection) -> bool:
    """
    New utilities to test connection properties:
    - metric compatibility: ∇ g = 0
    - torsion-free: Γ^i_{jk} = Γ^i_{kj}
    """
    # Torsion-free check
    coords = conn.coords
    Gamma = getattr(conn, 'Gamma', None)
    if Gamma:
        n = len(coords)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if Gamma[i][j][k] != Gamma[i][k][j]:
                        return False
    # Metric compatibility for Levi-Civita
    if isinstance(conn, LeviCivitaConnection):
        metric = conn.metric.g
        inv = conn.metric.invg
        for l in range(len(coords)):
            for i in range(len(coords)):
                for j in range(len(coords)):
                    deriv = sp.diff(metric[i,j], coords[l])
                    for k in range(len(coords)):
                        deriv -= metric[k,j]*conn.Gamma[k][l][i] + metric[i,k]*conn.Gamma[k][l][j]
                    if sp.simplify(deriv) != 0:
                        return False
    return True

# End of connections.py
