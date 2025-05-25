import sympy as sp
from sympy import Expr, Symbol, Function
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np

from charts import Chart
from connections import LeviCivitaConnection
from Riemannian_metric import RiemannianMetric
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from visualization import Visualizer


# ------------------------ VectorField Definition ------------------------
class VectorField:
    """
    Represents a smooth vector field on a coordinate chart.

    Attributes:
        chart: Chart defining the coordinate domain
        components: symbolic expressions V^i(x)
    """
    def __init__(self, chart: Chart, components: List[Expr]):
        self.chart = chart
        self.coords = chart.coords
        if len(components) != len(self.coords):
            raise ValueError(f"VectorField requires {len(self.coords)} components, got {len(components)}.")
        # simplify components
        self.components = [sp.simplify(c) for c in components]
        # cache numeric lambdify
        self._num_func = sp.lambdify(self.coords, self.components, 'numpy')

    def __repr__(self) -> str:
        return f"<VectorField on '{self.chart.name}' with components {self.components}>"

    def evaluate(self, point: Tuple[float, ...]) -> Tuple[float, ...]:
        """
        Numerically evaluate vector field at a given chart point.
        """
        vals = self._num_func(*point)
        return tuple(float(v) for v in np.array(vals).flatten())

    def as_dict(self) -> Dict[Symbol, Expr]:
        """
        Return mapping from coordinate symbol to its component expression.
        """
        return {self.coords[i]: self.components[i] for i in range(len(self.coords))}

    def lie_bracket(self, other: 'VectorField') -> 'VectorField':
        """
        Compute the Lie bracket [self, other] on the same chart.
        [V,W]^i = V^j ∂_j W^i - W^j ∂_j V^i
        """
        if self.chart != other.chart:
            raise ValueError("Both fields must lie on the same chart.")
        bracket = []
        for i in range(len(self.coords)):
            expr = 0
            for j in range(len(self.coords)):
                expr += self.components[j] * sp.diff(other.components[i], self.coords[j])
                expr -= other.components[j] * sp.diff(self.components[i], self.coords[j])
            bracket.append(sp.simplify(expr))
        return VectorField(self.chart, bracket)

    def flow_equations(self) -> Tuple[List[sp.Eq], List[Function]]:
        """
        Symbolic ODEs dx^i/dt = V^i(x) for the integral curves.
        Returns equations and symbolic functions x^i(t).
        """
        t = sp.symbols('t')
        funcs = [Function(str(c))(t) for c in self.coords]
        eqs = []
        subs = {self.coords[i]: funcs[i] for i in range(len(self.coords))}
        for i, fi in enumerate(funcs):
            ode = sp.Eq(sp.diff(fi, t), self.components[i].subs(subs))
            eqs.append(ode)
        return eqs, funcs

    def pullback(self, mapping: Dict[Symbol, Expr]) -> 'VectorField':
        """
        Pushforward of vector under coordinate transformation old->new: V'^i = ∂y^i/∂x^j V^j.
        """
        new_coords = list(mapping.keys())
        # Jacobian J^i_j = ∂y^i/∂x^j
        J = sp.Matrix([[sp.diff(mapping[new], old) for old in self.coords] for new in new_coords])
        # substitute old coords in components
        subs_map = {old: mapping[old] for old in self.coords}
        new_comps = []
        for i in range(len(new_coords)):
            expr = sum(J[i, j] * self.components[j].subs(subs_map) for j in range(len(self.coords)))
            new_comps.append(sp.simplify(expr))
        # new chart must be created by user
        return VectorField(self.chart, new_comps)

# -------------------- Discrete Vector Field Stub --------------------
class DiscreteVectorField:
    """
    Placeholder for vector fields defined on discrete meshes/graphs.

    Attributes:
        mesh: user-defined mesh object (vertices, connectivity)
        values: vector values at mesh vertices
    """
    def __init__(self, mesh: Any, values: Dict[Any, Tuple[float, ...]]):
        self.mesh = mesh
        self.values = values

    def sample(self, vertex: Any) -> Tuple[float, ...]:
        return self.values.get(vertex, None)

    def to_continuous(self) -> VectorField:
        """
        Interpolate discrete field to a continuous vector field (stub).
        """
        raise NotImplementedError("Discrete->continuous interpolation not implemented.")

# ------------------------ Parallel Transport ------------------------
class ParallelTransport:
    """
    Parallel transport along a curve under a given connection.
    """
    def __init__(self, connection: LeviCivitaConnection, curve_funcs: List[Function]):
        self.connection = connection
        self.curve = curve_funcs

    def equations(self, vector_syms: List[Function]) -> List[sp.Eq]:
        t = sp.symbols('t')
        Gamma = self.connection.Gamma
        eqs = []
        for i in range(len(vector_syms)):
            term = sp.diff(vector_syms[i], t)
            for j in range(len(self.curve)):
                for k in range(len(self.curve)):
                    term += Gamma[i][j][k] * sp.diff(self.curve[j], t) * vector_syms[k]
            eqs.append(sp.Eq(term, 0))
        return eqs

# -------------------- Geodesic Deviation --------------------
class GeodesicDeviation:
    """
    Jacobi equation: D^2J^i/ds^2 + R^i_{jkl} u^j J^k u^l = 0.
    """
    def __init__(self, metric: RiemannianMetric, u: List[Function], J: List[Function]):
        self.metric = metric
        self.u = u
        self.J = J

    def equations(self) -> List[sp.Eq]:
        t = sp.symbols('t')
        R = self.metric.riemann_tensor()
        eqs = []
        for i in range(len(self.u)):
            second = sp.diff(self.J[i], (t, 2))
            term = sum(
                R[i][j][k][l] * self.u[j] * self.J[k] * self.u[l]
                for j in range(len(self.u)) for k in range(len(self.u)) for l in range(len(self.u))
            )
            eqs.append(sp.Eq(second + term, 0))
        return eqs

# -------------------- Numeric Integration --------------------
class VectorFieldNumerics:
    """
    Numeric utilities for vector field integration.
    """
    @staticmethod
    def integrate_flow(
        vf: VectorField,
        initial_point: Tuple[float, ...],
        t_span: Tuple[float, float],
        num: int = 200
    ) -> Tuple[np.ndarray, np.ndarray]:
        try:
            from scipy.integrate import odeint
        except ImportError:
            raise ImportError("scipy is required for numeric integration.")
        def ode(x, t):
            return vf.evaluate(tuple(x))
        ts = np.linspace(t_span[0], t_span[1], num)
        sol = odeint(ode, np.array(initial_point, float), ts)
        return ts, sol

# -------------------- Utility Functions --------------------
def validate_vector_field(vf: VectorField) -> bool:
    """
    Ensure vector field components depend only on chart coordinates.
    """
    syms = set().union(*(comp.free_symbols for comp in vf.components))
    return syms.issubset(set(vf.coords))

# End of vector_fields.py


