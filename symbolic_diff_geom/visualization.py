
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from typing import List, Tuple, Any, Optional, TYPE_CHECKING
from sympy import lambdify, Expr, Symbol, Function

from charts import Chart, SymbolicManifold, Embedding, Domain

# Circular-import-safe VectorField import
if TYPE_CHECKING:
    from vector_fields import VectorField
from vector_fields import VectorField

from connections import Connection, LeviCivitaConnection
from Riemannian_metric import RiemannianMetric

# ======================= visualization.py =======================
# Provides plotting utilities for manifolds, vector fields, flows, and connections

# ---------------- Part 1: Chart Visualization Utilities ----------------
class ChartPlotter:
    """
    Plot chart domains and coordinate grid lines in parameter space.
    """
    @staticmethod
    def plot_chart_domain(
        chart: Chart,
        bounds: Optional[List[Tuple[float, float]]] = None,
        resolution: int = 50
    ) -> None:
        domain = chart.domain
        if hasattr(domain, 'bounds'):
            xs = np.linspace(domain.bounds[0][0], domain.bounds[0][1], resolution)
            ys = np.linspace(domain.bounds[1][0], domain.bounds[1][1], resolution)
            X, Y = np.meshgrid(xs, ys)
            mask = np.array([[domain.contains((x, y)) for x in xs] for y in ys])
            plt.contourf(X, Y, mask, alpha=0.3)
            plt.title(f"Domain of chart {chart.name}")
            plt.xlabel(str(chart.coords[0])); plt.ylabel(str(chart.coords[1])); plt.show()

    @staticmethod
    def plot_chart_grid(
        chart: Chart,
        grid_lines: int = 10
    ) -> None:
        rng = getattr(chart.domain, 'bounds', None)
        if not rng:
            raise ValueError("BoxDomain required for grid plotting.")
        xs = np.linspace(rng[0][0], rng[0][1], grid_lines)
        ys = np.linspace(rng[1][0], rng[1][1], grid_lines)
        for x in xs:
            plt.plot([x]*len(ys), ys, 'k:', alpha=0.5)
        for y in ys:
            plt.plot(xs, [y]*len(xs), 'k:', alpha=0.5)
        plt.title(f"Grid on chart {chart.name}")
        plt.xlabel(str(chart.coords[0])); plt.ylabel(str(chart.coords[1])); plt.show()

# ---------------- Part 2: Manifold Embedding ----------------
class ManifoldPlotter:
    """
    Plot embedded manifold surfaces and overlay charts/domain edges.
    """
    @staticmethod
    def plot_manifold(
        chart: Chart,
        resolution: int = 50,
        color: str = 'lightblue',
        alpha: float = 0.7,
        show_edges: bool = True
    ) -> None:
        if chart.embedding is None:
            raise ValueError("Chart has no embedding defined.")
        dom = chart.domain
        if hasattr(dom, 'bounds'):
            bounds = dom.bounds
        else:
            raise ValueError("Non-box domains require custom sampling.")
        us = np.linspace(bounds[0][0], bounds[0][1], resolution)
        vs = np.linspace(bounds[1][0], bounds[1][1], resolution)
        U, V = np.meshgrid(us, vs)
        points = np.vstack([U.flatten(), V.flatten()]).T
        emb_func = lambdify(chart.coords, chart.embedding.map_exprs, 'numpy')
        XYZ = np.array([emb_func(u, v) for u, v in points])
        X = XYZ[:,0].reshape(U.shape); Y = XYZ[:,1].reshape(U.shape)
        Z = XYZ[:,2].reshape(U.shape) if XYZ.shape[1]==3 else None
        fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
        if Z is not None:
            ax.plot_surface(X, Y, Z, color=color, alpha=alpha)
        else:
            ax.plot_surface(X, Y, np.zeros_like(X), color=color, alpha=alpha)
        if show_edges:
            ax.plot_wireframe(X, Y, Z if Z is not None else np.zeros_like(X), color='k', linewidth=0.3)
        plt.title(f"Embedded manifold: chart {chart.name}"); plt.show()

# ---------------- Part 3: Vector Field on Manifold ----------------
class VectorFieldPlotter:
    """
    Overlay symbolic vector fields on embedded manifold.
    """
    @staticmethod
    def plot_vector_field(
        vf: VectorField,
        chart: Chart,
        density: int = 20,
        scale: float = 0.1,
        color: str = 'red'
    ) -> None:
        if chart.embedding is None:
            raise ValueError("Chart must have an embedding for vector plots.")
        dom = chart.domain
        bounds = getattr(dom, 'bounds', None)
        if bounds is None:
            raise ValueError("Vector field plotting requires BoxDomain bounds.")
        xs = np.linspace(bounds[0][0], bounds[0][1], density)
        ys = np.linspace(bounds[1][0], bounds[1][1], density)
        pts_param = np.array([[x,y] for x in xs for y in ys if dom.contains((x,y))])
        emb_func = lambdify(chart.coords, chart.embedding.map_exprs, 'numpy')
        J = sp.Matrix(chart.embedding.map_exprs).jacobian(chart.coords)
        J_func = lambdify(chart.coords, J, 'numpy')
        fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
        for u,v in pts_param:
            P = np.array(emb_func(u,v),dtype=float).flatten()
            vec = np.array(vf.evaluate((u,v)),dtype=float)
            Jp = np.array(J_func(u,v),dtype=float)
            v_emb = Jp.dot(vec)
            ax.quiver(P[0],P[1],P[2] if len(P)>2 else 0, v_emb[0],v_emb[1],v_emb[2] if len(P)>2 else 0, length=scale, color=color)
        plt.title(f"Vector field on manifold ({chart.name})"); plt.show()

# ---------------- Part 4: Integral Curves and Geodesics ----------------
class FlowPlotter:
    """
    Plot integral curves (flows) and geodesics on the embedded manifold.
    """
    @staticmethod
    def plot_flow(
        vf: VectorField,
        chart: Chart,
        initial_points: List[Tuple[float, float]],
        t_span: Tuple[float, float] = (0, 1),
        num: int = 200,
        color: str = 'blue'
    ) -> None:
        from scipy.integrate import odeint
        sols = []
        for u0,v0 in initial_points:
            def ode(X,t): return vf.evaluate((X[0],X[1]))
            ts = np.linspace(t_span[0], t_span[1], num)
            sol = odeint(ode, [u0,v0], ts)
            sols.append(sol)
        emb_func = lambdify(chart.coords, chart.embedding.map_exprs, 'numpy')
        fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
        for sol in sols:
            pts = np.array([emb_func(x,y) for x,y in sol])
            ax.plot(pts[:,0], pts[:,1], pts[:,2] if pts.shape[1]==3 else None, color=color)
        plt.title(f"Flow lines on manifold ({chart.name})"); plt.show()

# ---------------- Part 5: Parallel Transport Visualization ----------------
class ConnectionPlotter:
    """
    Plot parallel-transported vectors along a curve on the manifold.
    """
    @staticmethod
    def plot_parallel_transport(
        conn: LeviCivitaConnection,
        curve_funcs: List[Function],
        init_vectors: List[Tuple[float,float]],
        t_span: Tuple[float,float] = (0,1),
        num: int = 200,
        color: str = 'magenta'
    ) -> None:
        from scipy.integrate import odeint
        t_vals = np.linspace(t_span[0], t_span[1], num)
        curve_num = np.array([[float(f(t)) for f in curve_funcs] for t in t_vals])
        emb_func = lambdify(conn.chart.coords, conn.chart.embedding.map_exprs, 'numpy')
        J = sp.Matrix(conn.chart.embedding.map_exprs).jacobian(conn.chart.coords)
        J_func = lambdify(conn.chart.coords, J, 'numpy')
        fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
        for v0 in init_vectors:
            def ode(V,t):
                idx = np.argmin(np.abs(t_vals - t))
                funcs_V = [Function('V0')(Symbol('t')), Function('V1')(Symbol('t'))]
                eqs = conn.parallel_transport_equations(curve_funcs, funcs_V)
                return [-float(eqs[0].rhs.subs({funcs_V[0]:V[0], funcs_V[1]:V[1]})),
                        -float(eqs[1].rhs.subs({funcs_V[0]:V[0], funcs_V[1]:V[1]}))]
            sol = odeint(ode, v0, t_vals)
            sample = np.linspace(0,len(t_vals)-1,10).astype(int)
            for idx in sample:
                pt_param = curve_num[idx]
                pt_emb = emb_func(*pt_param)
                v = sol[idx]
                Jp = np.array(J_func(*pt_param),dtype=float)
                v_emb = Jp.dot(v)
                ax.quiver(pt_emb[0],pt_emb[1],pt_emb[2] if len(pt_emb)>2 else 0,
                          v_emb[0],v_emb[1],v_emb[2] if len(v_emb)>2 else 0,
                          length=0.2,color=color)
        plt.title(f"Parallel transport on manifold ({conn.chart.name})"); plt.show()

