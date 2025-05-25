import sympy as sp
from sympy import symbols, Matrix, diff, simplify, latex, Function
from itertools import product

# Numeric solver dependencies
try:
    import numpy as np
    from scipy.integrate import odeint
except ImportError:
    np = None
    odeint = None

# Interactive visualization dependencies
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import ipywidgets as widgets
    from IPython.display import display, Latex
except ImportError:
    plt = None
    widgets = None
    display = None
    Latex = None

class Chart:
    """
    A coordinate chart on a manifold.
    - name: identifier
    - coords: list of sympy symbols
    - embedding: list of sympy expressions mapping coords to R^m for visualization
    - domain: function taking numeric tuple returns bool (point lies in domain)
    """
    def __init__(self, name, coords, embedding=None, domain=lambda *args: True):
        self.name = name
        self.coords = coords
        self.dim = len(coords)
        self.embedding = embedding  # e.g., [x, y, z] for surface in R^3
        self.domain = domain
        self.transitions = {}  # maps target_chart_name -> transition function (dict sympy Expr)

    def add_transition(self, target_chart_name, mapping):
        """
        mapping: dict mapping self.coords -> expressions in target chart coords
        """
        self.transitions[target_chart_name] = mapping

class SymbolicManifold:
    """
    An abstract manifold represented by multiple charts.
    """
    def __init__(self, name):
        self.name = name
        self.charts = {}

    def add_chart(self, chart: Chart):
        self.charts[chart.name] = chart

    def get_chart(self, name):
        return self.charts[name]

class MetricTensor:
    """
    Represents a metric tensor in a given chart.
    """
    def __init__(self, chart: Chart, metric_matrix: Matrix):
        assert metric_matrix.shape == (chart.dim, chart.dim)
        self.chart = chart
        self.M = metric_matrix
        self.invM = simplify(self.M.inv())
        self.coords = chart.coords

    def christoffel_symbols(self):
        n = len(self.coords)
        Gamma = [[[None]*n for _ in range(n)] for __ in range(n)]
        for k,i,j in product(range(n), repeat=3):
            term = sum(
                self.invM[k,l]*(
                    diff(self.M[l,j], self.coords[i]) +
                    diff(self.M[l,i], self.coords[j]) -
                    diff(self.M[i,j], self.coords[l])
                ) for l in range(n)
            )
            Gamma[k][i][j] = simplify(term/2)
        return Gamma

    def latex(self):
        return latex(self.M)

class VectorField:
    """
    Symbolic vector field within a specific chart.
    """
    def __init__(self, chart: Chart, components):
        assert len(components)==chart.dim
        self.chart = chart
        self.components = components

    def lie_bracket(self, other):
        C = self.chart
        bracket = []
        for k in range(C.dim):
            term = sum(
                self.components[j]*diff(other.components[k], C.coords[j])
                - other.components[j]*diff(self.components[k], C.coords[j])
                for j in range(C.dim)
            )
            bracket.append(simplify(term))
        return VectorField(C, bracket)

    def flow_equations(self):
        t = sp.symbols('t')
        curve = [Function(str(c))(t) for c in self.chart.coords]
        eqs = []
        subs_map = {self.chart.coords[j]: curve[j] for j in range(len(curve))}
        for i, xi in enumerate(curve):
            eqs.append(diff(xi, t) - self.components[i].subs(subs_map))
        return eqs, curve

class LeviCivitaConnection:
    def __init__(self, metric: MetricTensor):
        self.metric = metric
        self.coords = metric.coords
        self.Gamma = metric.christoffel_symbols()

    def covariant_derivative(self, vector_field, direction_index):
        n = len(self.coords)
        result = []
        for k in range(n):
            term = diff(vector_field.components[k], self.coords[direction_index])
            term += sum(self.Gamma[k][direction_index][j]*vector_field.components[j]
                        for j in range(n))
            result.append(simplify(term))
        return result

    def parallel_transport_equations(self, curve, vector):
        t = sp.symbols('t')
        eqs = []
        for k in range(len(self.coords)):
            total = diff(vector[k], t)
            total += sum(self.Gamma[k][i][j]*diff(curve[i], t)*vector[j]
                         for i,j in product(range(len(self.coords)), repeat=2))
            eqs.append(simplify(total))
        return eqs

    def geodesic_equations(self):
        t = sp.symbols('t')
        curve = [Function(str(c))(t) for c in self.coords]
        eqs = []
        for k in range(len(self.coords)):
            term = diff(curve[k], (t,2))
            term += sum(self.Gamma[k][i][j]*diff(curve[i],t)*diff(curve[j],t)
                        for i,j in product(range(len(self.coords)), repeat=2))
            eqs.append(simplify(term))
        return eqs, curve

    def latex_christoffel(self):
        latex_str = ''
        for k,i,j in product(range(len(self.coords)), repeat=3):
            expr = self.Gamma[k][i][j]
            if expr!=0:
                latex_str += f"\\Gamma^{{{k}}}_{{{i}{j}}} = {latex(expr)}\\\\\n"
        return latex_str

class LaTeXExporter:
    @staticmethod
    def export(expr, filename):
        with open(filename,'w') as f:
            f.write(sp.latex(expr))
        print(f"Exported LaTeX to {filename}")

class Visualizer:
    @staticmethod
    def plot_on_manifold(chart: Chart, embedding_map, points, vectors=None):
        """
        Scatter and optional tangent vectors on embedded manifold.
        - chart: coordinate chart.
        - embedding_map: list of sympy Expr mapping chart coords to R^m.
        - points: array of shape (N, chart.dim) numeric samples in chart coords.
        - vectors: optional list of sympy Expr (chart.dim) or numeric array (N, chart.dim).
        """
        if plt is None:
            print("matplotlib not available")
            return
        # Lambdify embedding and Jacobian
        emb_func = sp.lambdify(chart.coords, embedding_map, 'numpy')
        J = sp.Matrix(embedding_map).jacobian(chart.coords)
        J_func = sp.lambdify(chart.coords, J, 'numpy')
        # Compute embedded points once
        pts_emb = [emb_func(*tuple(p)) for p in points]
        fig = plt.figure()
        is3d = len(pts_emb[0]) == 3
        ax = fig.add_subplot(111, projection='3d' if is3d else None)
        # Plot points
        if is3d:
            ax.scatter(*zip(*pts_emb), s=10)
        else:
            plt.scatter(*zip(*pts_emb), s=10)
        # Plot vectors if provided
        if vectors is not None:
            for p, p_emb in zip(points, pts_emb):
                # evaluate vector
                if hasattr(vectors[0], 'subs'):
                    vec = [float(comp.subs({chart.coords[i]: p[i] for i in range(chart.dim)})) for comp in vectors]
                else:
                    vec = vectors[points.index(p)]
                # pushforward via Jacobian
                Jp = np.array(J_func(*tuple(p)), float)
                vec_emb = Jp.dot(vec)
                if is3d:
                    ax.quiver(p_emb[0], p_emb[1], p_emb[2], vec_emb[0], vec_emb[1], vec_emb[2], length=0.5)
                else:
                    plt.quiver(p_emb[0], p_emb[1], vec_emb[0], vec_emb[1])
        plt.title(f"Data on chart {chart.name}")
        plt.show()

    @staticmethod
    def flow_on_manifold(vector_field: VectorField, chart: Chart, embedding_map,
                         seed, t_span=(0,1), num=200):
        """
        Integrate flow dx/dt = V(x) in chart and embed trajectory.
        """
        if odeint is None or plt is None:
            print("Dependencies not available")
            return
        def ode_system(X, t):
            subs = {chart.coords[i]: X[i] for i in range(chart.dim)}
            return [float(vector_field.components[i].subs(subs)) for i in range(chart.dim)]
        t_vals = np.linspace(t_span[0], t_span[1], num)
        sol = odeint(ode_system, seed, t_vals)
        emb = sp.lambdify(chart.coords, embedding_map, 'numpy')
        traj = [emb(*tuple(pt)) for pt in sol]
        fig = plt.figure()
        is3d = len(traj[0])==3
        ax = fig.add_subplot(111, projection='3d' if is3d else None)
        if is3d:
            ax.plot(*zip(*traj))
        else:
            plt.plot(*zip(*traj))
        plt.title(f"Flow curve on chart {chart.name}")
        plt.show()

# Example usage: 2-chart sphere
if __name__=="__main__":
    u,v= symbols('u v')
    chart_N = Chart('north', [u,v],
                    embedding=[2*u/(1+u**2+v**2), 2*v/(1+u**2+v**2), ( -1+u**2+v**2)/(1+u**2+v**2) ])
    chart_S = Chart('south', [u,v],
                    embedding=[2*u/(1+u**2+v**2), 2*v/(1+u**2+v**2), (1-u**2-v**2)/(1+u**2+v**2) ])
    chart_N.add_transition('south', {u: u/(u**2+v**2), v: v/(u**2+v**2)})
    chart_S.add_transition('north', {u: u/(u**2+v**2), v: v/(u**2+v**2)})
    M = SymbolicManifold('S2')
    M.add_chart(chart_N)
    M.add_chart(chart_S)
    Vn = VectorField(chart_N, [ -v, u ])
    # Generate random points in chart domain
    pts = np.random.rand(20,2)*2-1
    # Visualize embedded vector field
    Visualizer.plot_on_manifold(chart_N, chart_N.embedding, pts, vectors=Vn.components)
    # Plot a flow curve
    Visualizer.flow_on_manifold(Vn, chart_N, chart_N.embedding, seed=[0.5,0.0])
