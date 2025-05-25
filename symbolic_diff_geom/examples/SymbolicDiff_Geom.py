import sympy as sp
from sympy import symbols, Matrix, diff, simplify, latex
from itertools import product
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets
from IPython.display import display, Latex

# Interactive visualization dependencies


class SymbolicManifold:
    """
    Represents an n-dimensional manifold with coordinate chart.
    """
    def __init__(self, coord_symbols):
        self.coords = coord_symbols
        self.dim = len(coord_symbols)


class MetricTensor:
    """
    Represents a metric tensor on a manifold.
    """
    def __init__(self, manifold, metric_matrix):
        assert metric_matrix.shape == (manifold.dim, manifold.dim)
        self.g = metric_matrix
        self.invg = simplify(self.g.inv())
        self.coords = manifold.coords

    def christoffel_symbols(self):
        """
        Compute Christoffel symbols \Gamma^k_{ij}.
        Returns a 3D list Gamma[k][i][j].
        """
        n = len(self.coords)
        Gamma = [[[None]*n for _ in range(n)] for __ in range(n)]
        for k, i, j in product(range(n), repeat=3):
            term = 0
            for l in range(n):
                term += self.invg[k,l] * (
                    diff(self.g[l,j], self.coords[i]) +
                    diff(self.g[l,i], self.coords[j]) -
                    diff(self.g[i,j], self.coords[l])
                )
            Gamma[k][i][j] = simplify(term/2)
        return Gamma

    def latex(self):
        """
        Export metric tensor to LaTeX.
        """
        return latex(self.g)

class LeviCivitaConnection:
    """
    Encapsulates covariant derivative and parallel transport.
    """
    def __init__(self, metric: MetricTensor):
        self.metric = metric
        self.Gamma = metric.christoffel_symbols()
        self.coords = metric.coords

    def covariant_derivative(self, vector_field, direction_index):
        """
        Compute (∇_direction V)^k where V is a list of sympy expressions for vector components.
        """
        n = len(self.coords)
        result = [None]*n
        for k in range(n):
            term = diff(vector_field[k], self.coords[direction_index])
            for j in range(n):
                term += self.Gamma[k][direction_index][j] * vector_field[j]
            result[k] = simplify(term)
        return result

    def parallel_transport_equations(self, curve, vector):
        """
        Return ODE system dV^k/dt + Γ^k_{ij} x'^i(t) V^j = 0 for parallel transport along curve x(t).
        """
        t = sp.symbols('t')
        n = len(self.coords)
        eqs = []
        # curve: list of functions x_i(t)
        for k in range(n):
            total = diff(vector[k], t)
            for i, j in product(range(n), repeat=2):
                total += self.Gamma[k][i][j] * diff(curve[i], t) * vector[j]
            eqs.append(simplify(total))
        return eqs

    def geodesic_equations(self):
        """
        Return the geodesic equations: x''^k + Γ^k_{ij} x'^i x'^j = 0.
        """
        t = sp.symbols('t')
        n = len(self.coords)
        eqs = []
        curve = [sp.Function(str(c))(t) for c in self.coords]
        for k in range(n):
            term = diff(curve[k], (t, 2))
            for i, j in product(range(n), repeat=2):
                term += self.Gamma[k][i][j] * diff(curve[i], t) * diff(curve[j], t)
            eqs.append(simplify(term))
        return eqs, curve

    def latex_christoffel(self):
        """
        Export Christoffel symbols to LaTeX.
        """
        Gamma = self.Gamma
        latex_str = ""
        n = len(self.coords)
        for k, i, j in product(range(n), repeat=3):
            expr = Gamma[k][i][j]
            if expr != 0:
                latex_str += f"\\Gamma^{{{k}}}_{{{i}{j}}} = {latex(expr)}\\\\\n"
        return latex_str

class LaTeXExporter:
    """
    Utility to export any symbolic expressions to LaTeX file or display.
    """
    @staticmethod
    def export(expr, filename):
        with open(filename, 'w') as f:
            f.write(sp.latex(expr))
        print(f"Exported LaTeX to {filename}")

class Visualizer:
    """
    Provides interactive demos in Jupyter notebooks.
    """
    @staticmethod
    def interactive_christoffel(connection: LeviCivitaConnection):
        if not widgets:
            print("ipywidgets not available")
            return
        n = len(connection.coords)
        i_sel = widgets.IntSlider(min=0, max=n-1, description='i')
        j_sel = widgets.IntSlider(min=0, max=n-1, description='j')
        k_sel = widgets.IntSlider(min=0, max=n-1, description='k')
        out = widgets.Output()
        def update(i, j, k):
            out.clear_output()
            with out:
                expr = connection.Gamma[k][i][j]
                display(Latex(f"$\\Gamma^{{{k}}}_{{{i}{j}}} = {latex(expr)}$"))
        ui = widgets.VBox([i_sel, j_sel, k_sel])
        widgets.interact(update, i=i_sel, j=j_sel, k=k_sel)
        display(ui, out)

    @staticmethod
    def plot_geodesic(connection: LeviCivitaConnection, metric: MetricTensor, initial_point, initial_vel, t_span=(0,1), num=100):
        if odeint is None or plt is None:
            print("Numeric solver or matplotlib not available")
            return
        # Prepare numeric Christoffel
        coords = metric.coords
        Gamma = connection.Gamma
        def geodesic_ode(Y, t):
            n = len(coords)
            x = Y[:n]
            v = Y[n:]
            dxdt = v
            dvdt = np.zeros(n)
            subs = {coords[i]: x[i] for i in range(n)}
            for k in range(n):
                s = 0
                for i, j in product(range(n), repeat=2):
                    s += float(Gamma[k][i][j].subs(subs)) * v[i] * v[j]
                dvdt[k] = -s
            return np.concatenate([dxdt, dvdt])
        Y0 = np.concatenate([np.array(initial_point, float), np.array(initial_vel, float)])
        t_vals = np.linspace(t_span[0], t_span[1], num)
        sol = odeint(geodesic_ode, Y0, t_vals)
        if len(initial_point) == 2:
            plt.figure()
            plt.plot(sol[:,0], sol[:,1])
            plt.xlabel(str(coords[0])); plt.ylabel(str(coords[1])); plt.title('Geodesic')
            plt.show()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(sol[:,0], sol[:,1], sol[:,2])
            ax.set_title('Geodesic in 3D')
            plt.show()

# Example usage:
if __name__ == "__main__":
    # 2D polar coordinates example
    r, th = symbols('r th', positive=True)
    coords = [r, th]
    M = Matrix([[1, 0], [0, r**2]])
    man = SymbolicManifold(coords)
    metric = MetricTensor(man, M)
    conn = LeviCivitaConnection(metric)
    print("Metric LaTeX:")
    print(metric.latex())
    print("Christoffel LaTeX:")
    print(conn.latex_christoffel())
    # Interactive demos require Jupyter environment
