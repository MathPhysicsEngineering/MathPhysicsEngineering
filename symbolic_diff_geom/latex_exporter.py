import sympy as sp
from sympy import Expr, Function, symbols, latex as sympy_latex
from typing import Any, List, Union, Tuple

class LaTeXExporter:
    """
    Consolidate LaTeX export for various symbolic objects in the symbolic_diff_geom package.

    Methods:
      - metric            : export metric matrices
      - christoffel       : export Christoffel symbols
      - vector_field      : export vector field components
      - riemann_tensor    : export Riemann curvature components
      - ricci_tensor      : export Ricci tensor components
      - scalar_curvature  : export scalar curvature
      - geodesics         : export geodesic equations
      - parallel_transport: export parallel transport ODEs
      - flow_equations    : export flow ODEs
      - general           : export any sympy Expr
    """

    @staticmethod
    def metric(metric: Any, filename: str) -> None:
        """
        Export a metric matrix g_{ij} to a .tex file.
        """
        tex = sympy_latex(metric.g)
        with open(filename, 'w') as f:
            f.write("\\[")
            f.write(tex)
            f.write("\\]")

    @staticmethod
    def christoffel(connection: Any, filename: str) -> None:
        """
        Export nonzero Christoffel symbols Γ^k_{ij}.
        """
        lines: List[str] = []
        for k, row in enumerate(connection.Gamma):
            for i, col in enumerate(row):
                for j, expr in enumerate(col):
                    if expr != 0:
                        lines.append(rf"\\Gamma^{{{k}}}_{{{i}{j}}} = {sympy_latex(expr)}\\")
        with open(filename, 'w') as f:
            f.write("\\n".join(lines))

    @staticmethod
    def vector_field(vf: Any, filename: str) -> None:
        """
        Export vector field V^i(x) components.
        """
        lines: List[str] = []
        for sym, expr in vf.as_dict().items():
            lines.append(rf"V^{{{sym}}} = {sympy_latex(expr)}\\")
        with open(filename, 'w') as f:
            f.write("\\n".join(lines))

    @staticmethod
    def riemann_tensor(metric: Any, filename: str) -> None:
        """
        Export nonzero Riemann tensor components R^i_{jkl}.
        """
        R = metric.riemann_tensor()
        lines: List[str] = []
        n = len(metric.coords)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        expr = R[i][j][k][l]
                        if expr != 0:
                            lines.append(rf"R^{{{i}}}_{{{j}{k}{l}}} = {sympy_latex(expr)}\\")
        with open(filename, 'w') as f:
            f.write("\\n".join(lines))

    @staticmethod
    def ricci_tensor(metric: Any, filename: str) -> None:
        """
        Export nonzero Ricci tensor Ric_{ij}.
        """
        Ric = metric.ricci_tensor()
        lines: List[str] = []
        n = len(metric.coords)
        for i in range(n):
            for j in range(n):
                expr = Ric[i][j]
                if expr != 0:
                    lines.append(rf"Ric_{{{i}{j}}} = {sympy_latex(expr)}\\")
        with open(filename, 'w') as f:
            f.write("\\n".join(lines))

    @staticmethod
    def scalar_curvature(metric: Any, filename: str) -> None:
        """
        Export scalar curvature R.
        """
        Rsc = metric.scalar_curvature()
        with open(filename, 'w') as f:
            f.write(rf"R = {sympy_latex(Rsc)}")

    @staticmethod
    def geodesics(connection: Any, filename: str) -> None:
        """
        Export geodesic equations x''^i + Γ^i_{jk} x'^j x'^k = 0.
        """
        eqs, funcs = connection.geodesic_equations()
        lines = [sympy_latex(eq) for eq in eqs]
        with open(filename, 'w') as f:
            f.write("\\n".join(lines))

    @staticmethod
    def parallel_transport(connection: Any, curve_funcs: List[Function], filename: str) -> None:
        """
        Export parallel transport ODEs along a curve.
        """
        vec_syms = [Function(f'V{i}')(symbols('t')) for i in range(len(curve_funcs))]
        eqs = connection.parallel_transport_equations(curve_funcs, vec_syms)
        lines = [sympy_latex(eq) for eq in eqs]
        with open(filename, 'w') as f:
            f.write("\\n".join(lines))

    @staticmethod
    def flow_equations(vf: Any, filename: str) -> None:
        """
        Export flow ODEs dx^i/dt = V^i(x).
        """
        eqs, funcs = vf.flow_equations()
        lines = [sympy_latex(eq) for eq in eqs]
        with open(filename, 'w') as f:
            f.write("\\n".join(lines))

    @staticmethod
    def general(expr: Expr, filename: str) -> None:
        """
        Export any sympy expression to LaTeX.
        """
        tex = sympy_latex(expr)
        with open(filename, 'w') as f:
            f.write("\\[")
            f.write(tex)
            f.write("\\]")

# End of latex_exporter.py
