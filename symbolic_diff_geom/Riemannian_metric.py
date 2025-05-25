import sympy as sp
from sympy import Matrix, Expr, symbols, simplify, latex
from typing import List, Tuple, Dict, Any, Callable

class RiemannianMetric:
    """
    Represents a Riemannian metric on an n-dimensional chart.

    Attributes:
        coords: List of sympy Symbols for local coordinates.
        g: sympy Matrix representing the metric (0,2)-tensor.
        invg: inverse metric, raised indices.
    """
    def __init__(self, coords: List[sp.Symbol], metric_matrix: Matrix):
        # Validate dimensions
        n = len(coords)
        if metric_matrix.shape != (n, n):
            raise ValueError(f"Metric matrix must be {n}x{n} for coords length {n}.")
        self.coords = coords
        self.g = simplify(metric_matrix)
        self.invg = simplify(self.g.inv())
        # Initialize cache for computed tensors
        self._cache: Dict[str, Any] = {}

    def clear_cache(self) -> None:
        """
        Clear all cached computations (Christoffel, Riemann, Ricci, scalar).
        """
        self._cache.clear()

    def get_cached(self, key: str, compute_fn: Callable[[], Any]) -> Any:
        """
        Retrieve a value from cache by key, or compute and cache it if missing.
        """
        if key not in self._cache:
            self._cache[key] = compute_fn()
        return self._cache[key]

    def to_latex(self) -> str:
        """
        Export the metric matrix to a LaTeX bmatrix.
        """
        return latex(self.g)

    def christoffel_symbols(self) -> List[List[List[Expr]]]:
        """
        Compute or fetch cached Christoffel symbols Γ^k_{ij}.
        """
        return self.get_cached('Gamma', self._compute_christoffel)

    def _compute_christoffel(self) -> List[List[List[Expr]]]:
        """
        Internal: compute Christoffel symbols without caching logic.
        """
        n = len(self.coords)
        Gamma: List[List[List[Expr]]] = [[[0]*n for _ in range(n)] for _ in range(n)]
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    expr = 0
                    for l in range(n):
                        expr += (
                            self.invg[k, l] * (
                                sp.diff(self.g[l, j], self.coords[i]) +
                                sp.diff(self.g[l, i], self.coords[j]) -
                                sp.diff(self.g[i, j], self.coords[l])
                            )
                        )
                    Gamma[k][i][j] = simplify(expr / 2)
        return Gamma

    def riemann_tensor(self) -> List[List[List[List[Expr]]]]:
        """
        Compute or fetch cached Riemann curvature tensor R^i_{ jkl }.
        """
        return self.get_cached('Riemann', self._compute_riemann)

    def _compute_riemann(self) -> List[List[List[List[Expr]]]]:
        coords = self.coords
        n = len(coords)
        Gamma = self.christoffel_symbols()
        R = [[[[0]*n for _ in range(n)] for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        term1 = sp.diff(Gamma[i][j][l], coords[k])
                        term2 = sp.diff(Gamma[i][j][k], coords[l])
                        term3 = sum(Gamma[i][k][m] * Gamma[m][j][l] for m in range(n))
                        term4 = sum(Gamma[i][l][m] * Gamma[m][j][k] for m in range(n))
                        R[i][j][k][l] = simplify(term1 - term2 + term3 - term4)
        return R

    def ricci_tensor(self) -> List[List[Expr]]:
        """
        Compute or fetch cached Ricci tensor Ric_{ij} via contraction.
        """
        return self.get_cached('Ricci', self._compute_ricci)

    def _compute_ricci(self) -> List[List[Expr]]:
        R = self.riemann_tensor()
        n = len(self.coords)
        Ric = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                Ric[i][j] = sum(R[k][i][k][j] for k in range(n))
        return Ric

    def scalar_curvature(self) -> Expr:
        """
        Compute or fetch cached scalar curvature R.
        """
        return self.get_cached('Scalar', self._compute_scalar)

    def _compute_scalar(self) -> Expr:
        Ric = self.ricci_tensor()
        n = len(self.coords)
        return simplify(
            sum(self.invg[i, j] * Ric[i][j] for i in range(n) for j in range(n))
        )

    def sectional_curvature(self, u: Tuple[Expr, ...], v: Tuple[Expr, ...]) -> Expr:
        """
        Compute sectional curvature K(u,v) with validation.
        """
        if len(u) != len(self.coords) or len(v) != len(self.coords):
            raise ValueError("Vectors must match metric dimension.")
        R = self.riemann_tensor()
        n = len(self.coords)
        num = sum(
            R[i][j][k][l] * u[i] * v[j] * u[k] * v[l]
            for i in range(n) for j in range(n) for k in range(n) for l in range(n)
        )
        denom = sum(
            (self.g[i, k]*self.g[j, l] - self.g[i, l]*self.g[j, k]) *
            u[i] * v[j] * u[k] * v[l]
            for i in range(n) for j in range(n) for k in range(n) for l in range(n)
        )
        return simplify(num/denom)

    def to_latex_all(self) -> str:
        """
        Export metric, Christoffel, Riemann, Ricci, scalar curvature as LaTeX.
        """
        sections: List[str] = []
        sections.append(r"\textbf{Metric:}" )
        sections.append(latex(self.g))
        # Christoffel
        lines = []
        Gamma = self.christoffel_symbols()
        n = len(self.coords)
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    expr = Gamma[k][i][j]
                    if expr != 0:
                        lines.append(rf"\Gamma^{{{k}}}_{{{i}{j}}} = {latex(expr)} ")
        sections.append(r"\textbf{Christoffel:} ")
        sections.extend(lines)
        # Riemann
        lines = []
        R = self.riemann_tensor()
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        expr = R[i][j][k][l]
                        if expr != 0:
                            lines.append(rf"R^{{{i}}}_{{{j}{k}{l}}} = {latex(expr)} ")
        sections.append(r"\textbf{Riemann:} ")
        sections.extend(lines)
        # Ricci
        lines = []
        Ric = self.ricci_tensor()
        for i in range(n):
            for j in range(n):
                expr = Ric[i][j]
                if expr != 0:
                    lines.append(rf"Ric_{{{i}{j}}} = {latex(expr)} ")
        sections.append(r"\textbf{Ricci:} ")
        sections.extend(lines)
        # Scalar
        sections.append(r"\textbf{Scalar:} ")
        sections.append(latex(self.scalar_curvature()))
        return "\n".join(sections)

    def pullback(self, mapping: Dict[sp.Symbol, Expr]) -> 'RiemannianMetric':
        """
        Compute the pullback of this metric under a coordinate transformation.
        """
        new_coords = list(mapping.keys())
        g_sub = self.g.subs({old: mapping[old] for old in self.coords})
        J = sp.Matrix([
            [sp.diff(mapping[old], new) for old in self.coords]
            for new in new_coords
        ])
        g_pulled = simplify(J * g_sub * J.T)
        return RiemannianMetric(new_coords, g_pulled)

    def geodesic_deviation(self, u: Tuple[Any, ...], w: Tuple[Any, ...]) -> List[Expr]:
        """
        Compute geodesic deviation: D^2 w^i/ds^2 + R^i_{jkl} u^j w^k u^l = 0.
        """
        if len(u) != len(self.coords) or len(w) != len(self.coords):
            raise ValueError("Vectors must match metric dimension.")
        R = self.riemann_tensor()
        n = len(self.coords)
        return [simplify(sum(R[i][j][k][l] * u[j] * w[k] * u[l]
                        for j in range(n) for k in range(n) for l in range(n)))
                for i in range(n)]

    def __repr__(self) -> str:
        return f"<RiemannianMetric dim={len(self.coords)} coords={self.coords}>"

# End of Riemannian_metric.py
