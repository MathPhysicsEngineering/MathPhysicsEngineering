import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from charts import (
    Chart,
    SymbolicManifold,
    BoxDomain,
    ParametricEmbedding
)
from Riemannian_metric import RiemannianMetric
from connections import LeviCivitaConnection
from vector_fields import VectorField
from visualization import (
    ManifoldPlotter,
    VectorFieldPlotter,
    ConnectionPlotter
)
from latex_exporter import LaTeXExporter

# -----------------------------------------------------------------------------
# 1) Set up sphere atlas (single chart in spherical coords)
# -----------------------------------------------------------------------------
phi, theta = sp.symbols('phi theta')
map_exprs = [
    sp.sin(phi)*sp.cos(theta),
    sp.sin(phi)*sp.sin(theta),
    sp.cos(phi),
]
chart = Chart(
    name='sphere',
    coords=[phi, theta],
    domain=BoxDomain([(0.0, np.pi), (0.0, 2.0*np.pi)]),
    embedding=ParametricEmbedding([phi, theta], map_exprs)
)
M = SymbolicManifold('S2')
M.add_chart(chart, default=True)

# -----------------------------------------------------------------------------
# 2) Define metric g = dφ² + sin²φ dθ², build connection, print Christoffel
# -----------------------------------------------------------------------------
g = sp.diag(1, sp.sin(phi)**2)
metric = RiemannianMetric([phi, theta], g)
conn = LeviCivitaConnection(metric, chart)

print("\nNonzero Christoffel symbols (Γᵏ_{ij}):")
Gamma = metric.christoffel_symbols()
for k in range(2):
    for i in range(2):
        for j in range(2):
            if Gamma[k][i][j] != 0:
                print(f"  Γ^{k}_{{{i}{j}}} =", Gamma[k][i][j])
# write them out to LaTeX
LaTeXExporter.christoffel(conn, 'sphere_christoffel.tex')

# -----------------------------------------------------------------------------
# 3) Visualize the embedded sphere (transparent) + vector field overlay
# -----------------------------------------------------------------------------
# a) Plot the transparent sphere
ManifoldPlotter.plot_manifold(chart, resolution=60, color='lightblue', alpha=0.3)

# b) Define a vector field: unit “eastward” flow ∂/∂θ
vf = VectorField(chart, [0, 1])

# c) Overlay quiver arrows on the same manifold
VectorFieldPlotter.plot_vector_field(vf, chart, density=20, scale=0.2, color='crimson')

# -----------------------------------------------------------------------------
# 4) Integral curves of that vector field starting at various latitudes
# -----------------------------------------------------------------------------
# build numeric embedding
emb_func = sp.lambdify(chart.coords, map_exprs, 'numpy')
def flow_ode(X, t):
    return vf.evaluate((X[0], X[1]))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# draw sphere lightly in background
U = np.linspace(0, np.pi, 40)
V = np.linspace(0, 2*np.pi, 80)
UU, VV = np.meshgrid(U, V)
XYZ = np.array([emb_func(u,v) for u,v in np.vstack([UU.flatten(), VV.flatten()]).T])
X = XYZ[:,0].reshape(UU.shape); Y = XYZ[:,1].reshape(UU.shape); Z = XYZ[:,2].reshape(UU.shape)
ax.plot_surface(X, Y, Z, color='lightgray', alpha=0.2)

# sample a few starting points
starts = [(np.pi/6, 0), (np.pi/3, 0), (np.pi/2, 0), (2*np.pi/3, 0)]
t_vals = np.linspace(0, 4*np.pi, 400)
for p0 in starts:
    sol = odeint(flow_ode, [p0[0], p0[1]], t_vals)
    curve = np.array([emb_func(u,th) for u,th in sol])
    ax.plot(curve[:,0], curve[:,1], curve[:,2], label=f'φ₀={p0[0]:.2f}')

ax.legend()
ax.set_title("Integral curves of ∂/∂θ on S²")
plt.show()

# -----------------------------------------------------------------------------
# 5) Parallel transport: carry a vector along the equator
# -----------------------------------------------------------------------------
# define the equatorial curve φ(t)=π/2, θ(t)=t
t = sp.symbols('t')
phi_t = sp.Lambda(t, sp.pi/2)
theta_t = sp.Lambda(t, t)
# we need actual sympy Function objects for ConnectionPlotter:
Ft = sp.Function
curve_funcs = [Ft('φ')(t), Ft('θ')(t)]
# override their definitions for d/dt when integrating:
# Transport the tangent vector (1,0) at t=0 all around
init_vecs = [(1.0, 0.0)]
ConnectionPlotter.plot_parallel_transport(
    conn,
    curve_funcs=curve_funcs,
    init_vectors=init_vecs,
    t_span=(0, 2*np.pi),
    num=200,
    color='magenta'
)

# -----------------------------------------------------------------------------
# 6) Exponential map at a base point p₀ and its visualization
# -----------------------------------------------------------------------------
def exponential_map(conn, p0, v0, t_max=1.0, num=100):
    """
    Numerically integrate the geodesic ODE with initial (p0, v0)
    and return the curve of points on the manifold.
    """
    eqs, funcs = conn.geodesic_equations()
    # funcs = [φ(t), θ(t)]
    def geo(Y, tt):
        subs = {
            funcs[0]: Y[0], funcs[1]: Y[1],
            sp.diff(funcs[0],t): Y[2], sp.diff(funcs[1],t): Y[3]
        }
        # RHS: [φ', θ', φ'', θ'']
        phi_dot, theta_dot = Y[2], Y[3]
        accel = [
            -float(eqs[0].rhs.subs(subs)),
            -float(eqs[1].rhs.subs(subs))
        ]
        return [phi_dot, theta_dot, accel[0], accel[1]]

    Y0 = [p0[0], p0[1], v0[0], v0[1]]
    ts = np.linspace(0, t_max, num)
    sol = odeint(geo, Y0, ts)
    return sol  # columns: [φ, θ, φ', θ']

# choose base point and two orthonormal directions
p0 = (np.pi/4, 0.0)
# tangent-space basis: ∂/∂φ and (1/sinφ) ∂/∂θ at p0
v_phi = np.array([1.0, 0.0])
v_theta = np.array([0.0, 1.0/np.sin(p0[0])])

# plot exp map curves
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, color='lightgray', alpha=0.2)  # sphere background

for v0 in [v_phi, v_theta]:
    sol = exponential_map(conn, p0, v0, t_max=2.0, num=200)
    pts = np.array([emb_func(u,th) for u,th,_,_ in sol])
    ax.plot(pts[:,0], pts[:,1], pts[:,2], linewidth=2)

# mark the base point and initial tangent vectors
P0 = emb_func(*p0)
ax.scatter(*P0, color='black', s=50)
ax.quiver(
    P0[0], P0[1], P0[2],
    v_phi[0], v_phi[1], 0,
    length=0.5, color='green'
)
ax.quiver(
    P0[0], P0[1], P0[2],
    v_theta[0], v_theta[1], 0,
    length=0.5, color='blue'
)

ax.set_title("Exponential map curves from p₀ on S²")
plt.show()

# -----------------------------------------------------------------------------
# 7) Export everything to LaTeX
# -----------------------------------------------------------------------------
LaTeXExporter.metric(metric, 'sphere_metric.tex')
LaTeXExporter.christoffel(conn, 'sphere_christoffel.tex')
eqs, _ = conn.geodesic_equations()
with open('sphere_geodesics.tex', 'w') as f:
    f.write("\n".join(sp.latex(eq) for eq in eqs))

print("✅ Sphere demo complete — all features exercised and visualized.")
