# polytrope_app.py  –  Interactive Polytropic Star Explorer
# Run with:  streamlit run polytrope_app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Polytropic Star Explorer", layout="wide")
st.title("🌟 Polytropic Star Explorer")
st.caption("Based on G. Cowan, RHUL Physics (2021), J. Maund, RHUL Physics (2026) — interactive version")

# ── Sidebar: tunable parameters ────────────────────────────────────────────────
st.sidebar.header("Parameters")

gamma = st.sidebar.slider(
    "Adiabatic index γ  (n = 1/(γ−1))",
    min_value=1.01, max_value=2.0, value=4/3, step=0.01,
    help="γ = 4/3 → n = 3 (radiation-dominated, Eddington model)\n"
         "γ = 5/3 → n = 1.5 (non-relativistic degenerate)",
)
n_poly = 1.0 / (gamma - 1.0)
st.sidebar.markdown(f"**Polytropic index n = {n_poly:.2f}**")

rhoc = st.sidebar.number_input(
    "Central density ρ_c  [kg/m³]",
    min_value=1e3, max_value=1e8, value=2e5, step=1e4, format="%.2e",
    help="Boundary condition at r = 0",
)

Pc = st.sidebar.number_input(
    "Central pressure P_c  [J/m³]",
    min_value=1e13, max_value=1e20, value=4e16, step=1e14, format="%.2e",
    help="Sets the equation of state constant K",
)

dx = st.sidebar.select_slider(
    "Integration step size dx",
    options=[1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
    value=1e-5,
    format_func=lambda x: f"{x:.0e}",
    help="Smaller = more accurate but slower",
)

show_mass = st.sidebar.checkbox("Show enclosed mass profile", value=False)
show_steps = st.sidebar.checkbox("Show solver diagnostics", value=True)

# ── Physical constants (fixed) ──────────────────────────────────────────────────
G_N  = 6.674e-11   # m^3 / kg / s^2
MSun = 1.989e30    # kg
RSun = 6.9634e8    # m

# ── Theory expander ─────────────────────────────────────────────────────────────
with st.expander("📐 The physics & equations", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Equation of state (polytrope)**")
        st.latex(r"P = K \rho^{\gamma}, \qquad \gamma = 1 + \frac{1}{n}")
        st.markdown("**Hydrostatic equilibrium**")
        st.latex(r"\frac{dP}{dr} = -\frac{G\, m(r)\,\rho}{r^2}")
        st.markdown("**Mass continuity**")
        st.latex(r"\frac{dm}{dr} = 4\pi r^2 \rho")
    with col2:
        st.markdown("**Dimensionless variables**  (scaling out units)")
        st.latex(r"x = \frac{r}{R_0}, \quad v = \frac{\rho}{\rho_c}, \quad u = \frac{m}{M_0}")
        st.latex(
            r"R_0 = \sqrt{\frac{K}{G}}\,\rho_c^{\,\gamma/2 - 1}, "
            r"\qquad M_0 = \rho_c R_0^3"
        )
        st.markdown("**The two ODEs actually integrated:**")
        st.latex(r"\frac{du}{dx} = 4\pi x^2\, v")
        st.latex(r"\frac{dv}{dx} = -\frac{1}{\gamma}\frac{u}{x^2}\,v^{2-\gamma}")

# ── Solver expander ──────────────────────────────────────────────────────────────
with st.expander("⚙️ How it is solved (forward Euler)", expanded=show_steps):
    st.markdown(
        """
        The two ODEs are stepped forward from the centre using the **forward Euler method**:

        $$f_{i+1} = f_i + dx \\cdot f'(x_i, f_i)$$

        where $f = (u, v)$.  Integration stops when the density $v$ would become negative
        (i.e. when the proposed step $|\\Delta v| \\geq v$), which defines the stellar surface.

        **Boundary conditions at** $x = 0$:
        - $v(0) = 1$ &nbsp; (density equals central density by definition)
        - $u(0) = 0$ &nbsp; (no enclosed mass at the centre)

        The $dv/dx$ derivative is set to zero at $x = 0$ to avoid a $1/x^2$ singularity —
        this is consistent with L'Hôpital's rule applied to the Lane–Emden equation.
        """
    )

# ── Derived constants ────────────────────────────────────────────────────────────
K   = Pc / rhoc**gamma
R0  = np.sqrt(K / G_N) * rhoc**(gamma / 2.0 - 1.0)
M0  = rhoc * R0**3

# ── ODE right-hand sides ─────────────────────────────────────────────────────────
def dudx(x, v):
    return 4.0 * np.pi * x**2 * v

def dvdx(x, u, v):
    if x > 0 and v > 0:
        return -(1.0 / gamma) * (u / x**2) * v**(2.0 - gamma)
    return 0.0

def derivs(x, f):
    u, v = f
    return np.array([dudx(x, v), dvdx(x, u, v)])

# ── Integrate ────────────────────────────────────────────────────────────────────
dxPlot = 1e-3

x   = 0.0
f   = np.array([0.0, 1.0])   # u=0, v=1 at centre

rPlot   = [0.0]
rhoPlot = [rhoc]
mPlot   = [0.0]
step_count = 0

takeStep = True
while takeStep:
    df = dx * derivs(x, f)
    takeStep = f[1] > abs(df[1])
    if takeStep:
        f += df
        x += dx
        step_count += 1
    if abs(x % dxPlot) < dx or not takeStep:
        rPlot.append(x * R0 / RSun)
        rhoPlot.append(f[1] * rhoc)
        mPlot.append(f[0] * M0 / MSun)

R_star = rPlot[-1]
M_star = mPlot[-1]

rPlot   = np.array(rPlot)
rhoPlot = np.array(rhoPlot)
mPlot   = np.array(mPlot)

# ── Results banner ───────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Radius",    f"{R_star:.3f} R☉")
col2.metric("Total Mass",      f"{M_star:.3f} M☉")
col3.metric("K (eq. of state)", f"{K:.3e}")
col4.metric("Steps taken",     f"{step_count:,}")

# ── Plot ─────────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({'font.size': 14})

if show_mass:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
else:
    fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))

fig.patch.set_facecolor('#0e1117')
for ax in ([ax1, ax2] if show_mass else [ax1]):
    ax.set_facecolor('#0e1117')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

# Density profile
ax1.plot(rPlot, rhoPlot, color='dodgerblue', linewidth=2)
ax1.set_xlabel(r'$r$  [$R_{\odot}$]')
ax1.set_ylabel(r'$\rho$  [kg m$^{-3}$]')
ax1.set_title(rf'Density profile  (γ={gamma:.2f}, n={n_poly:.2f})')
ax1.set_xlim(0, R_star * 1.2)
ax1.set_ylim(0, rhoPlot[0] * 1.2)
ax1.axvline(R_star, color='tomato', linestyle='--', linewidth=1, label=f'R = {R_star:.2f} R☉')
ax1.legend(facecolor='#1a1a2e', labelcolor='white')

# Mass profile (optional)
if show_mass:
    ax2.plot(rPlot, mPlot, color='orange', linewidth=2)
    ax2.set_xlabel(r'$r$  [$R_{\odot}$]')
    ax2.set_ylabel(r'$m(r)$  [$M_{\odot}$]')
    ax2.set_title(rf'Enclosed mass profile  (γ={gamma:.2f}, n={n_poly:.2f})')
    ax2.set_xlim(0, R_star * 1.2)
    ax2.set_ylim(0, M_star * 1.2)
    ax2.axhline(M_star, color='tomato', linestyle='--', linewidth=1, label=f'M = {M_star:.2f} M☉')
    ax2.legend(facecolor='#1a1a2e', labelcolor='white')

plt.tight_layout()
st.pyplot(fig)

# ── Solver diagnostics ───────────────────────────────────────────────────────────
if show_steps:
    st.markdown("---")
    st.subheader("Solver diagnostics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Step size dx",     f"{dx:.0e}")
    c2.metric("Plot sample every", f"{dxPlot:.0e}")
    c3.metric("Plot points",      f"{len(rPlot):,}")

    st.markdown(
        f"""
        The Euler integrator took **{step_count:,} steps** of size $dx = {dx:.0e}$
        across dimensionless radius $x \\in [0,\\ {x:.3f}]$.  
        The surface is found when the next Euler step would make $v < 0$
        (i.e. $|\\Delta v| \\geq v$), giving $R = {R_star:.4f}\\,R_\\odot$.
        """
    )
