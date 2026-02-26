# polytrope_compare.py  –  Polytropic Star: compare multiple n values
# Run with:  streamlit run polytrope_compare.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

st.set_page_config(page_title="Polytrope Comparison", layout="wide")
st.title("🌟 Polytropic Star — Comparing Different n")
st.caption("Based on G. Cowan, RHUL Physics (2021), J. Maund, RHUL Physics (2026) — interactive version")
st.markdown(
    r"""
    Each polytropic index $n$ gives a different **Lane–Emden solution** $\theta(\xi)$,
    and therefore a different stellar structure.  Select several values of $n$ below
    to overlay them on the same axes — first in dimensionless form, then rescaled to
    physical units using shared boundary conditions $\rho_c$ and $P_c$.
    """
)

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.header("Choose polytropic indices")

PRESETS = {
    "n = 0  (incompressible)": 0.01,
    "n = 1": 1.0,
    "n = 1.5  (non-rel. degenerate)": 1.5,
    "n = 2": 2.0,
    "n = 3  (Eddington / rel. degenerate)": 3.0,
    "n = 4": 4.0,
    "n = 4.9  (→ ∞ radius)": 4.9,
}

selected_labels = st.sidebar.multiselect(
    "Select n values to plot",
    options=list(PRESETS.keys()),
    default=["n = 1.5  (non-rel. degenerate)", "n = 3  (Eddington / rel. degenerate)"],
)

custom_n_str = st.sidebar.text_input(
    "Or add custom n values (comma-separated, 0 < n < 5)",
    value="",
    placeholder="e.g. 0.5, 2.5",
)

dx = st.sidebar.select_slider(
    "Integration step size dξ",
    options=[1e-5, 5e-5, 1e-4, 5e-4],
    value=1e-4,
    format_func=lambda v: f"{v:.0e}",
)

st.sidebar.markdown("---")
st.sidebar.header("Physical boundary conditions")
rhoc = st.sidebar.number_input(
    "Central density ρ_c  [kg/m³]",
    min_value=1e3, max_value=1e8, value=2e5, format="%.2e",
)
Pc = st.sidebar.number_input(
    "Central pressure P_c  [J/m³]",
    min_value=1e13, max_value=1e20, value=4e16, format="%.2e",
)

# ── Build list of n values ─────────────────────────────────────────────────────
n_values = [PRESETS[lbl] for lbl in selected_labels]

if custom_n_str.strip():
    for tok in custom_n_str.split(","):
        try:
            v = float(tok.strip())
            if 0 < v < 5:
                n_values.append(v)
        except ValueError:
            pass

n_values = sorted(set(round(v, 4) for v in n_values))

if not n_values:
    st.warning("Please select at least one value of n from the sidebar.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# Lane-Emden solver
# ══════════════════════════════════════════════════════════════════════════════
def solve_lane_emden(n, dxi):
    xi, th, phi = 0.0, 1.0, 0.0
    xi_list, th_list, phi_list = [xi], [th], [phi]
    dxi_plot = 2e-2
    xi_1 = None

    while th > 0:
        dth  = phi
        dphi = -1.0/3.0 if xi < dxi else -th**n - (2.0 / xi) * phi

        th_new  = th  + dxi * dth
        phi_new = phi + dxi * dphi
        xi_new  = xi  + dxi

        if th_new <= 0:
            xi_1 = xi - th * (dxi / (th_new - th))
            phi_surface = phi
            break

        th, phi, xi = th_new, phi_new, xi_new
        if abs(xi % dxi_plot) < dxi:
            xi_list.append(xi); th_list.append(th); phi_list.append(phi)

    if xi_1 is None:
        xi_1 = xi
        phi_surface = phi

    xi_list.append(xi_1); th_list.append(0.0); phi_list.append(phi_surface)
    return (np.array(xi_list), np.array(th_list), np.array(phi_list),
            xi_1, phi_surface)

# ══════════════════════════════════════════════════════════════════════════════
# Solve all and compute physical scales
# ══════════════════════════════════════════════════════════════════════════════
G_N  = 6.674e-11
MSun = 1.989e30
RSun = 6.9634e8

solutions = {}
for n in n_values:
    gamma = 1.0 + 1.0/n
    K     = Pc / rhoc**gamma
    alpha = (np.sqrt((n + 1) * K / (4 * np.pi * G_N))
             * rhoc**((1 - n) / (2 * n)))
    xi_arr, th_arr, phi_arr, xi_1, phi_s = solve_lane_emden(n, dx)
    xi2dphi = -xi_1**2 * phi_s
    R_star  = alpha * xi_1 / RSun
    M_star  = 4 * np.pi * alpha**3 * rhoc * xi2dphi / MSun
    solutions[n] = dict(
        xi=xi_arr, th=th_arr, phi=phi_arr,
        xi_1=xi_1, phi_s=phi_s, xi2dphi=xi2dphi,
        alpha=alpha, K=K, gamma=gamma,
        R=R_star, M=M_star,
    )

# ══════════════════════════════════════════════════════════════════════════════
# Colours
# ══════════════════════════════════════════════════════════════════════════════
colours = [
    '#4fc3f7', '#ff7043', '#66bb6a', '#ffa726',
    '#ab47bc', '#26c6da', '#d4e157', '#ef5350',
]
colour_map = {n: colours[i % len(colours)] for i, n in enumerate(n_values)}

def style_ax(ax):
    ax.set_facecolor('#0e1117')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for sp in ax.spines.values(): sp.set_edgecolor('#444')

plt.rcParams.update({'font.size': 13})

# ══════════════════════════════════════════════════════════════════════════════
# Lane–Emden equation reminder
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("The Lane–Emden equation", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.latex(
            r"\frac{1}{\xi^2}\frac{d}{d\xi}\!\left(\xi^2\frac{d\theta}{d\xi}\right)"
            r"= -\theta^n"
        )
        st.markdown(r"$\theta(0)=1,\quad \theta'(0)=0,\quad$ surface at $\theta(\xi_1)=0$")
    with col2:
        st.markdown(
            r"""
            Physical mapping (set by $\rho_c$, $P_c$ in the sidebar):
            $$r = \alpha\,\xi, \qquad \rho = \rho_c\,\theta^n$$
            $$\alpha = \sqrt{\frac{(n+1)K}{4\pi G}}\;\rho_c^{(1-n)/(2n)}$$
            """
        )

# ══════════════════════════════════════════════════════════════════════════════
# Summary table
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Summary")
rows = []
for n, s in solutions.items():
    rows.append({
        "n": f"{n:.2f}",
        "γ": f"{s['gamma']:.3f}",
        "ξ₁ (surface)": f"{s['xi_1']:.3f}",
        "−ξ²dθ/dξ|surface": f"{s['xi2dphi']:.4f}",
        "α  [m]": f"{s['alpha']:.3e}",
        "R  [R☉]": f"{s['R']:.3f}",
        "M  [M☉]": f"{s['M']:.3f}",
    })

import pandas as pd
df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 — Dimensionless: θ(ξ)
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Dimensionless Lane–Emden profiles  θ(ξ)")
st.caption(
    "Pure mathematics — no physical units.  "
    "The horizontal extent is ξ₁, which depends only on n.  "
    "Changing ρ_c or P_c in the sidebar has no effect here."
)

fig1, ax1 = plt.subplots(figsize=(9, 4.5))
fig1.patch.set_facecolor('#0e1117')
style_ax(ax1)

xi_max = max(s['xi_1'] for s in solutions.values())
for n, s in solutions.items():
    c = colour_map[n]
    ax1.plot(s['xi'], s['th'], color=c, linewidth=2,
             label=rf"$n={n:.2f}$,  $\xi_1={s['xi_1']:.2f}$")
    ax1.axvline(s['xi_1'], color=c, linestyle=':', linewidth=1, alpha=0.5)

ax1.axhline(0, color='#555', linewidth=0.8)
ax1.set_xlabel(r'$\xi$  (dimensionless radius)')
ax1.set_ylabel(r'$\theta(\xi)$')
ax1.set_title('Lane–Emden solutions for selected n')
ax1.set_xlim(0, xi_max * 1.08)
ax1.set_ylim(-0.05, 1.05)
ax1.legend(facecolor='#1a1a2e', labelcolor='white', loc='upper right')
plt.tight_layout()
st.pyplot(fig1)

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 — Physical: ρ(r)
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Physical density profiles  ρ(r)")
st.caption(
    "The same curves as above, with axes rescaled by α and ρ_c.  "
    "Adjust ρ_c or P_c in the sidebar to see how the physical scale changes "
    "without altering the shape."
)

fig2, ax2 = plt.subplots(figsize=(9, 4.5))
fig2.patch.set_facecolor('#0e1117')
style_ax(ax2)

for n, s in solutions.items():
    c = colour_map[n]
    rho_phys = rhoc * np.maximum(s['th'], 0)**n
    r_phys   = s['alpha'] * s['xi'] / RSun
    ax2.plot(r_phys, rho_phys, color=c, linewidth=2,
             label=rf"$n={n:.2f}$,  $R={s['R']:.2f}\,R_\odot$,  $M={s['M']:.2f}\,M_\odot$")
    ax2.axvline(s['R'], color=c, linestyle=':', linewidth=1, alpha=0.5)

ax2.axhline(0, color='#555', linewidth=0.8)
ax2.set_xlabel(r'$r = \alpha\,\xi$  [$R_{\odot}$]')
ax2.set_ylabel(r'$\rho = \rho_c\,\theta^n$  [kg m$^{-3}$]')
ax2.set_title(rf'Density profiles  ($\rho_c = {rhoc:.1e}$ kg/m³,  $P_c = {Pc:.1e}$ J/m³)')
ax2.set_xlim(0); ax2.set_ylim(0, rhoc * 1.25)
ax2.legend(facecolor='#1a1a2e', labelcolor='white', loc='upper right')
plt.tight_layout()
st.pyplot(fig2)

st.info(
    "**Notice:** larger $n$ → more centrally concentrated profile and (for these boundary "
    "conditions) larger radius.  The shape difference is visible in Plot 1; "
    "Plot 2 adds the physical scale set by $\\rho_c$ and $P_c$."
)
