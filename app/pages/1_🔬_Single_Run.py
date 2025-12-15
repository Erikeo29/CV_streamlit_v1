"""
1_Single_Run.py - Page pour lancer une simulation unique
"""

import streamlit as st
from pathlib import Path
import sys
import json
import time
import numpy as np
import pandas as pd

# Setup paths
APP_DIR = Path(__file__).parent.parent
PROJECT_DIR = APP_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from core.parameters import (
    SimulationConfig, PhysicalParameters, CVParameters,
    NumericalParameters, GeometryParameters
)

# === CONSTANTES PHYSIQUES ===
F = 96485.0  # C/mol (Faraday)
R = 8.314    # J/(mol·K)
T = 298.15   # K (25°C)


def generate_demo_cv(E_array, D, k0, alpha, E0, c_bulk, scan_rate, n_electrons=1):
    """
    Génère un CV synthétique réaliste.

    Utilise une fonction de forme analytique basée sur la théorie
    de Nicholson-Shain pour systèmes quasi-réversibles.

    Le paramètre Λ = k0/√(nFDv/RT) contrôle la réversibilité:
    - Λ > 15: réversible (ΔEp ≈ 57/n mV)
    - Λ < 0.01: irréversible (grands ΔEp)
    """
    n = n_electrons
    f = n * F / (R * T)  # ~38.9 V^-1 à 25°C

    # Aire électrode (m²)
    A = 1e-6  # 1 mm²

    # Paramètre de Matsuda-Ayabe (réversibilité)
    Lambda = k0 / np.sqrt(D * n * F * scan_rate / (R * T))

    # Courant de pic Randles-Sevcik (système réversible)
    i_p = 0.4463 * n * F * A * c_bulk * np.sqrt(n * F * D * scan_rate / (R * T))

    # Décalage du pic selon Lambda (courbe de Nicholson)
    # Pour Lambda petit: pics plus éloignés de E0
    if Lambda > 10:
        delta_Ep = 0.057 / n  # ~57 mV pour réversible
    else:
        # Approximation empirique de la courbe de Nicholson
        delta_Ep = 0.057 / n + 0.029 / n * np.log10(1 / Lambda)
        delta_Ep = min(delta_Ep, 0.4)  # Limiter à 400 mV max

    # Positions des pics
    Ep_a = E0 + delta_Ep / 2  # Pic anodique
    Ep_c = E0 - delta_Ep / 2  # Pic cathodique

    # Largeur des pics (dépend de alpha et Lambda)
    width_factor = 0.05 + 0.02 * (1 - alpha) + 0.03 / (Lambda + 0.1)

    current = np.zeros(len(E_array))

    # Déterminer la direction de balayage pour chaque point
    # en utilisant une fenêtre glissante pour éviter les oscillations
    sweep_direction = np.zeros(len(E_array))
    window = 5  # points pour moyenner la direction

    for i in range(len(E_array)):
        if i < window:
            sweep_direction[i] = np.sign(E_array[min(i + window, len(E_array)-1)] - E_array[0])
        elif i >= len(E_array) - window:
            sweep_direction[i] = np.sign(E_array[-1] - E_array[max(i - window, 0)])
        else:
            sweep_direction[i] = np.sign(E_array[i + window] - E_array[i - window])

    # Lisser la direction pour éviter les transitions brusques
    from scipy.ndimage import median_filter
    sweep_direction = median_filter(sweep_direction, size=11)

    for i, E in enumerate(E_array):
        # Fonction de forme pour le pic (gaussienne)
        peak_a = np.exp(-(E - Ep_a)**2 / (2 * width_factor**2))
        peak_c = np.exp(-(E - Ep_c)**2 / (2 * width_factor**2))

        direction = sweep_direction[i]

        # Courant capacitif de base
        i_cap = 0.02 * i_p * direction

        # Contribution faradique selon la direction
        if direction > 0:  # Balayage anodique (vers potentiels positifs)
            i_farad = i_p * peak_a
            # Effet de déplétion après le pic anodique
            if E > Ep_a:
                i_farad *= np.exp(-0.8 * (E - Ep_a) / width_factor)
        else:  # Balayage cathodique (vers potentiels négatifs)
            i_farad = -i_p * peak_c
            # Effet de déplétion après le pic cathodique
            if E < Ep_c:
                i_farad *= np.exp(-0.8 * (Ep_c - E) / width_factor)

        current[i] = (i_farad + i_cap) * 1e6  # en µA

    # Ajouter un peu de décroissance Cottrell après les pics
    # pour simuler la diffusion
    from scipy.ndimage import gaussian_filter1d

    # Lissage léger
    current = gaussian_filter1d(current, sigma=3)

    return current


st.set_page_config(page_title="Single Run - CV Simulation", layout="wide")

st.title("🔬 Simulation unique")
st.markdown("Configurez et lancez une simulation de voltammetrie cyclique.")


# === FORMULAIRE PARAMETRES ===
with st.sidebar:
    st.header("Parametres")

    # Mode demo toggle
    demo_mode = st.toggle("🎭 Mode Démo", value=False,
                          help="Génère un CV synthétique sans Firedrake (test UI)")

    # Parametres physiques
    with st.expander("⚛️ Physique", expanded=True):
        D = st.number_input("D (m²/s)", value=7.0e-9, format="%.2e",
                           help="Coefficient de diffusion")
        k0 = st.number_input("k₀ (m/s)", value=1.0e-5, format="%.2e",
                            help="Constante cinetique heterogene")
        alpha = st.slider("α", 0.0, 1.0, 0.5,
                         help="Coefficient de transfert")
        E0 = st.number_input("E°' (V)", value=0.36,
                            help="Potentiel formel")
        c_bulk = st.number_input("c_bulk (mol/m³)", value=1.0,
                                help="Concentration bulk")

    # Parametres CV
    with st.expander("📈 Signal CV", expanded=True):
        E_start = st.number_input("E_start (V)", value=0.36)
        E_vertex1 = st.number_input("E_vertex1 (V)", value=0.86,
                                   help="Vertex anodique")
        E_vertex2 = st.number_input("E_vertex2 (V)", value=-0.14,
                                   help="Vertex cathodique")
        scan_rate = st.number_input("Scan rate (V/s)", value=0.1)
        n_cycles = st.number_input("Cycles", value=2, min_value=1, max_value=10)

    # Parametres numeriques
    with st.expander("🔢 Numerique"):
        dt = st.number_input("dt (s)", value=0.005, format="%.4f",
                            help="Pas de temps")
        vtk_interval = st.number_input("VTK interval", value=50, min_value=1,
                                       help="Intervalle export VTK")

    # Geometrie
    with st.expander("📐 Geometrie"):
        we_x_mm = st.number_input("WE position X (mm)", value=-2.5,
                                  help="Position centre WE")


# === ZONE PRINCIPALE ===
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Configuration")

    # Construire la config
    config = SimulationConfig(
        name=st.text_input("Nom de la simulation", value="run_001"),
        physical=PhysicalParameters(D=D, k0=k0, alpha=alpha, E0=E0, c_bulk=c_bulk),
        cv=CVParameters(E_start=E_start, E_vertex1=E_vertex1, E_vertex2=E_vertex2,
                       scan_rate=scan_rate, n_cycles=n_cycles),
        numerical=NumericalParameters(dt=dt, vtk_interval=vtk_interval),
        geometry=GeometryParameters(we_x=we_x_mm * 1e-3),
    )

    # Afficher resume
    st.json(config.to_dict())

with col2:
    st.subheader("Actions")

    # Estimation temps
    t_cycle = (abs(E_vertex1 - E_start) + abs(E_vertex1 - E_vertex2) +
               abs(E_start - E_vertex2)) / scan_rate
    t_total = n_cycles * t_cycle
    n_steps = int(t_total / dt)

    st.metric("Duree simulation", f"{t_total:.1f} s")
    st.metric("Nombre de pas", f"{n_steps:,}")
    st.metric("Fichiers VTK", f"{n_steps // vtk_interval}")

    st.markdown("---")

    # Bouton lancement
    if st.button("🚀 Lancer la simulation", type="primary", use_container_width=True):
        # Sauvegarder config
        results_dir = PROJECT_DIR / "data" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Trouver prochain numero
        existing = list(results_dir.glob("[0-9][0-9][0-9]"))
        next_num = max([int(d.name) for d in existing], default=0) + 1
        run_dir = results_dir / f"{next_num:03d}"
        run_dir.mkdir(exist_ok=True)

        config.output_dir = str(run_dir)
        config.mesh_path = str(PROJECT_DIR / "data" / "meshes" / "electrode_wells.msh")

        config_path = run_dir / "config.json"
        config.save(config_path)

        st.success(f"Configuration sauvegardee: {config_path}")

        if demo_mode:
            st.info("Mode Démo activé - simulation non lancée")
        else:
            # Lancer la simulation en subprocess
            import subprocess

            st.info("🔥 Lancement de la simulation Firedrake...")

            # Commande pour lancer avec Firedrake
            cmd = f'''source ~/firedrake-native/firedrake-env/bin/activate && \
                      export PETSC_DIR="$HOME/firedrake-native/petsc" && \
                      export PETSC_ARCH="arch-firedrake-default" && \
                      cd "{PROJECT_DIR}" && \
                      python core/run_simulation.py --config "{config_path}"'''

            # Afficher la commande
            with st.expander("Commande exécutée"):
                st.code(cmd, language="bash")

            # Lancer en arrière-plan
            log_file = run_dir / "simulation.log"

            try:
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    executable='/bin/bash',
                    stdout=open(log_file, 'w'),
                    stderr=subprocess.STDOUT,
                    start_new_session=True
                )

                st.success(f"✅ Simulation lancée (PID: {process.pid})")
                st.info(f"📁 Log: {log_file}")
                st.info(f"📊 Résultats dans: {run_dir}")

                # Sauvegarder le PID
                with open(run_dir / "pid.txt", 'w') as f:
                    f.write(str(process.pid))

                st.warning("⏳ La simulation tourne en arrière-plan (~25-30 min). Rafraîchissez la page Results pour voir les résultats.")

            except Exception as e:
                st.error(f"Erreur lancement: {e}")
                st.info("""
                **Lancement manuel:**
                ```bash
                start_fire
                cd "{}"
                python core/run_simulation.py --config {}
                ```
                """.format(PROJECT_DIR, config_path))


# === VISUALISATION SIGNAL ===
st.markdown("---")
st.subheader("Aperçu du signal CV")

# Generer signal E(t)
t1 = abs(E_vertex1 - E_start) / scan_rate
t2 = abs(E_vertex1 - E_vertex2) / scan_rate
t3 = abs(E_start - E_vertex2) / scan_rate
t_cycle = t1 + t2 + t3

t_plot = np.linspace(0, n_cycles * t_cycle, 1000)
E_plot = []
for t in t_plot:
    t_mod = t % t_cycle
    if t_mod < t1:
        E = E_start + scan_rate * t_mod
    elif t_mod < t1 + t2:
        E = E_vertex1 - scan_rate * (t_mod - t1)
    else:
        E = E_vertex2 + scan_rate * (t_mod - t1 - t2)
    E_plot.append(E)

df_signal = pd.DataFrame({"t (s)": t_plot, "E (V)": E_plot})

col1, col2 = st.columns(2)

with col1:
    st.line_chart(df_signal, x="t (s)", y="E (V)")
    st.caption("Potentiel vs Temps")

with col2:
    if demo_mode:
        # Générer le voltammogramme démo
        E_array = np.array(E_plot)
        i_demo = generate_demo_cv(E_array, D, k0, alpha, E0, c_bulk, scan_rate)

        # Utiliser Plotly pour préserver l'ordre des points (st.line_chart trie par X!)
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=E_array, y=i_demo, mode='lines', name='CV'))
        fig.update_layout(
            xaxis_title="E (V)",
            yaxis_title="i (µA)",
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=E0, line_dash="dot", line_color="green", opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Voltammogramme CV (Démo)")
    else:
        st.info("Le voltammogramme sera affiché après la simulation.")
