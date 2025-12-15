"""
2_Parametric.py - Page pour les etudes parametriques
"""

import streamlit as st
from pathlib import Path
import sys
import numpy as np

APP_DIR = Path(__file__).parent.parent
PROJECT_DIR = APP_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from core.parameters import (
    SimulationConfig, PhysicalParameters, CVParameters,
    NumericalParameters, GeometryParameters, ParametricStudy
)

st.set_page_config(page_title="Parametric Study - CV Simulation", layout="wide")

st.title("📊 Etudes parametriques")
st.markdown("Explorez l'influence des parametres sur le voltammogramme.")


# === CONFIGURATION DE BASE ===
st.sidebar.header("Configuration de base")

with st.sidebar.expander("Parametres fixes", expanded=True):
    D_base = st.number_input("D (m²/s)", value=7.0e-9, format="%.2e", key="D_base")
    k0_base = st.number_input("k₀ (m/s)", value=1.0e-5, format="%.2e", key="k0_base")
    scan_rate_base = st.number_input("Scan rate (V/s)", value=0.1, key="sr_base")
    n_cycles_base = st.number_input("Cycles", value=2, min_value=1, key="nc_base")


# === SELECTION PARAMETRE A VARIER ===
st.subheader("Parametre a varier")

col1, col2 = st.columns(2)

with col1:
    param_choice = st.selectbox(
        "Parametre",
        options=[
            ("D - Coefficient de diffusion", "physical.D"),
            ("k₀ - Constante cinetique", "physical.k0"),
            ("α - Coefficient de transfert", "physical.alpha"),
            ("Scan rate", "cv.scan_rate"),
            ("Position WE (X)", "geometry.we_x"),
            ("c_bulk - Concentration", "physical.c_bulk"),
        ],
        format_func=lambda x: x[0]
    )
    param_name, param_path = param_choice

with col2:
    st.markdown("**Description:**")
    descriptions = {
        "physical.D": "Coefficient de diffusion des especes (m²/s). Affecte la forme du pic et le courant limite.",
        "physical.k0": "Constante cinetique heterogene (m/s). Determine la reversibilite du systeme.",
        "physical.alpha": "Coefficient de transfert. Affecte l'asymetrie des pics.",
        "cv.scan_rate": "Vitesse de balayage (V/s). Affecte l'amplitude des pics (∝ √v).",
        "geometry.we_x": "Position X de l'electrode de travail (m).",
        "physical.c_bulk": "Concentration bulk (mol/m³). Le courant est proportionnel a c_bulk.",
    }
    st.info(descriptions.get(param_path, ""))


# === DEFINITION DES VALEURS ===
st.subheader("Valeurs a tester")

col1, col2, col3 = st.columns(3)

# Valeurs par defaut selon le parametre
defaults = {
    "physical.D": (1e-9, 1e-8, 5),
    "physical.k0": (1e-6, 1e-3, 5),
    "physical.alpha": (0.3, 0.7, 5),
    "cv.scan_rate": (0.01, 0.5, 5),
    "geometry.we_x": (-4e-3, -1e-3, 4),
    "physical.c_bulk": (0.5, 5.0, 5),
}
default_min, default_max, default_n = defaults.get(param_path, (0.1, 1.0, 5))

with col1:
    val_min = st.number_input("Valeur min", value=default_min, format="%.2e")
with col2:
    val_max = st.number_input("Valeur max", value=default_max, format="%.2e")
with col3:
    n_values = st.number_input("Nombre de valeurs", value=default_n, min_value=2, max_value=20)

# Type d'echelle
scale_type = st.radio("Echelle", ["Lineaire", "Logarithmique"], horizontal=True)

if scale_type == "Lineaire":
    values = np.linspace(val_min, val_max, n_values)
else:
    values = np.logspace(np.log10(val_min), np.log10(val_max), n_values)

st.markdown("**Valeurs:**")
st.code(", ".join([f"{v:.2e}" for v in values]))


# === RESUME ETUDE ===
st.markdown("---")
st.subheader("Resume de l'etude")

col1, col2 = st.columns(2)

with col1:
    st.metric("Nombre de simulations", len(values))
    t_cycle = 20.0  # Approximation
    st.metric("Temps estime par simulation", f"~{t_cycle:.0f} s")
    st.metric("Temps total estime", f"~{len(values) * t_cycle / 60:.1f} min")

with col2:
    # Creer l'etude parametrique
    base_config = SimulationConfig(
        physical=PhysicalParameters(D=D_base, k0=k0_base),
        cv=CVParameters(scan_rate=scan_rate_base, n_cycles=n_cycles_base),
    )

    study = ParametricStudy(
        name=f"study_{param_path.split('.')[-1]}",
        base_config=base_config,
        parameter_path=param_path,
        values=values.tolist(),
    )

    st.markdown("**Configurations generees:**")
    configs = study.generate_configs()
    for i, cfg in enumerate(configs[:3]):
        st.text(f"  {cfg.name}: {param_path} = {values[i]:.2e}")
    if len(configs) > 3:
        st.text(f"  ... et {len(configs) - 3} autres")


# === BOUTON LANCEMENT ===
st.markdown("---")

import json

# Mode démo toggle
demo_mode = st.sidebar.toggle("🎭 Mode Démo", value=False,
                               help="Si activé, ne lance pas les simulations réelles")

col1, col2 = st.columns(2)

with col1:
    if st.button("💾 Sauvegarder configuration", use_container_width=True):
        # Sauvegarder la config de l'etude
        config_dir = PROJECT_DIR / "data" / "studies"
        config_dir.mkdir(parents=True, exist_ok=True)

        study_file = config_dir / f"{study.name}.json"
        with open(study_file, 'w') as f:
            json.dump({
                "name": study.name,
                "parameter_path": param_path,
                "values": values.tolist(),
                "base_config": base_config.to_dict(),
            }, f, indent=2)

        st.success(f"Etude sauvegardee: {study_file}")

with col2:
    if st.button("🚀 Lancer l'etude", type="primary", use_container_width=True):
        # D'abord sauvegarder la config
        config_dir = PROJECT_DIR / "data" / "studies"
        config_dir.mkdir(parents=True, exist_ok=True)

        study_file = config_dir / f"{study.name}.json"
        with open(study_file, 'w') as f:
            json.dump({
                "name": study.name,
                "parameter_path": param_path,
                "values": values.tolist(),
                "base_config": base_config.to_dict(),
            }, f, indent=2)

        if demo_mode:
            st.info("🎭 Mode Démo activé - Simulation non lancée")
            st.code(f"""
# Commande pour lancer manuellement:
cd "{PROJECT_DIR}"
start_fire
python core/run_parametric.py --study "{study_file}"
            """, language="bash")
        else:
            # Créer un fichier log pour l'étude
            log_file = config_dir / f"{study.name}.log"

            # Lancer via subprocess avec environnement Firedrake
            import subprocess

            cmd = f'''source ~/firedrake-native/firedrake-env/bin/activate && \\
                      export PETSC_DIR="$HOME/firedrake-native/petsc" && \\
                      export PETSC_ARCH="arch-firedrake-default" && \\
                      cd "{PROJECT_DIR}" && \\
                      python core/run_parametric.py --study "{study_file}"'''

            process = subprocess.Popen(
                cmd,
                shell=True,
                executable='/bin/bash',
                stdout=open(log_file, 'w'),
                stderr=subprocess.STDOUT,
                start_new_session=True
            )

            st.success(f"""
            **Étude paramétrique lancée!**

            - PID: {process.pid}
            - Log: `{log_file}`
            - Simulations: {len(values)}

            Les résultats apparaîtront dans `data/results/` au fur et à mesure.
            """)

            # Afficher le lien vers le log
            st.markdown("**Suivre la progression:**")
            st.code(f"tail -f {log_file}", language="bash")


# === RESULTATS EXISTANTS ===
st.markdown("---")
st.subheader("Resultats d'etudes precedentes")

studies_dir = PROJECT_DIR / "data" / "studies"
if studies_dir.exists():
    study_files = list(studies_dir.glob("*.json"))
    if study_files:
        selected_study = st.selectbox("Etude", study_files, format_func=lambda x: x.stem)
        if selected_study:
            import json
            with open(selected_study) as f:
                study_data = json.load(f)
            st.json(study_data)
    else:
        st.info("Aucune etude sauvegardee.")
else:
    st.info("Dossier d'etudes non trouve.")
