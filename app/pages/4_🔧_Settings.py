"""
4_Settings.py - Page de configuration
"""

import streamlit as st
from pathlib import Path
import sys
import json

APP_DIR = Path(__file__).parent.parent
PROJECT_DIR = APP_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

st.set_page_config(page_title="Settings - CV Simulation", layout="wide")

st.title("🔧 Configuration")


# === CHEMINS ===
st.subheader("📁 Chemins")

col1, col2 = st.columns(2)

with col1:
    st.text_input("Dossier projet", value=str(PROJECT_DIR), disabled=True)
    st.text_input("Dossier resultats", value=str(PROJECT_DIR / "data" / "results"))
    st.text_input("Dossier maillages", value=str(PROJECT_DIR / "data" / "meshes"))

with col2:
    # Verifier les chemins
    results_ok = (PROJECT_DIR / "data" / "results").exists()
    meshes_ok = (PROJECT_DIR / "data" / "meshes").exists()
    mesh_default = (PROJECT_DIR / "data" / "meshes" / "electrode_comsol.msh").exists()

    st.markdown("**Status:**")
    st.markdown(f"- Resultats: {'✅' if results_ok else '❌'}")
    st.markdown(f"- Maillages: {'✅' if meshes_ok else '❌'}")
    st.markdown(f"- Maillage defaut: {'✅' if mesh_default else '❌'}")


# === FIREDRAKE ===
st.markdown("---")
st.subheader("🔥 Firedrake")

st.markdown("""
Pour lancer les simulations, Firedrake doit etre active:

```bash
# Activer l'environnement Firedrake
source ~/firedrake/firedrake-env/bin/activate

# Ou utiliser l'alias (si configure)
start_fire

# Verifier l'installation
python -c "import firedrake; print('Firedrake OK')"
python -c "import echemfem; print('EchemFEM OK')"
```
""")

# Test Firedrake (dans un subprocess)
if st.button("🧪 Tester Firedrake"):
    import subprocess
    try:
        result = subprocess.run(
            ["bash", "-c", "source ~/firedrake/firedrake-env/bin/activate && python -c 'import firedrake; print(firedrake.__version__)'"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            st.success(f"Firedrake disponible: {result.stdout.strip()}")
        else:
            st.error(f"Erreur Firedrake: {result.stderr}")
    except subprocess.TimeoutExpired:
        st.error("Timeout - Firedrake ne repond pas")
    except Exception as e:
        st.error(f"Erreur: {e}")


# === PARAMETRES PAR DEFAUT ===
st.markdown("---")
st.subheader("⚙️ Parametres par defaut")

config_file = PROJECT_DIR / "config" / "default_params.yaml"

# Charger ou creer config
default_config = {
    "physical": {
        "D": 7.0e-9,
        "k0": 1.0e-5,
        "alpha": 0.5,
        "E0": 0.36,
        "c_bulk": 1.0,
    },
    "cv": {
        "E_start": 0.36,
        "E_vertex1": 0.86,
        "E_vertex2": -0.14,
        "scan_rate": 0.1,
        "n_cycles": 2,
    },
    "numerical": {
        "dt": 0.005,
        "vtk_interval": 50,
    },
}

# Editeur JSON
st.json(default_config)

col1, col2 = st.columns(2)

with col1:
    if st.button("💾 Sauvegarder config par defaut"):
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            import yaml
            yaml.dump(default_config, f, default_flow_style=False)
        st.success(f"Config sauvegardee: {config_file}")

with col2:
    if st.button("🔄 Reinitialiser"):
        st.info("Config reinitialiser aux valeurs par defaut")


# === THEME ===
st.markdown("---")
st.subheader("🎨 Theme")

theme = st.selectbox("Theme graphiques", ["Default", "Dark", "Light", "Seaborn"])

st.info("Le theme sera applique aux prochains graphiques generes.")


# === INFORMATIONS SYSTEME ===
st.markdown("---")
st.subheader("💻 Systeme")

import platform

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"**OS:** {platform.system()} {platform.release()}")
    st.markdown(f"**Python:** {platform.python_version()}")
    st.markdown(f"**Architecture:** {platform.machine()}")

with col2:
    # Versions des packages
    try:
        import numpy
        st.markdown(f"**NumPy:** {numpy.__version__}")
    except:
        pass

    try:
        import pandas
        st.markdown(f"**Pandas:** {pandas.__version__}")
    except:
        pass

    try:
        import plotly
        st.markdown(f"**Plotly:** {plotly.__version__}")
    except:
        pass


# === LOGS ===
st.markdown("---")
st.subheader("📜 Logs")

log_dir = PROJECT_DIR / "logs"
if log_dir.exists():
    log_files = list(log_dir.glob("*.log"))
    if log_files:
        selected_log = st.selectbox("Fichier log", log_files)
        if selected_log:
            with open(selected_log) as f:
                content = f.read()
            st.code(content[-5000:], language="text")  # Derniers 5000 chars
else:
    st.info("Aucun fichier log disponible.")
