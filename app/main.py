"""
main.py - Point d'entree de l'application Streamlit CV Simulation
"""

import streamlit as st
from pathlib import Path
import sys

# Ajouter le chemin parent pour les imports
APP_DIR = Path(__file__).parent
PROJECT_DIR = APP_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

# Configuration de la page
st.set_page_config(
    page_title="CV Simulation - Firedrake",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Style CSS personnalise
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .status-running {
        color: #FFA726;
        font-weight: bold;
    }
    .status-completed {
        color: #66BB6A;
        font-weight: bold;
    }
    .status-failed {
        color: #EF5350;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<p class="main-header">Cyclic Voltammetry Simulation</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Cyclic_voltammetry_physical_picture.svg/220px-Cyclic_voltammetry_physical_picture.svg.png", width=200)
        st.markdown("### Navigation")
        st.markdown("""
        - **Single Run**: Lancer une simulation
        - **Parametric**: Etudes parametriques
        - **Results**: Visualiser les resultats
        - **Settings**: Configuration
        """)

        st.markdown("---")
        st.markdown("### Systeme")
        st.markdown("**Fe(CN)₆³⁻/Fe(CN)₆⁴⁻**")
        st.markdown("Ferri/Ferrocyanide")

        st.markdown("---")
        st.markdown("### Liens")
        st.markdown("[Firedrake](https://firedrakeproject.org)")
        st.markdown("[EchemFEM](https://github.com/LLNL/echemfem)")

    # Contenu principal
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🔬 Simulation")
        st.markdown("""
        Lancez des simulations de voltammetrie cyclique
        avec des parametres personnalises.
        """)
        if st.button("Nouvelle simulation", type="primary"):
            st.switch_page("pages/1_🔬_Single_Run.py")

    with col2:
        st.markdown("### 📊 Etudes parametriques")
        st.markdown("""
        Explorez l'influence des parametres:
        D, k₀, scan_rate, geometrie...
        """)
        if st.button("Etude parametrique"):
            st.switch_page("pages/2_📊_Parametric.py")

    with col3:
        st.markdown("### 📈 Resultats")
        st.markdown("""
        Visualisez et comparez les voltammogrammes,
        exportez les donnees.
        """)
        if st.button("Voir resultats"):
            st.switch_page("pages/3_📈_Results.py")

    st.markdown("---")

    # Resume des derniers runs
    st.markdown("### Derniers runs")

    results_dir = PROJECT_DIR / "data" / "results"
    if results_dir.exists():
        runs = sorted(results_dir.glob("*/"), reverse=True)[:5]
        if runs:
            for run_dir in runs:
                metrics_file = run_dir / "metrics.txt"
                if metrics_file.exists():
                    with open(metrics_file) as f:
                        content = f.read()
                    with st.expander(f"📁 {run_dir.name}"):
                        st.text(content)
        else:
            st.info("Aucun resultat disponible. Lancez une simulation!")
    else:
        st.info("Dossier de resultats non trouve.")


if __name__ == "__main__":
    main()
