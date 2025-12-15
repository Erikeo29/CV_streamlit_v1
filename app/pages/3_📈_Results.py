"""
3_Results.py - Page de visualisation des resultats
"""

import streamlit as st
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import json

APP_DIR = Path(__file__).parent.parent
PROJECT_DIR = APP_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

# === CONSTANTES PHYSIQUES ===
F = 96485.0  # C/mol (Faraday)
R = 8.314    # J/(mol·K)
T = 298.15   # K (25°C)


def generate_demo_cv_from_config(config: dict) -> tuple:
    """
    Génère un CV synthétique à partir d'une config JSON.
    Retourne (E_array, i_array, metrics).
    """
    # Extraire paramètres
    phys = config.get('physical', {})
    cv_params = config.get('cv', {})

    D = phys.get('D', 7.0e-9)
    k0 = phys.get('k0', 1.0e-5)
    alpha = phys.get('alpha', 0.5)
    E0 = phys.get('E0', 0.36)
    c_bulk = phys.get('c_bulk', 1.0)

    E_start = cv_params.get('E_start', 0.36)
    E_vertex1 = cv_params.get('E_vertex1', 0.86)
    E_vertex2 = cv_params.get('E_vertex2', -0.14)
    scan_rate = cv_params.get('scan_rate', 0.1)
    n_cycles = cv_params.get('n_cycles', 2)

    n = 1  # electrons
    f = F / (R * T)
    A = 1e-6  # 1 mm²

    # Générer le signal E(t)
    t1 = abs(E_vertex1 - E_start) / scan_rate
    t2 = abs(E_vertex1 - E_vertex2) / scan_rate
    t3 = abs(E_start - E_vertex2) / scan_rate
    t_cycle = t1 + t2 + t3

    t_array = np.linspace(0, n_cycles * t_cycle, 2000)
    E_array = []
    for t in t_array:
        t_mod = t % t_cycle
        if t_mod < t1:
            E = E_start + scan_rate * t_mod
        elif t_mod < t1 + t2:
            E = E_vertex1 - scan_rate * (t_mod - t1)
        else:
            E = E_vertex2 + scan_rate * (t_mod - t1 - t2)
        E_array.append(E)
    E_array = np.array(E_array)

    # Courant de pic théorique (Randles-Sevcik)
    i_p_rev = 0.4463 * n * F * A * c_bulk * np.sqrt(n * F * D * scan_rate / (R * T))

    # Générer courant avec Butler-Volmer simplifié
    current = []
    c_ox_surf = c_bulk
    c_red_surf = 0.0
    dt = abs(E_array[1] - E_array[0]) / scan_rate if len(E_array) > 1 else 0.01

    for i, E in enumerate(E_array):
        eta = E - E0

        i_f = n * F * A * k0 * c_ox_surf * np.exp(-alpha * n * f * eta)
        i_b = n * F * A * k0 * c_red_surf * np.exp((1 - alpha) * n * f * eta)
        i_BV = i_f - i_b

        if i > 0:
            t_elapsed = i * dt
            if t_elapsed > 0:
                i_diff_lim = n * F * A * D * c_bulk / np.sqrt(np.pi * D * t_elapsed)
                i_diff_lim = max(i_diff_lim, i_p_rev * 0.1)
            else:
                i_diff_lim = i_p_rev * 10
        else:
            i_diff_lim = i_p_rev * 10

        if abs(i_BV) > 0:
            i_net = i_BV * i_diff_lim / (abs(i_BV) + i_diff_lim)
        else:
            i_net = 0

        current.append(i_net)

        if i_net > 0:
            c_ox_surf = max(0, c_ox_surf - abs(i_net) * dt / (n * F * A * np.sqrt(D * dt + 1e-12)))
            c_red_surf = min(c_bulk, c_red_surf + abs(i_net) * dt / (n * F * A * np.sqrt(D * dt + 1e-12)))
        else:
            c_ox_surf = min(c_bulk, c_ox_surf + abs(i_net) * dt / (n * F * A * np.sqrt(D * dt + 1e-12)))
            c_red_surf = max(0, c_red_surf - abs(i_net) * dt / (n * F * A * np.sqrt(D * dt + 1e-12)))

    current = np.array(current)

    # Lisser
    from scipy.ndimage import gaussian_filter1d
    current = gaussian_filter1d(current, sigma=5)

    # Calculer métriques
    i_uA = current * 1e6
    Ipa = np.max(i_uA)
    Ipc = np.min(i_uA)
    Epa = E_array[np.argmax(i_uA)]
    Epc = E_array[np.argmin(i_uA)]
    dEp = abs(Epa - Epc) * 1000  # mV

    metrics = {
        'Ipa': Ipa,
        'Ipc': Ipc,
        'Epa': Epa,
        'Epc': Epc,
        '|Ipa/Ipc|': abs(Ipa / Ipc) if Ipc != 0 else 0,
        'ΔEp': dEp,
    }

    return E_array, current, metrics


st.set_page_config(page_title="Results - CV Simulation", layout="wide")

st.title("📈 Resultats")
st.markdown("Visualisez et comparez les voltammogrammes.")

# Mode démo toggle
demo_mode = st.sidebar.toggle("🎭 Mode Démo", value=False,
                               help="Génère des CV synthétiques à partir des configs")


# === SELECTION DES RUNS ===
results_dir = PROJECT_DIR / "data" / "results"

if not results_dir.exists():
    st.warning("Aucun resultat disponible. Lancez une simulation d'abord!")
    st.stop()

runs = sorted([d for d in results_dir.iterdir() if d.is_dir()], reverse=True)

if not runs:
    st.warning("Aucun run trouve.")
    st.stop()

# Multi-selection pour comparaison
selected_runs = st.multiselect(
    "Selectionner les runs a afficher",
    options=runs,
    format_func=lambda x: x.name,
    default=[runs[0]] if runs else []
)


# === CHARGEMENT DES DONNEES ===
def load_config(run_path: Path) -> dict:
    """Charge la config JSON d'un run."""
    config_path = run_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return None


def load_cv_data(run_path: Path, use_demo: bool = False) -> pd.DataFrame:
    """Charge les donnees CV d'un run."""
    csv_path = run_path / "cv_data.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df

    # Mode démo: générer à partir de la config
    if use_demo:
        config = load_config(run_path)
        if config:
            E_array, current, _ = generate_demo_cv_from_config(config)
            return pd.DataFrame({
                'E_V': E_array,
                'I_A': current
            })
    return None


def load_metrics(run_path: Path, use_demo: bool = False) -> dict:
    """Charge les metriques d'un run."""
    metrics_path = run_path / "metrics.txt"
    if metrics_path.exists():
        metrics = {}
        with open(metrics_path) as f:
            for line in f:
                if "=" in line:
                    # Parse "Ipa = 19.15 uA @ Epa = 0.521 V"
                    parts = line.split("=")
                    if len(parts) >= 2:
                        key = parts[0].strip().split()[-1]
                        val = parts[1].strip().split()[0]
                        try:
                            metrics[key] = float(val)
                        except:
                            pass
        return metrics

    # Mode démo: générer à partir de la config
    if use_demo:
        config = load_config(run_path)
        if config:
            _, _, metrics = generate_demo_cv_from_config(config)
            return metrics
    return {}


# === AFFICHAGE ===
if selected_runs:
    # Tabs pour differentes vues
    tab1, tab2, tab3 = st.tabs(["📊 Voltammogrammes", "📋 Metriques", "🗂️ Fichiers"])

    with tab1:
        st.subheader("Voltammogrammes")
        if demo_mode:
            st.caption("🎭 Mode Démo - CV synthétiques générés à partir des paramètres de config")

        # Charger les donnees
        all_data = {}
        for run_path in selected_runs:
            df = load_cv_data(run_path, use_demo=demo_mode)
            if df is not None:
                all_data[run_path.name] = df

        if all_data:
            # Construire le graphique
            import plotly.graph_objects as go

            fig = go.Figure()

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

            for i, (name, df) in enumerate(all_data.items()):
                color = colors[i % len(colors)]
                fig.add_trace(go.Scatter(
                    x=df['E_V'],
                    y=df['I_A'] * 1e6,  # Convertir en uA
                    mode='lines',
                    name=name,
                    line=dict(color=color, width=2),
                ))

            fig.update_layout(
                title="Cyclic Voltammogram - Fe(CN)₆³⁻/Fe(CN)₆⁴⁻",
                xaxis_title="Potential E (V vs Ref)",
                yaxis_title="Current I (μA)",
                hovermode='closest',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                height=500,
            )

            # Ligne zero
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

            # E0 reference
            fig.add_vline(x=0.36, line_dash="dot", line_color="green", opacity=0.5,
                         annotation_text="E°'=0.36V")

            st.plotly_chart(fig, use_container_width=True)

            # Export
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Exporter PNG"):
                    fig.write_image("cv_comparison.png")
                    st.success("Image exportee: cv_comparison.png")
            with col2:
                if st.button("📥 Exporter CSV"):
                    # Combiner les donnees
                    combined = pd.DataFrame()
                    for name, df in all_data.items():
                        df_copy = df.copy()
                        df_copy['run'] = name
                        combined = pd.concat([combined, df_copy])
                    combined.to_csv("cv_combined.csv", index=False)
                    st.success("Donnees exportees: cv_combined.csv")

        else:
            st.warning("Aucune donnee CV trouvee dans les runs selectionnes.")

    with tab2:
        st.subheader("Metriques")

        # Tableau comparatif
        metrics_data = []
        for run_path in selected_runs:
            metrics = load_metrics(run_path, use_demo=demo_mode)
            metrics['Run'] = run_path.name
            metrics_data.append(metrics)

        if metrics_data:
            df_metrics = pd.DataFrame(metrics_data)
            df_metrics = df_metrics.set_index('Run')

            # Renommer colonnes
            rename_map = {
                'Ipa': 'Ipa (μA)',
                'Ipc': 'Ipc (μA)',
            }
            df_metrics = df_metrics.rename(columns=rename_map)

            st.dataframe(df_metrics, use_container_width=True)

            # Graphiques des metriques
            col1, col2 = st.columns(2)

            with col1:
                if '|Ipa/Ipc|' in df_metrics.columns:
                    st.bar_chart(df_metrics['|Ipa/Ipc|'])
                    st.caption("Ratio |Ipa/Ipc| (ideal: 1.0)")

            with col2:
                if 'Ep' in df_metrics.columns:
                    st.bar_chart(df_metrics['Ep'])
                    st.caption("ΔEp (mV) (ideal Nernstien: 57 mV)")

    with tab3:
        st.subheader("Fichiers")

        for run_path in selected_runs:
            with st.expander(f"📁 {run_path.name}"):
                files = list(run_path.iterdir())
                for f in sorted(files):
                    if f.is_file():
                        size = f.stat().st_size
                        size_str = f"{size/1024:.1f} KB" if size > 1024 else f"{size} B"
                        st.text(f"  {f.name} ({size_str})")

                # Bouton ParaView
                pvd_path = run_path / "concentrations.pvd"
                if pvd_path.exists():
                    st.code(f"paraview {pvd_path}", language="bash")

else:
    st.info("Selectionnez au moins un run pour afficher les resultats.")


# === VISUALISATION 2D (Concentrations) ===
st.markdown("---")
st.subheader("🎨 Visualisation 2D - Concentrations")

if selected_runs:
    if demo_mode:
        st.info("🎭 Mode Démo: La visualisation 2D nécessite des fichiers VTU générés par Firedrake.")
    else:
        run_for_2d = st.selectbox("Run pour visualisation 2D", selected_runs,
                                  format_func=lambda x: x.name, key="run_2d")

        snapshots_dir = run_for_2d / "snapshots"
        vtu_dir = run_for_2d / "concentrations"
        index_file = snapshots_dir / "index.csv"

        # Option pour générer les images si pas encore fait
        if not snapshots_dir.exists() or not list(snapshots_dir.glob("*.png")):
            st.warning("Images non générées. Cliquez ci-dessous pour les créer (peut prendre quelques minutes).")

            col1, col2 = st.columns(2)
            with col1:
                field_choice = st.selectbox("Champ", ["ferro", "ferri"], key="field_gen")
            with col2:
                cmap_choice = st.selectbox("Colormap", ["coolwarm", "viridis", "plasma", "RdBu_r"], key="cmap_gen")

            if st.button("🖼️ Générer les images PNG", key="gen_png"):
                with st.spinner("Génération des images en cours..."):
                    try:
                        import sys
                        sys.path.insert(0, str(PROJECT_DIR / "core"))
                        from postprocess import generate_concentration_images

                        generated = generate_concentration_images(
                            run_for_2d, field=field_choice, cmap=cmap_choice
                        )
                        st.success(f"{len(generated)} images générées!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erreur: {e}")

        # Afficher les images si elles existent
        elif index_file.exists():
            df_index = pd.read_csv(index_file)

            # Filtrer les entrées avec potentiel valide
            df_valid = df_index.dropna(subset=['potential_V'])

            if not df_valid.empty:
                # Slider basé sur le potentiel
                potentials = df_valid['potential_V'].tolist()
                frames = df_valid['frame'].tolist()
                paths = df_valid['path'].tolist()

                # Trouver les potentiels caractéristiques pour le slider
                E_min = min(potentials)
                E_max = max(potentials)

                # Layout compact: slider + info sur une ligne
                col_slider, col_info = st.columns([3, 1])
                with col_slider:
                    selected_idx = st.slider(
                        f"Potentiel E (V) - [{E_min:.2f} V to {E_max:.2f} V]",
                        min_value=0,
                        max_value=len(potentials) - 1,
                        value=0,
                        format="%d",
                        key="pot_slider"
                    )

                selected_potential = potentials[selected_idx]
                selected_frame = frames[selected_idx]
                selected_path = paths[selected_idx]
                selected_time = df_valid.iloc[selected_idx]['time_s']

                with col_info:
                    st.markdown(f"**E = {selected_potential:.3f} V**")
                    st.caption(f"t = {selected_time:.2f} s")

                # Afficher l'image avec taille réduite (600px max)
                img_path = snapshots_dir / selected_path
                if img_path.exists():
                    st.image(str(img_path), width=600)
                else:
                    st.error(f"Image non trouvée: {img_path}")

                # Bouton pour animation
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    gif_path = run_for_2d / "ferro_animation.gif"
                    if gif_path.exists():
                        st.markdown("**Animation disponible:**")
                        st.image(str(gif_path))
                    else:
                        if st.button("🎬 Créer animation GIF"):
                            with st.spinner("Création de l'animation..."):
                                try:
                                    from postprocess import create_animation
                                    create_animation(run_for_2d, field="ferro", fps=15)
                                    st.success("Animation créée!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Erreur: {e}")

                with col2:
                    # Regénérer les images
                    if st.button("🔄 Regénérer les images"):
                        import shutil
                        shutil.rmtree(snapshots_dir)
                        st.rerun()
            else:
                st.warning("Index des images invalide.")
        else:
            # Images existent mais pas d'index
            png_files = sorted(snapshots_dir.glob("ferro_*.png"),
                               key=lambda x: int(x.stem.split("_")[-1]))
            if png_files:
                st.caption(f"{len(png_files)} images trouvées (sans index de potentiel)")
                selected_png = st.select_slider(
                    "Frame",
                    options=png_files,
                    format_func=lambda x: x.stem.split("_")[-1]
                )
                st.image(str(selected_png), use_container_width=True)
            else:
                st.info("Aucune image trouvée.")

        # Lien ParaView
        pvd_path = run_for_2d / "concentrations.pvd"
        if pvd_path.exists():
            st.markdown("---")
            st.markdown("**Visualisation avancée avec ParaView:**")
            st.code(f"paraview {pvd_path}", language="bash")
