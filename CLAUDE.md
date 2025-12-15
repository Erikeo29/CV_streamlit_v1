# CLAUDE.md

Instructions pour Claude Code dans ce projet.

## Project Overview

Interface Streamlit pour simulations CV (voltammétrie cyclique) avec Firedrake/EchemFEM.
Système: Fe(CN)₆³⁻/Fe(CN)₆⁴⁻

## Commands

```bash
# Lancer l'interface Streamlit
streamlit run app/main.py

# Activer Firedrake (pour simulations)
start_fire

# Lancer une simulation directement
python core/run_simulation.py --config data/results/001/config.json

# Générer un maillage
python core/mesh_generator.py --we-x -2.5 --view
```

## Architecture

```
app/                    # Interface Streamlit
├── main.py            # Page d'accueil
└── pages/             # Pages de l'app
    ├── 1_🔬_Single_Run.py
    ├── 2_📊_Parametric.py
    ├── 3_📈_Results.py
    └── 4_🔧_Settings.py

core/                   # Logique simulation
├── simulation.py      # Classe CVSimulationFull
├── mesh_generator.py  # Génération maillage Gmsh
├── parameters.py      # Dataclasses config
└── worker.py          # Exécution background

data/
├── meshes/            # Fichiers .msh
└── results/           # Résultats (NNN/)
```

## Key Files

- `core/parameters.py`: Dataclasses pour tous les paramètres
- `core/worker.py`: Gestion des jobs de simulation
- `app/pages/3_📈_Results.py`: Visualisation PyVista/Plotly

## Physical Parameters

| Param | Default | Description |
|-------|---------|-------------|
| D | 7.0e-9 m²/s | Diffusion coefficient |
| k₀ | 1.0e-5 m/s | Rate constant |
| α | 0.5 | Transfer coefficient |
| E°' | 0.36 V | Formal potential |
| c_bulk | 1.0 mol/m³ | Bulk concentration |

## DO

- Utiliser les dataclasses de `parameters.py`
- Stocker résultats dans `data/results/NNN/`
- Exporter en JSON/CSV
- Garder l'interface responsive (async pour simulations)

## DO NOT

- Lancer Firedrake dans le même process que Streamlit
- Créer de fichiers à la racine du projet
- Modifier les scripts core/ sans tester
