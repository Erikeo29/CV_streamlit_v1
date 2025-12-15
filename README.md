# CV Simulation - Streamlit Interface

Interface web pour la simulation de voltammétrie cyclique du système Fe(CN)₆³⁻/Fe(CN)₆⁴⁻ avec Firedrake/EchemFEM.

## Installation

### 1. Environnement Python (pour Streamlit)

```bash
# Créer un environnement conda
conda create -n cv-streamlit python=3.10
conda activate cv-streamlit

# Installer les dépendances
pip install -r requirements.txt
```

### 2. Firedrake (pour les simulations)

Firedrake doit être installé séparément. Voir [firedrakeproject.org](https://firedrakeproject.org).

```bash
# Activer Firedrake
start_fire  # ou source ~/firedrake/firedrake-env/bin/activate
```

## Utilisation

### Lancer l'interface Streamlit

```bash
cd "/home/erikeo29/15_R&D_CV/06_CV (Firedrake)/03_CV (param & streamlit)"
streamlit run app/main.py
```

L'interface sera disponible sur http://localhost:8501

### Structure des pages

| Page | Description |
|------|-------------|
| **Single Run** | Lancer une simulation avec paramètres personnalisés |
| **Parametric** | Définir et lancer des études paramétriques |
| **Results** | Visualiser et comparer les voltammogrammes |
| **Settings** | Configuration de l'application |

## Structure du projet

```
03_CV (param & streamlit)/
├── app/
│   ├── main.py                 # Point d'entrée Streamlit
│   └── pages/
│       ├── 1_🔬_Single_Run.py
│       ├── 2_📊_Parametric.py
│       ├── 3_📈_Results.py
│       └── 4_🔧_Settings.py
├── core/
│   ├── simulation.py           # Classe simulation CV
│   ├── mesh_generator.py       # Génération maillage Gmsh
│   ├── parameters.py           # Dataclasses paramètres
│   └── worker.py               # Exécution background
├── data/
│   ├── meshes/                 # Maillages (.msh)
│   └── results/                # Résultats simulations
├── config/
│   └── default_params.yaml
├── requirements.txt
└── README.md
```

## Workflow typique

1. **Configurer** les paramètres dans la page "Single Run"
2. **Lancer** la simulation (génère les fichiers dans `data/results/NNN/`)
3. **Visualiser** les résultats dans la page "Results"
4. **Comparer** plusieurs runs en les sélectionnant

## Études paramétriques

La page "Parametric" permet de :
- Choisir un paramètre à varier (D, k₀, α, scan_rate, etc.)
- Définir une plage de valeurs (linéaire ou logarithmique)
- Générer automatiquement les configurations
- Lancer les simulations en séquence

## Visualisation 3D

La page "Results" intègre PyVista pour visualiser les champs de concentration en 2D/3D directement dans le navigateur.

## Notes techniques

- Les simulations Firedrake tournent dans des subprocesses séparés
- Les données sont stockées en JSON/CSV pour la portabilité
- L'interface Streamlit se rafraîchit automatiquement

## Liens

- [Firedrake](https://firedrakeproject.org)
- [EchemFEM](https://github.com/LLNL/echemfem)
- [Streamlit](https://streamlit.io)
- [PyVista](https://pyvista.org)
