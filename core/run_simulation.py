#!/usr/bin/env python3
"""
run_simulation.py - Lance une simulation CV avec config.json

Usage:
    python run_simulation.py --config path/to/config.json
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Ajouter le chemin du projet
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from simulation import CVSimulationFull


def run_from_config(config_path: Path):
    """
    Lance une simulation à partir d'un fichier config.json.
    """
    # Charger config
    with open(config_path) as f:
        config = json.load(f)

    # Extraire paramètres
    phys = config.get('physical', {})
    cv_params = config.get('cv', {})
    num = config.get('numerical', {})
    geom = config.get('geometry', {})

    # Chemins
    mesh_path = config.get('mesh_path')
    if not mesh_path or not Path(mesh_path).exists():
        mesh_path = PROJECT_DIR / "data" / "meshes" / "electrode_wells.msh"

    output_dir = config.get('output_dir')
    if not output_dir:
        results_dir = PROJECT_DIR / "data" / "results"
        existing = list(results_dir.glob("[0-9][0-9][0-9]"))
        next_num = max([int(d.name) for d in existing], default=0) + 1
        output_dir = results_dir / f"{next_num:03d}"

    output_dir = Path(output_dir)

    # Créer simulation
    sim = CVSimulationFull(str(mesh_path), str(output_dir))

    # Appliquer paramètres physiques
    sim.D = phys.get('D', 7.0e-9)
    sim.k0 = phys.get('k0', 1e-5)
    sim.alpha = phys.get('alpha', 0.5)
    sim.E0 = phys.get('E0', 0.36)
    sim.c_bulk = phys.get('c_bulk', 1.0)

    # Appliquer paramètres CV
    sim.E_start = cv_params.get('E_start', 0.36)
    sim.E_vertex1 = cv_params.get('E_vertex1', 0.86)
    sim.E_vertex2 = cv_params.get('E_vertex2', -0.14)
    sim.scan_rate = cv_params.get('scan_rate', 0.1)

    # Paramètres numériques
    dt = num.get('dt', 0.01)
    n_cycles = cv_params.get('n_cycles', 1)
    vtk_interval = num.get('vtk_interval', 10)

    # Géométrie
    sim.depth = geom.get('depth', 1.5e-3)

    # Sauvegarder config dans output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Lancer simulation
    print(f"\n{'#'*60}")
    print(f"#  SIMULATION CV")
    print(f"#  Output: {output_dir}")
    print(f"{'#'*60}")

    results = sim.run(dt=dt, n_cycles=n_cycles, vtk_interval=vtk_interval)

    # Sauvegarder résultats
    metrics = sim.save_results()

    print(f"\n{'='*60}")
    print(f"RESULTATS")
    print(f"{'='*60}")
    print(f"Ipa = {metrics['Ipa']*1e6:+.2f} uA @ Epa = {metrics['Epa']:.3f} V")
    print(f"Ipc = {metrics['Ipc']*1e6:+.2f} uA @ Epc = {metrics['Epc']:.3f} V")
    print(f"Ratio |Ipa/Ipc| = {metrics['ratio']:.2f}")
    print(f"Delta Ep = {metrics['delta_Ep']*1000:.0f} mV")
    print(f"{'='*60}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run CV simulation from config")
    parser.add_argument("--config", required=True, help="Path to config.json")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERREUR: Config non trouvée: {config_path}")
        sys.exit(1)

    run_from_config(config_path)


if __name__ == "__main__":
    main()
