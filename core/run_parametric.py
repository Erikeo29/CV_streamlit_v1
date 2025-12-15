"""
run_parametric.py - Lance une étude paramétrique
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Ajouter le dossier parent pour les imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))


def get_next_run_number(results_dir: Path) -> int:
    """Retourne le prochain numéro de run disponible."""
    existing = [int(d.name) for d in results_dir.iterdir()
                if d.is_dir() and d.name.isdigit()]
    return max(existing, default=0) + 1


def set_nested_value(d: dict, path: str, value):
    """Set une valeur dans un dict imbriqué via un chemin 'a.b.c'."""
    keys = path.split('.')
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def run_parametric_study(study_path: Path):
    """
    Lance une étude paramétrique.

    Args:
        study_path: Chemin vers le fichier JSON de l'étude
    """
    # Charger l'étude
    with open(study_path) as f:
        study = json.load(f)

    name = study['name']
    param_path = study['parameter_path']
    values = study['values']
    base_config = study['base_config']

    print(f"=== Étude paramétrique: {name} ===")
    print(f"Paramètre: {param_path}")
    print(f"Valeurs: {values}")
    print(f"Nombre de simulations: {len(values)}")
    print()

    # Dossier résultats
    results_dir = PROJECT_DIR / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Chemin du mesh
    mesh_path = PROJECT_DIR / "data" / "meshes" / "electrode_comsol.msh"
    if not mesh_path.exists():
        print(f"ERREUR: Mesh non trouvé: {mesh_path}")
        sys.exit(1)

    # Import Firedrake (doit être fait après activation de l'environnement)
    print("Import de Firedrake...")
    from simulation import CVSimulationFull

    # Exécuter chaque simulation
    results = []
    for i, value in enumerate(values):
        run_num = get_next_run_number(results_dir)
        run_dir = results_dir / f"{run_num:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Simulation {i+1}/{len(values)}: {param_path} = {value:.2e}")
        print(f"Run: {run_num:03d}")
        print(f"{'='*60}")

        # Créer la config pour cette simulation
        config = base_config.copy()
        config['name'] = f"{name}_{i+1:02d}"
        config['output_dir'] = str(run_dir)
        config['mesh_path'] = str(mesh_path)

        # Appliquer la valeur du paramètre
        set_nested_value(config, param_path, value)

        # Sauvegarder la config
        config_path = run_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Extraire les paramètres
        phys = config.get('physical', {})
        cv_params = config.get('cv', {})
        numerical = config.get('numerical', {})

        # Créer et configurer la simulation
        try:
            sim = CVSimulationFull(str(mesh_path), str(run_dir))

            # Paramètres physiques
            sim.D = phys.get('D', 7.0e-9)
            sim.k0 = phys.get('k0', 1.0e-5)
            sim.alpha = phys.get('alpha', 0.5)
            sim.E0 = phys.get('E0', 0.36)
            sim.c_bulk = phys.get('c_bulk', 1.0)

            # Paramètres CV
            sim.E_start = cv_params.get('E_start', 0.36)
            sim.E_vertex1 = cv_params.get('E_vertex1', 0.86)
            sim.E_vertex2 = cv_params.get('E_vertex2', -0.14)
            sim.scan_rate = cv_params.get('scan_rate', 0.1)

            # Paramètres numériques
            dt = numerical.get('dt', 0.005)
            n_cycles = cv_params.get('n_cycles', 2)
            vtk_interval = numerical.get('vtk_interval', 50)

            # Lancer la simulation
            start_time = datetime.now()
            sim.run(dt=dt, n_cycles=n_cycles, vtk_interval=vtk_interval)
            metrics = sim.save_results()
            end_time = datetime.now()

            elapsed = (end_time - start_time).total_seconds()
            print(f"\nSimulation terminée en {elapsed:.1f}s")
            print(f"Métriques: {metrics}")

            results.append({
                'run': run_num,
                'value': value,
                'metrics': metrics,
                'elapsed_s': elapsed,
                'status': 'success'
            })

        except Exception as e:
            print(f"\nERREUR: {e}")
            results.append({
                'run': run_num,
                'value': value,
                'error': str(e),
                'status': 'failed'
            })

    # Résumé final
    print("\n" + "="*60)
    print("RÉSUMÉ DE L'ÉTUDE PARAMÉTRIQUE")
    print("="*60)

    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"Simulations réussies: {success_count}/{len(results)}")

    # Sauvegarder les résultats de l'étude
    study_results_file = PROJECT_DIR / "data" / "studies" / f"{name}_results.json"
    with open(study_results_file, 'w') as f:
        json.dump({
            'study': study,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\nRésultats sauvegardés: {study_results_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lance une étude paramétrique CV")
    parser.add_argument("--study", type=Path, required=True,
                        help="Chemin vers le fichier JSON de l'étude")

    args = parser.parse_args()

    if not args.study.exists():
        print(f"ERREUR: Fichier d'étude non trouvé: {args.study}")
        sys.exit(1)

    run_parametric_study(args.study)
