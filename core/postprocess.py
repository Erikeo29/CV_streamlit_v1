"""
postprocess.py - Génère des images PNG à partir des fichiers VTU
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def generate_concentration_images(run_dir: Path, field: str = "ferro",
                                   cmap: str = "coolwarm", dpi: int = 150):
    """
    Génère des images PNG pour chaque fichier VTU.

    Args:
        run_dir: Dossier du run (contient concentrations/ et time_potential_mapping.csv)
        field: Champ à visualiser ("ferro" ou "ferri")
        cmap: Colormap matplotlib
        dpi: Résolution des images

    Returns:
        Liste des chemins des images générées
    """
    import pyvista as pv

    # Désactiver l'affichage interactif
    pv.OFF_SCREEN = True

    vtu_dir = run_dir / "concentrations"
    output_dir = run_dir / "snapshots"
    output_dir.mkdir(exist_ok=True)

    # Charger le mapping temps -> potentiel
    mapping_file = run_dir / "time_potential_mapping.csv"
    if mapping_file.exists():
        df_map = pd.read_csv(mapping_file, comment='#')
        time_to_potential = dict(zip(df_map['time_s'], df_map['potential_V']))
    else:
        time_to_potential = {}

    # Lister les fichiers VTU
    vtu_files = sorted(vtu_dir.glob("*.vtu"),
                       key=lambda x: int(x.stem.split("_")[-1]))

    if not vtu_files:
        print("Aucun fichier VTU trouvé")
        return []

    # Calculer les limites globales pour le colorbar
    print(f"Analyse de {len(vtu_files)} fichiers VTU...")
    global_min = float('inf')
    global_max = float('-inf')

    for vtu_path in vtu_files:
        mesh = pv.read(str(vtu_path))
        if field in mesh.array_names:
            data = mesh[field]
            global_min = min(global_min, np.min(data))
            global_max = max(global_max, np.max(data))

    print(f"Plage de {field}: [{global_min:.4f}, {global_max:.4f}] mol/m³")

    # Lire le config pour obtenir dt et vtk_interval
    config_file = run_dir / "config.json"
    dt = 0.005  # default
    vtk_interval = 50  # default

    if config_file.exists():
        import json
        with open(config_file) as f:
            config = json.load(f)
        numerical = config.get('numerical', {})
        dt = numerical.get('dt', 0.005)
        vtk_interval = numerical.get('vtk_interval', 50)

    # Générer les images
    generated = []
    print(f"Génération des images PNG...")

    for i, vtu_path in enumerate(vtu_files):
        mesh = pv.read(str(vtu_path))

        # Calculer le temps correspondant
        frame_idx = int(vtu_path.stem.split("_")[-1])
        time_s = frame_idx * vtk_interval * dt

        # Obtenir le potentiel correspondant
        # Trouver le temps le plus proche dans le mapping
        if time_to_potential:
            times = np.array(list(time_to_potential.keys()))
            closest_idx = np.argmin(np.abs(times - time_s))
            closest_time = times[closest_idx]
            potential = time_to_potential[closest_time]
        else:
            potential = None

        # Créer le plotter
        plotter = pv.Plotter(off_screen=True, window_size=[800, 600])

        if field in mesh.array_names:
            plotter.add_mesh(mesh, scalars=field, cmap=cmap,
                           clim=[global_min, global_max],
                           scalar_bar_args={
                               "title": f"{field.capitalize()} (mol/m³)",
                               "vertical": True,
                               "position_x": 0.85,
                               "position_y": 0.1,
                               "height": 0.8,
                               "width": 0.1,
                           })
        else:
            plotter.add_mesh(mesh, color="lightgray")

        # Titre avec potentiel
        if potential is not None:
            title = f"E = {potential:.3f} V (t = {time_s:.2f} s)"
        else:
            title = f"Frame {frame_idx} (t = {time_s:.2f} s)"

        plotter.add_title(title, font_size=12)
        plotter.view_xy()
        plotter.camera.zoom(1.2)

        # Sauvegarder
        output_path = output_dir / f"{field}_{frame_idx:04d}.png"
        plotter.screenshot(str(output_path))
        plotter.close()

        generated.append({
            'path': output_path,
            'frame': frame_idx,
            'time_s': time_s,
            'potential_V': potential
        })

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(vtu_files)} images générées")

    print(f"Terminé: {len(generated)} images dans {output_dir}")

    # Sauvegarder l'index des images
    df_index = pd.DataFrame(generated)
    df_index['path'] = df_index['path'].apply(lambda x: x.name)
    df_index.to_csv(output_dir / "index.csv", index=False)

    return generated


def create_animation(run_dir: Path, field: str = "ferro", fps: int = 10):
    """
    Crée une animation GIF à partir des images PNG.

    Args:
        run_dir: Dossier du run
        field: Champ ("ferro" ou "ferri")
        fps: Frames par seconde
    """
    from PIL import Image

    snapshots_dir = run_dir / "snapshots"
    if not snapshots_dir.exists():
        print("Dossier snapshots non trouvé. Lancez d'abord generate_concentration_images()")
        return None

    # Lister les images
    images = sorted(snapshots_dir.glob(f"{field}_*.png"),
                    key=lambda x: int(x.stem.split("_")[-1]))

    if not images:
        print(f"Aucune image {field}_*.png trouvée")
        return None

    print(f"Création de l'animation à partir de {len(images)} images...")

    # Charger les images
    frames = [Image.open(str(img)) for img in images]

    # Sauvegarder en GIF
    output_path = run_dir / f"{field}_animation.gif"
    frames[0].save(
        str(output_path),
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0
    )

    print(f"Animation sauvegardée: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-traitement des résultats CV")
    parser.add_argument("run_dir", type=Path, help="Dossier du run")
    parser.add_argument("--field", default="ferro", choices=["ferro", "ferri"],
                        help="Champ à visualiser")
    parser.add_argument("--cmap", default="coolwarm", help="Colormap")
    parser.add_argument("--dpi", type=int, default=150, help="Résolution DPI")
    parser.add_argument("--animation", action="store_true", help="Créer animation GIF")
    parser.add_argument("--fps", type=int, default=10, help="FPS pour l'animation")

    args = parser.parse_args()

    generate_concentration_images(args.run_dir, args.field, args.cmap, args.dpi)

    if args.animation:
        create_animation(args.run_dir, args.field, args.fps)
