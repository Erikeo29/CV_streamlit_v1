#!/usr/bin/env python3
"""
mesh_comsol_geometry.py - Maillage fidele a la geometrie COMSOL

Geometrie demi-goutte avec 3 electrodes en PUITS:
- WE (Working Electrode) : gauche, centre X = -2.5 mm, diam 1.5 mm
- REF (Reference Electrode) : centre, centre X = 0 mm, diam 0.8 mm
- CE (Counter Electrode) : droite, centre X = +2.5 mm, diam 1.5 mm

IMPORTANT: Les electrodes sont dans des PUITS (wells):
- Surface isolante entre electrodes: Y = 0
- Fond des puits (electrodes): Y = -0.13 mm
- Parois verticales des puits: isolantes

Tags Physical Groups:
- 10: WE (Working Electrode) - electrode de travail
- 11: CE (Counter Electrode) - contre-electrode
- 12: REF (Reference Electrode) - electrode de reference
- 13: bulk (surface superieure de la goutte)
- 14: insulator (zones isolantes: surface + parois puits)
- 1: electrolyte (domaine 2D)

Usage:
    python mesh_comsol_geometry.py
    python mesh_comsol_geometry.py --view  # Ouvrir dans Gmsh GUI

Sortie:
    06_Mesh/electrode_wells.msh
"""

import gmsh
import sys
import os
import argparse
import numpy as np


def create_comsol_droplet_mesh(output_path, lc_electrode=2e-5, lc_bulk=1e-4, lc_droplet=5e-5, we_center_x=-2.5e-3):
    """
    Cree un maillage reproduisant la geometrie COMSOL demi-goutte avec electrodes en puits.

    Geometrie:
    - Surface isolante: Y = 0
    - Electrodes (fond des puits): Y = -0.13 mm
    - Parois des puits: isolantes (verticales)

    Args:
        output_path: Chemin fichier .msh
        lc_electrode: Taille maille aux electrodes (m)
        lc_bulk: Taille maille dans le bulk (m)
        lc_droplet: Taille maille sur la surface goutte (m)
    """
    gmsh.initialize()
    gmsh.model.add("cv_comsol_droplet_wells")

    # ==========================================================
    # DIMENSIONS (en metres)
    # ==========================================================

    # Domaine
    x_min = -5e-3   # -5 mm
    x_max = 5e-3    # +5 mm

    # Hauteur de la goutte (forme elliptique)
    y_droplet_max = 3e-3  # 3 mm au centre

    # Profondeur des puits (electrodes)
    well_depth = 0.13e-3  # 0.13 mm
    y_electrode = -well_depth  # Y = -0.13 mm
    y_surface = 0.0  # Y = 0 (surface isolante)

    # Electrodes - positions X (centres et demi-largeurs)
    # WE: centre parametrable, diam 1.5 mm
    we_center = we_center_x
    we_half = 0.75e-3
    we_x1 = we_center - we_half  # -3.25 mm
    we_x2 = we_center + we_half  # -1.75 mm

    # REF: centre 0 mm, diam 0.8 mm
    ref_center = 0.0
    ref_half = 0.4e-3
    ref_x1 = ref_center - ref_half  # -0.4 mm
    ref_x2 = ref_center + ref_half  # +0.4 mm

    # CE: centre +2.5 mm, diam 1.5 mm
    ce_center = 2.5e-3
    ce_half = 0.75e-3
    ce_x1 = ce_center - ce_half  # +1.75 mm
    ce_x2 = ce_center + ce_half  # +3.25 mm

    # ==========================================================
    # POINTS DU CONTOUR (avec puits)
    # De gauche a droite, en suivant le contour bas
    # ==========================================================

    # Point 1: coin bas gauche (bord goutte a Y=0)
    p1 = gmsh.model.geo.addPoint(x_min, y_surface, 0, lc_bulk)

    # --- PUITS WE ---
    # Point 2: avant puits WE (surface Y=0)
    p2 = gmsh.model.geo.addPoint(we_x1, y_surface, 0, lc_electrode)
    # Point 3: coin haut gauche puits WE (descente)
    p3 = gmsh.model.geo.addPoint(we_x1, y_electrode, 0, lc_electrode)
    # Point 4: coin bas droit puits WE (fond)
    p4 = gmsh.model.geo.addPoint(we_x2, y_electrode, 0, lc_electrode)
    # Point 5: coin haut droit puits WE (montee)
    p5 = gmsh.model.geo.addPoint(we_x2, y_surface, 0, lc_electrode)

    # --- PUITS REF ---
    # Point 6: avant puits REF (surface Y=0)
    p6 = gmsh.model.geo.addPoint(ref_x1, y_surface, 0, lc_electrode)
    # Point 7: coin haut gauche puits REF
    p7 = gmsh.model.geo.addPoint(ref_x1, y_electrode, 0, lc_electrode)
    # Point 8: coin bas droit puits REF
    p8 = gmsh.model.geo.addPoint(ref_x2, y_electrode, 0, lc_electrode)
    # Point 9: coin haut droit puits REF
    p9 = gmsh.model.geo.addPoint(ref_x2, y_surface, 0, lc_electrode)

    # --- PUITS CE ---
    # Point 10: avant puits CE (surface Y=0)
    p10 = gmsh.model.geo.addPoint(ce_x1, y_surface, 0, lc_electrode)
    # Point 11: coin haut gauche puits CE
    p11 = gmsh.model.geo.addPoint(ce_x1, y_electrode, 0, lc_electrode)
    # Point 12: coin bas droit puits CE
    p12 = gmsh.model.geo.addPoint(ce_x2, y_electrode, 0, lc_electrode)
    # Point 13: coin haut droit puits CE
    p13 = gmsh.model.geo.addPoint(ce_x2, y_surface, 0, lc_electrode)

    # Point 14: coin bas droit (bord goutte a Y=0)
    p14 = gmsh.model.geo.addPoint(x_max, y_surface, 0, lc_bulk)

    # ==========================================================
    # LIGNES DU CONTOUR INFERIEUR (avec puits)
    # ==========================================================

    # Surface isolante gauche (bord -> WE)
    l_iso_left = gmsh.model.geo.addLine(p1, p2)

    # Puits WE
    l_we_wall_left = gmsh.model.geo.addLine(p2, p3)   # Paroi gauche (descente)
    l_we = gmsh.model.geo.addLine(p3, p4)             # Electrode WE (fond)
    l_we_wall_right = gmsh.model.geo.addLine(p4, p5)  # Paroi droite (montee)

    # Surface isolante WE -> REF
    l_iso_we_ref = gmsh.model.geo.addLine(p5, p6)

    # Puits REF
    l_ref_wall_left = gmsh.model.geo.addLine(p6, p7)   # Paroi gauche
    l_ref = gmsh.model.geo.addLine(p7, p8)             # Electrode REF
    l_ref_wall_right = gmsh.model.geo.addLine(p8, p9)  # Paroi droite

    # Surface isolante REF -> CE
    l_iso_ref_ce = gmsh.model.geo.addLine(p9, p10)

    # Puits CE
    l_ce_wall_left = gmsh.model.geo.addLine(p10, p11)   # Paroi gauche
    l_ce = gmsh.model.geo.addLine(p11, p12)             # Electrode CE
    l_ce_wall_right = gmsh.model.geo.addLine(p12, p13)  # Paroi droite

    # Surface isolante droite (CE -> bord)
    l_iso_right = gmsh.model.geo.addLine(p13, p14)

    # ==========================================================
    # SURFACE SUPERIEURE: DEMI-GOUTTE (arc elliptique)
    # ==========================================================

    # Points sur l'arc de la goutte (de droite a gauche)
    n_arc_points = 30
    arc_points = []

    # Epaisseur minimale aux bords (comme COMSOL)
    y_edge_min = 0.13e-3  # 0.13 mm

    for i in range(n_arc_points + 1):
        t = i / n_arc_points
        # X: de x_max a x_min
        x = x_max - t * (x_max - x_min)
        # Y: forme elliptique avec hauteur minimale aux bords
        x_norm = x / x_max
        y_ellipse = y_droplet_max * np.sqrt(max(0, 1 - x_norm**2))
        y = max(y_ellipse, y_edge_min)  # Au moins 0.13 mm aux bords

        lc = lc_droplet if abs(x) < 2e-3 else lc_bulk
        p = gmsh.model.geo.addPoint(x, y, 0, lc)
        arc_points.append(p)

    # Spline pour la surface de la goutte
    l_droplet = gmsh.model.geo.addSpline(arc_points)

    # ==========================================================
    # LIGNES VERTICALES AUX BORDS (goutte)
    # ==========================================================

    # Bord droit: p14 (Y=0) -> premier point arc (haut)
    l_right = gmsh.model.geo.addLine(p14, arc_points[0])

    # Bord gauche: dernier point arc -> p1 (Y=0)
    l_left = gmsh.model.geo.addLine(arc_points[-1], p1)

    # ==========================================================
    # CONTOUR FERME ET SURFACE
    # ==========================================================

    loop = gmsh.model.geo.addCurveLoop([
        # Bas: gauche -> droite avec puits
        l_iso_left,
        l_we_wall_left, l_we, l_we_wall_right,
        l_iso_we_ref,
        l_ref_wall_left, l_ref, l_ref_wall_right,
        l_iso_ref_ce,
        l_ce_wall_left, l_ce, l_ce_wall_right,
        l_iso_right,
        # Droite -> haut -> gauche
        l_right, l_droplet, l_left
    ])

    surf = gmsh.model.geo.addPlaneSurface([loop])

    gmsh.model.geo.synchronize()

    # ==========================================================
    # PHYSICAL GROUPS (tags pour Firedrake)
    # ==========================================================

    # Tag 10: WE (Working Electrode) - fond du puits
    gmsh.model.addPhysicalGroup(1, [l_we], 10)
    gmsh.model.setPhysicalName(1, 10, "WE")

    # Tag 11: CE (Counter Electrode) - fond du puits
    gmsh.model.addPhysicalGroup(1, [l_ce], 11)
    gmsh.model.setPhysicalName(1, 11, "CE")

    # Tag 12: REF (Reference Electrode) - fond du puits
    gmsh.model.addPhysicalGroup(1, [l_ref], 12)
    gmsh.model.setPhysicalName(1, 12, "REF")

    # Tag 13: bulk (surface goutte - condition Dirichlet)
    gmsh.model.addPhysicalGroup(1, [l_droplet], 13)
    gmsh.model.setPhysicalName(1, 13, "bulk")

    # Tag 14: insulator (surfaces isolantes + parois puits + bords)
    insulator_lines = [
        l_iso_left, l_iso_we_ref, l_iso_ref_ce, l_iso_right,  # Surfaces horizontales
        l_we_wall_left, l_we_wall_right,                       # Parois puits WE
        l_ref_wall_left, l_ref_wall_right,                     # Parois puits REF
        l_ce_wall_left, l_ce_wall_right,                       # Parois puits CE
        l_left, l_right                                        # Bords verticaux goutte
    ]
    gmsh.model.addPhysicalGroup(1, insulator_lines, 14)
    gmsh.model.setPhysicalName(1, 14, "insulator")

    # Tag 1: domaine electrolyte (surface 2D)
    gmsh.model.addPhysicalGroup(2, [surf], 1)
    gmsh.model.setPhysicalName(2, 1, "electrolyte")

    # ==========================================================
    # RAFFINEMENT ADAPTATIF
    # ==========================================================

    # Champ de distance aux electrodes (fond des puits)
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", [l_we, l_ce, l_ref])
    gmsh.model.mesh.field.setNumber(1, "Sampling", 200)

    # Champ de distance aux parois des puits
    gmsh.model.mesh.field.add("Distance", 2)
    gmsh.model.mesh.field.setNumbers(2, "CurvesList", [
        l_we_wall_left, l_we_wall_right,
        l_ref_wall_left, l_ref_wall_right,
        l_ce_wall_left, l_ce_wall_right
    ])
    gmsh.model.mesh.field.setNumber(2, "Sampling", 100)

    # Minimum des distances
    gmsh.model.mesh.field.add("Min", 3)
    gmsh.model.mesh.field.setNumbers(3, "FieldsList", [1, 2])

    # Transition de taille
    gmsh.model.mesh.field.add("Threshold", 4)
    gmsh.model.mesh.field.setNumber(4, "InField", 3)
    gmsh.model.mesh.field.setNumber(4, "SizeMin", lc_electrode)
    gmsh.model.mesh.field.setNumber(4, "SizeMax", lc_bulk)
    gmsh.model.mesh.field.setNumber(4, "DistMin", 0)
    gmsh.model.mesh.field.setNumber(4, "DistMax", 5e-4)  # Transition sur 500 um

    gmsh.model.mesh.field.setAsBackgroundMesh(4)

    # Options de maillage
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    # ==========================================================
    # GENERATION ET SAUVEGARDE
    # ==========================================================

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Netgen")

    # Format Gmsh 2.2 pour compatibilite Firedrake
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(output_path)

    # Statistiques
    nodes = gmsh.model.mesh.getNodes()
    elements_2d = gmsh.model.mesh.getElements(2)
    elements_1d = gmsh.model.mesh.getElements(1)

    print(f"\n{'='*60}")
    print(f"MAILLAGE GEOMETRIE COMSOL - ELECTRODES EN PUITS")
    print(f"{'='*60}")
    print(f"Fichier: {output_path}")
    print(f"Noeuds: {len(nodes[0])}")
    print(f"Elements 2D (triangles): {len(elements_2d[1][0]) if elements_2d[1] else 0}")
    print(f"Elements 1D (segments): {sum(len(e) for e in elements_1d[1]) if elements_1d[1] else 0}")
    print(f"\nGeometrie des puits:")
    print(f"  Surface isolante: Y = 0 mm")
    print(f"  Fond des puits (electrodes): Y = {y_electrode*1e3:.2f} mm")
    print(f"  Profondeur puits: {well_depth*1e3:.2f} mm")
    print(f"\nElectrodes (fond des puits):")
    print(f"  WE:  X in [{we_x1*1e3:.2f}, {we_x2*1e3:.2f}] mm (diam {we_half*2*1e3:.1f} mm)")
    print(f"  REF: X in [{ref_x1*1e3:.2f}, {ref_x2*1e3:.2f}] mm (diam {ref_half*2*1e3:.1f} mm)")
    print(f"  CE:  X in [{ce_x1*1e3:.2f}, {ce_x2*1e3:.2f}] mm (diam {ce_half*2*1e3:.1f} mm)")
    print(f"\nPhysical Groups (tags):")
    print(f"  10: WE  (fond puits)")
    print(f"  11: CE  (fond puits)")
    print(f"  12: REF (fond puits)")
    print(f"  13: bulk (surface goutte)")
    print(f"  14: insulator (surfaces + parois puits)")
    print(f"  1:  electrolyte (domaine 2D)")
    print(f"{'='*60}\n")

    return gmsh


def main():
    parser = argparse.ArgumentParser(description="Maillage geometrie COMSOL avec puits")
    parser.add_argument("--output", "-o", default=None, help="Fichier de sortie")
    parser.add_argument("--lc-electrode", type=float, default=2e-5, help="Taille maille electrode (m)")
    parser.add_argument("--lc-bulk", type=float, default=1e-4, help="Taille maille bulk (m)")
    parser.add_argument("--we-x", type=float, default=-2.5, help="Position X centre WE en mm (default: -2.5)")
    parser.add_argument("--view", action="store_true", help="Ouvrir dans Gmsh GUI")

    args = parser.parse_args()

    # Chemin de sortie
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    if args.output:
        output_path = args.output
    else:
        mesh_dir = os.path.join(project_dir, "data", "meshes")
        os.makedirs(mesh_dir, exist_ok=True)
        output_path = os.path.join(mesh_dir, "electrode_wells.msh")

    # Creer le maillage
    we_center_x = args.we_x * 1e-3  # Convertir mm -> m
    gmsh_instance = create_comsol_droplet_mesh(
        output_path,
        lc_electrode=args.lc_electrode,
        lc_bulk=args.lc_bulk,
        we_center_x=we_center_x
    )

    # Ouvrir GUI si demande
    if args.view:
        print("Ouverture Gmsh GUI...")
        gmsh.fltk.run()

    gmsh.finalize()

    print(f"Maillage sauvegarde: {output_path}")


if __name__ == "__main__":
    main()
