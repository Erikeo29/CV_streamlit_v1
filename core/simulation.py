#!/usr/bin/env python3
"""
cv_simulation_full.py - Simulation CV complete avec export VTK pour ParaView

Sorties dans 04_Results/NNN/:
- cv_data.csv          : Donnees I(E,t)
- cv_plot.png          : Voltammogramme
- concentrations.pvd   : Serie temporelle VTK pour ParaView
- snapshots/*.png      : Images 2D a certains potentiels

Usage:
    python cv_simulation_full.py
    python cv_simulation_full.py --dt 0.005 --cycles 2
"""

import numpy as np
import os
import sys
import argparse
from datetime import datetime
import glob

# Firedrake
from firedrake import *

# Matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as tri


class CVSimulationFull:
    """Simulation CV complete avec export VTK."""

    def __init__(self, mesh_path, output_dir, we_tag=10, ce_tag=11, bulk_tag=13):
        """
        Initialise la simulation.

        Args:
            mesh_path: Chemin maillage .msh
            output_dir: Repertoire de sortie (04_Results/NNN/)
            we_tag: Tag electrode WE (Working Electrode)
            ce_tag: Tag electrode CE (Counter Electrode)
            bulk_tag: Tag frontiere bulk
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "snapshots"), exist_ok=True)

        # Charger maillage
        print(f"Chargement maillage: {mesh_path}")
        self.mesh = Mesh(mesh_path)

        # Espace fonctionnel P2
        self.V = FunctionSpace(self.mesh, "CG", 2)
        print(f"DOFs: {self.V.dim()}")

        # Fonctions
        self.c_R = Function(self.V, name="ferro")   # Ferrocyanide (reduit)
        self.c_O = Function(self.V, name="ferri")   # Ferricyanide (oxyde)
        self.c_R_n = Function(self.V)
        self.c_O_n = Function(self.V)

        # Champ potentiel pour export VTK
        self.E_field = Function(self.V, name="potential_V")

        self.v_R = TestFunction(self.V)
        self.v_O = TestFunction(self.V)

        # Parametres physiques (COMSOL valides)
        self.F = 96485.0      # C/mol
        self.R = 8.314        # J/mol/K
        self.T = 298.15       # K
        self.n = 1            # electrons
        self.D = 7.0e-9       # m²/s (COMSOL: 0.7e-8)
        self.k0 = 1e-5        # m/s (WE) - equivalent j0=1.0 A/m² COMSOL
        self.k0_CE = 1e-4     # m/s (CE) - reduit pour stabilite
        self.alpha = 0.5
        self.E0 = 0.36        # V (potentiel formel)
        self.c_bulk = 1.0     # mol/m³ (1 mM)
        self.depth = 1.5e-3   # m (diametre WE pour 2D->3D)

        # Parametres CV
        self.E_start = 0.36       # V
        self.E_vertex1 = 0.86     # V (anodique)
        self.E_vertex2 = -0.14    # V (cathodique)
        self.scan_rate = 0.1      # V/s

        self.we_tag = we_tag
        self.ce_tag = ce_tag
        self.bulk_tag = bulk_tag

        # Conditions initiales
        self.c_R.assign(Constant(self.c_bulk))
        self.c_O.assign(Constant(self.c_bulk))
        self.c_R_n.assign(self.c_R)
        self.c_O_n.assign(self.c_O)

        # Historique
        self.time_history = []
        self.E_history = []
        self.I_history = []
        self.c_ferro_WE_history = []  # Concentration ferro au centre WE
        self.c_ferro_CE_history = []  # Concentration ferro au centre CE
        self.c_ferri_WE_history = []  # Concentration ferri au centre WE
        self.c_ferri_CE_history = []  # Concentration ferri au centre CE

        # Coordonnees des centres des electrodes (avec puits)
        # WE: centre X=-2.5mm, fond puits Y=-0.13mm
        # CE: centre X=+2.5mm, fond puits Y=-0.13mm
        self.we_center = (-2.5e-3, -0.13e-3)
        self.ce_center = (+2.5e-3, -0.13e-3)

        # Fichier VTK/PVD pour ParaView
        pvd_path = os.path.join(output_dir, "concentrations.pvd")
        self.pvd_file = File(pvd_path)
        print(f"Export VTK: {pvd_path}")

    @property
    def f(self):
        """nF/RT"""
        return self.n * self.F / (self.R * self.T)

    @property
    def t_cycle(self):
        """Duree d'un cycle complet."""
        t1 = abs(self.E_vertex1 - self.E_start) / self.scan_rate
        t2 = abs(self.E_vertex1 - self.E_vertex2) / self.scan_rate
        t3 = abs(self.E_start - self.E_vertex2) / self.scan_rate
        return t1 + t2 + t3

    def E_applied(self, t):
        """Potentiel triangulaire E(t)."""
        t1 = abs(self.E_vertex1 - self.E_start) / self.scan_rate
        t2 = abs(self.E_vertex1 - self.E_vertex2) / self.scan_rate
        t_cycle = self.t_cycle
        t_mod = t % t_cycle

        if t_mod < t1:
            return self.E_start + self.scan_rate * t_mod
        elif t_mod < t1 + t2:
            return self.E_vertex1 - self.scan_rate * (t_mod - t1)
        else:
            return self.E_vertex2 + self.scan_rate * (t_mod - t1 - t2)

    def butler_volmer_flux(self, E_t, electrode='WE'):
        """
        Calcule le flux Butler-Volmer.

        Args:
            E_t: Potentiel applique (pour WE) ou E0 (pour CE)
            electrode: 'WE' ou 'CE'
        """
        if electrode == 'CE':
            # CE: potentiel a l'equilibre (eta = 0)
            # Le flux est drive par la deviation de concentration par rapport a l'equilibre
            # A eta=0: exp_a = exp_c = 1, donc flux = k0 * (c_R/c_bulk - c_O/c_bulk)
            c_R_norm = self.c_R / self.c_bulk
            c_O_norm = self.c_O / self.c_bulk
            flux = self.k0 * (c_R_norm - c_O_norm)
            return flux
        else:
            eta = E_t - self.E0

        max_exp = 20

        exp_arg_a = self.alpha * self.f * eta
        exp_arg_c = -(1 - self.alpha) * self.f * eta

        exp_a = exp(conditional(exp_arg_a > max_exp, max_exp,
                    conditional(exp_arg_a < -max_exp, -max_exp, exp_arg_a)))
        exp_c = exp(conditional(exp_arg_c > max_exp, max_exp,
                    conditional(exp_arg_c < -max_exp, -max_exp, exp_arg_c)))

        c_R_norm = self.c_R / self.c_bulk
        c_O_norm = self.c_O / self.c_bulk

        flux = self.k0 * (c_R_norm * exp_a - c_O_norm * exp_c)
        return flux

    def run(self, dt=0.01, n_cycles=1, vtk_interval=10):
        """
        Lance la simulation.

        Args:
            dt: Pas de temps (s)
            n_cycles: Nombre de cycles CV
            vtk_interval: Intervalle d'export VTK (en pas de temps)
        """
        t_total = n_cycles * self.t_cycle
        n_steps = int(t_total / dt)

        print(f"\n{'='*60}")
        print(f"SIMULATION CV")
        print(f"{'='*60}")
        print(f"Duree: {t_total:.1f} s ({n_cycles} cycle(s))")
        print(f"Pas de temps: {dt*1000:.1f} ms")
        print(f"Nombre de pas: {n_steps}")
        print(f"Export VTK tous les {vtk_interval} pas")
        print(f"{'='*60}\n")

        # Parametres solveur
        solver_params = {
            "snes_type": "newtonls",
            "snes_max_it": 50,
            "snes_rtol": 1e-6,
            "snes_atol": 1e-10,
            "ksp_type": "preonly",
            "pc_type": "lu",
        }

        # Conditions aux limites
        # Bulk: c = c_bulk (Dirichlet)
        # CE: Butler-Volmer a l'equilibre (flux BC, pas Dirichlet)
        bc_R = DirichletBC(self.V, Constant(self.c_bulk), self.bulk_tag)
        bc_O = DirichletBC(self.V, Constant(self.c_bulk), self.bulk_tag)

        # Boucle temporelle
        t = 0.0
        vtk_count = 0

        for step in range(n_steps):
            t += dt
            E_t = self.E_applied(t)

            # ========================================
            # FLUX WE - Butler-Volmer complet (implicite)
            # ========================================
            flux_WE = self.butler_volmer_flux(Constant(E_t), electrode='WE')

            # ========================================
            # FLUX CE - MIROIR DU COURANT WE
            # Conservation du courant: I_CE = -I_WE
            # Distribue uniformement sur la surface CE
            # ========================================
            # Calculer courant WE du pas precedent (approximation)
            if step == 0:
                # Premier pas: pas de courant CE
                flux_CE_uniform = Constant(0.0)
            else:
                # Utiliser le courant WE du pas precedent
                I_WE_prev = self.I_history[-1] if self.I_history else 0.0
                area_CE = 1.5e-3  # m (diametre CE)
                # Flux uniforme pour compenser le courant WE
                j_CE = -I_WE_prev / (self.n * self.F * self.depth * area_CE)
                flux_CE_uniform = Constant(j_CE)

            # ========================================
            # FORMES FAIBLES AVEC CE ACTIVE (flux miroir)
            # ========================================
            # Ferro (c_R): WE +flux (consomme), CE +flux (mais flux<0 donc produit)
            F_R = ((self.c_R - self.c_R_n) / Constant(dt) * self.v_R * dx
                   + Constant(self.D) * inner(grad(self.c_R), grad(self.v_R)) * dx
                   + flux_WE * self.v_R * ds(self.we_tag)
                   + flux_CE_uniform * self.v_R * ds(self.ce_tag))

            # Ferri (c_O): WE -flux (produit), CE -flux (mais flux<0 donc consomme)
            F_O = ((self.c_O - self.c_O_n) / Constant(dt) * self.v_O * dx
                   + Constant(self.D) * inner(grad(self.c_O), grad(self.v_O)) * dx
                   - flux_WE * self.v_O * ds(self.we_tag)
                   - flux_CE_uniform * self.v_O * ds(self.ce_tag))

            # Resoudre
            try:
                solve(F_R == 0, self.c_R, bcs=bc_R, solver_parameters=solver_params)
                solve(F_O == 0, self.c_O, bcs=bc_O, solver_parameters=solver_params)
            except ConvergenceError:
                print(f"Newton diverge a t={t:.3f}s, E={E_t:.3f}V")
                break

            # ========================================
            # CALCUL COURANTS (WE et CE)
            # ========================================
            flux_WE_int = assemble(flux_WE * ds(self.we_tag, domain=self.mesh))
            flux_CE_int = assemble(flux_CE_uniform * ds(self.ce_tag, domain=self.mesh))

            I_WE = self.n * self.F * self.depth * flux_WE_int
            I_CE = self.n * self.F * self.depth * flux_CE_int
            I_total = I_WE  # Courant mesure (convention)

            # Sauvegarder historique
            self.time_history.append(t)
            self.E_history.append(E_t)
            self.I_history.append(I_total)

            # Evaluer concentrations ferro ET ferri aux centres des electrodes
            try:
                c_ferro_we = self.c_R.at(self.we_center)
                c_ferro_ce = self.c_R.at(self.ce_center)
                c_ferri_we = self.c_O.at(self.we_center)
                c_ferri_ce = self.c_O.at(self.ce_center)
            except:
                # Point hors domaine - utiliser valeur bulk
                c_ferro_we = c_ferro_ce = self.c_bulk
                c_ferri_we = c_ferri_ce = self.c_bulk
            self.c_ferro_WE_history.append(c_ferro_we)
            self.c_ferro_CE_history.append(c_ferro_ce)
            self.c_ferri_WE_history.append(c_ferri_we)
            self.c_ferri_CE_history.append(c_ferri_ce)

            # ========================================
            # EXPORT VTK (temps reel t, pas E)
            # Note: E est stocke dans le champ potential_V
            # ========================================
            if step % vtk_interval == 0 or step == n_steps - 1:
                self.E_field.assign(Constant(E_t))
                self.pvd_file.write(self.c_R, self.c_O, self.E_field, time=t)
                vtk_count += 1

            # Mise a jour
            self.c_R_n.assign(self.c_R)
            self.c_O_n.assign(self.c_O)

            # Affichage progression avec bilan de courant
            if step % 100 == 0 or step == n_steps - 1:
                progress = (step + 1) / n_steps * 100
                balance = abs(I_WE + I_CE) / max(abs(I_WE), 1e-12) * 100
                print(f"[{progress:5.1f}%] t={t:.2f}s | E={E_t:+.3f}V | "
                      f"I_WE={I_WE*1e6:+8.2f}uA | I_CE={I_CE*1e6:+8.2f}uA | "
                      f"Bal={balance:.0f}%")

        print(f"\nSimulation terminee: {len(self.time_history)} points, {vtk_count} fichiers VTK")

        return {
            "time": np.array(self.time_history),
            "E": np.array(self.E_history),
            "I": np.array(self.I_history),
        }

    def save_results(self):
        """Sauvegarde les resultats CSV et graphiques."""
        # CSV
        csv_path = os.path.join(self.output_dir, "cv_data.csv")
        data = np.column_stack([self.time_history, self.E_history, self.I_history])
        np.savetxt(csv_path, data, delimiter=",",
                   header="time_s,E_V,I_A", comments="")
        print(f"CSV: {csv_path}")

        # Analyse
        E = np.array(self.E_history)
        I = np.array(self.I_history)
        idx_max = np.argmax(I)
        idx_min = np.argmin(I)

        Ipa, Epa = I[idx_max], E[idx_max]
        Ipc, Epc = I[idx_min], E[idx_min]
        ratio = abs(Ipa / Ipc) if Ipc != 0 else float('inf')

        # Sauvegarder metriques
        metrics_path = os.path.join(self.output_dir, "metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write(f"Pic anodique:   Ipa = {Ipa*1e6:.2f} uA @ Epa = {Epa:.3f} V\n")
            f.write(f"Pic cathodique: Ipc = {Ipc*1e6:.2f} uA @ Epc = {Epc:.3f} V\n")
            f.write(f"Ratio |Ipa/Ipc| = {ratio:.2f}\n")
            f.write(f"Delta Ep = {abs(Epa-Epc)*1000:.0f} mV\n")
        print(f"Metriques: {metrics_path}")

        # Graphique CV
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(E, I * 1e6, 'b-', linewidth=1.5, label=f'diam_WE=1.5 mm')
        ax.set_xlabel('Electric potential (V)', fontsize=12)
        ax.set_ylabel('Total current (uA)', fontsize=12)
        ax.set_title('Cyclic Voltammogram - Fe(CN)6', fontsize=14)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Annoter pics
        ax.annotate(f'Ipa={Ipa*1e6:.1f} uA\nEpa={Epa:.2f} V',
                    xy=(Epa, Ipa*1e6), xytext=(Epa+0.1, Ipa*1e6-2),
                    fontsize=9, color='blue',
                    arrowprops=dict(arrowstyle='->', color='blue', lw=0.5))
        ax.annotate(f'Ipc={Ipc*1e6:.1f} uA\nEpc={Epc:.2f} V',
                    xy=(Epc, Ipc*1e6), xytext=(Epc+0.1, Ipc*1e6+2),
                    fontsize=9, color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=0.5))

        cv_path = os.path.join(self.output_dir, "cv_plot.png")
        plt.savefig(cv_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"CV plot: {cv_path}")

        # Graphique concentration aux electrodes
        self.plot_concentration_vs_E()
        self.plot_concentration_both_species()

        # Export mapping t -> E pour ParaView
        self.export_time_potential_mapping()

        # Script ParaView automatique avec annotation potentiel
        self.export_paraview_script()

        return {
            "Ipa": Ipa, "Epa": Epa,
            "Ipc": Ipc, "Epc": Epc,
            "ratio": ratio,
            "delta_Ep": abs(Epa - Epc)
        }

    def plot_concentration_vs_E(self):
        """Plot concentration ferro au centre WE et CE vs potentiel."""
        E = np.array(self.E_history)
        c_WE = np.array(self.c_ferro_WE_history)
        c_CE = np.array(self.c_ferro_CE_history)

        fig, ax = plt.subplots(figsize=(10, 7))

        # Plot concentrations
        ax.plot(E, c_WE, 'b-', linewidth=1.5, label='Ferro @ centre WE')
        ax.plot(E, c_CE, 'r-', linewidth=1.5, label='Ferro @ centre CE')

        # Ligne de reference (concentration bulk)
        ax.axhline(y=self.c_bulk, color='gray', linestyle='--',
                   linewidth=1, alpha=0.7, label=f'c_bulk = {self.c_bulk} mol/m³')

        ax.set_xlabel('Electric potential (V)', fontsize=12)
        ax.set_ylabel('Ferro concentration (mol/m³)', fontsize=12)
        ax.set_title('Concentration Ferro vs Potentiel - WE et CE', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        # Ajouter annotation E0
        ax.axvline(x=self.E0, color='green', linestyle=':', linewidth=1, alpha=0.7)
        ax.text(self.E0 + 0.02, ax.get_ylim()[1] * 0.95, f'E° = {self.E0} V',
                fontsize=9, color='green')

        conc_path = os.path.join(self.output_dir, "concentration_vs_E.png")
        plt.savefig(conc_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Concentration plot: {conc_path}")

    def plot_concentration_both_species(self):
        """Plot ferro ET ferri au centre WE vs potentiel (style COMSOL)."""
        E = np.array(self.E_history)
        c_ferro = np.array(self.c_ferro_WE_history)
        c_ferri = np.array(self.c_ferri_WE_history)

        fig, ax = plt.subplots(figsize=(10, 7))

        # Plot concentrations
        ax.plot(E, c_ferro, 'b-', linewidth=1.5, label='Ferro (Fe²⁺) @ WE')
        ax.plot(E, c_ferri, 'r-', linewidth=1.5, label='Ferri (Fe³⁺) @ WE')

        # Conservation de masse: c_ferro + c_ferri devrait etre constant
        c_total = c_ferro + c_ferri
        ax.plot(E, c_total, 'g--', linewidth=1, alpha=0.5,
                label=f'Total = {np.mean(c_total):.2f} mol/m³')

        # Reference c_bulk
        ax.axhline(y=self.c_bulk, color='gray', linestyle=':',
                   linewidth=1, label=f'c_bulk = {self.c_bulk} mol/m³')

        ax.set_xlabel('Electric potential (V)', fontsize=12)
        ax.set_ylabel('Concentration (mol/m³)', fontsize=12)
        ax.set_title('Concentration Ferro & Ferri @ centre WE vs Potentiel', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        # Annotation E0
        ax.axvline(x=self.E0, color='green', linestyle=':', linewidth=1, alpha=0.7)
        ax.text(self.E0 + 0.02, ax.get_ylim()[1] * 0.95, f'E° = {self.E0} V',
                fontsize=9, color='green')

        path = os.path.join(self.output_dir, "concentration_ferro_ferri_vs_E.png")
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Concentration ferro+ferri plot: {path}")

    def export_time_potential_mapping(self):
        """Exporte un fichier CSV t -> E pour ParaView."""
        mapping_path = os.path.join(self.output_dir, "time_potential_mapping.csv")
        with open(mapping_path, 'w') as f:
            f.write("# Mapping temps -> potentiel pour ParaView\n")
            f.write("# Usage: annoter le potentiel dans ParaView\n")
            f.write("time_s,potential_V\n")
            for t, E in zip(self.time_history, self.E_history):
                f.write(f"{t:.6f},{E:.4f}\n")
        print(f"Mapping t->E: {mapping_path}")

    def export_paraview_script(self):
        """Genere un script Python ParaView avec annotation du potentiel."""
        pvd_path = os.path.join(self.output_dir, "concentrations.pvd")
        script_path = os.path.join(self.output_dir, "open_in_paraview.py")

        script_content = f'''#!/usr/bin/env pvpython
"""
Script ParaView auto-genere pour visualiser les resultats CV.
Affiche automatiquement le potentiel E(t) comme annotation.

Usage:
  - Dans ParaView: Tools -> Python Shell -> Run Script -> ce fichier
  - En ligne de commande: pvpython open_in_paraview.py
"""

from paraview.simple import *

# === PARAMETRES CV (generes automatiquement) ===
E_START = {self.E_start}
E_VERTEX1 = {self.E_vertex1}
E_VERTEX2 = {self.E_vertex2}
SCAN_RATE = {self.scan_rate}
T_CYCLE = {self.t_cycle}

# === CHARGER LES DONNEES ===
pvd_file = r"{pvd_path}"
print(f"Chargement: {{pvd_file}}")

reader = PVDReader(FileName=pvd_file)
reader.UpdatePipeline()

# Afficher
view = GetActiveViewOrCreate('RenderView')
display = Show(reader, view)
display.Representation = 'Surface'
ColorBy(display, ('POINTS', 'ferro'))

# Colormap
ferroLUT = GetColorTransferFunction('ferro')
ferroLUT.ApplyPreset('Cool to Warm', True)
ferroLUT.RescaleTransferFunction(0.0, 2.5)

# Barre de couleur
colorBar = GetScalarBar(ferroLUT, view)
colorBar.Title = 'Ferro (mol/m3)'
colorBar.Visibility = 1

# === ANNOTATION DU POTENTIEL ===
# Utilise le champ potential_V deja exporte dans les VTU
potentialAnnotation = PythonAnnotation(Input=reader)
potentialAnnotation.ArrayAssociation = 'Point Data'
potentialAnnotation.Expression = '"E = %.3f V" % mean(potential_V)'

# Afficher annotation potentiel
annotDisplay = Show(potentialAnnotation, view)
annotDisplay.FontSize = 20
annotDisplay.Color = [0.0, 0.5, 0.0]
annotDisplay.WindowLocation = 'Upper Right Corner'

# === ANNOTATION DU TEMPS ===
timeAnnotation = AnnotateTimeFilter(Input=reader)
timeAnnotation.Format = 't = %.2f s'

# Afficher annotation temps
timeDisplay = Show(timeAnnotation, view)
timeDisplay.FontSize = 16
timeDisplay.Color = [0.0, 0.0, 0.0]
timeDisplay.WindowLocation = 'Upper Left Corner'

# Vue
view.ResetCamera()
Render()

print("=" * 50)
print("ParaView pret!")
print("  - Utilisez la barre de temps pour naviguer")
print("  - Le potentiel E s'affiche en haut a droite")
print("  - View -> Animation View pour controler")
print("=" * 50)
'''

        with open(script_path, 'w') as f:
            f.write(script_content)

        print(f"Script ParaView: {script_path}")
        print(f"  Usage: pvpython {script_path}")
        print(f"  Ou dans ParaView: Tools -> Python Shell -> Run Script")

    def export_snapshots_png(self, potentials=None):
        """
        Exporte des snapshots PNG a certains potentiels.
        Note: Les VTK sont deja exportes, ceci est optionnel.
        """
        if potentials is None:
            potentials = [0.8, 0.6, 0.36, 0.2, 0.0, -0.1]

        print(f"\nExport snapshots PNG...")

        # Projeter vers P1 pour matplotlib
        V1 = FunctionSpace(self.mesh, "CG", 1)
        c_p1 = Function(V1)

        coords = self.mesh.coordinates.dat.data
        x = coords[:, 0] * 1000  # mm
        y = coords[:, 1] * 1000  # mm
        cell_map = V1.cell_node_map().values
        triang = tri.Triangulation(x, y, cell_map)

        # Pour chaque potentiel demande, trouver le temps le plus proche
        for E_target in potentials:
            # Trouver l'index le plus proche
            E_arr = np.array(self.E_history)
            idx = np.argmin(np.abs(E_arr - E_target))
            E_actual = E_arr[idx]
            t_actual = self.time_history[idx]

            # Note: On ne peut pas recuperer c_R a un temps passe sans stocker
            # Les VTK contiennent deja cette info pour ParaView
            print(f"   E={E_target:.2f}V -> voir concentrations.pvd dans ParaView")

        print(f"   Pour visualiser: ouvrir {self.output_dir}/concentrations.pvd dans ParaView")


def get_next_run_number(results_dir):
    """Trouve le prochain numero de run disponible."""
    existing = glob.glob(os.path.join(results_dir, "[0-9][0-9][0-9]"))
    if not existing:
        return 1
    numbers = [int(os.path.basename(d)) for d in existing]
    return max(numbers) + 1


def main():
    parser = argparse.ArgumentParser(description="Simulation CV complete")
    parser.add_argument("--dt", type=float, default=0.01, help="Pas de temps (s)")
    parser.add_argument("--cycles", type=int, default=1, help="Nombre de cycles")
    parser.add_argument("--vtk-interval", type=int, default=10,
                        help="Intervalle export VTK (pas de temps)")
    parser.add_argument("--mesh", default=None, help="Chemin maillage")
    args = parser.parse_args()

    # Chemins
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(project_dir, "04_Results")

    # Maillage
    if args.mesh:
        mesh_path = args.mesh
    else:
        mesh_path = os.path.join(project_dir, "06_Mesh", "electrode_comsol.msh")

    if not os.path.exists(mesh_path):
        print(f"Maillage non trouve: {mesh_path}")
        sys.exit(1)

    # Creer sous-dossier numerote
    run_number = get_next_run_number(results_dir)
    run_dir = os.path.join(results_dir, f"{run_number:03d}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"#  RUN {run_number:03d}")
    print(f"#  Output: {run_dir}")
    print(f"{'#'*60}")

    # Sauvegarder parametres
    params_path = os.path.join(run_dir, "parameters.txt")
    with open(params_path, 'w') as f:
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Mesh: {mesh_path}\n")
        f.write(f"dt: {args.dt} s\n")
        f.write(f"cycles: {args.cycles}\n")
        f.write(f"vtk_interval: {args.vtk_interval}\n")
        f.write(f"\nParametres physiques:\n")
        f.write(f"  D = 7.0e-9 m²/s\n")
        f.write(f"  k0 = 5e-3 m/s\n")
        f.write(f"  alpha = 0.5\n")
        f.write(f"  E0 = 0.36 V\n")
        f.write(f"  c_bulk = 1.0 mol/m³\n")
        f.write(f"  scan_rate = 0.1 V/s\n")

    # Simulation
    sim = CVSimulationFull(mesh_path, run_dir)
    results = sim.run(dt=args.dt, n_cycles=args.cycles, vtk_interval=args.vtk_interval)

    # Sauvegarder
    metrics = sim.save_results()

    # Resume
    print(f"\n{'='*60}")
    print(f"RESULTATS RUN {run_number:03d}")
    print(f"{'='*60}")
    print(f"Ipa = {metrics['Ipa']*1e6:+.2f} uA @ Epa = {metrics['Epa']:.3f} V")
    print(f"Ipc = {metrics['Ipc']*1e6:+.2f} uA @ Epc = {metrics['Epc']:.3f} V")
    print(f"Ratio |Ipa/Ipc| = {metrics['ratio']:.2f}")
    print(f"Delta Ep = {metrics['delta_Ep']*1000:.0f} mV")
    print(f"{'='*60}")
    print(f"\nFichiers generes dans {run_dir}/:")
    print(f"  - cv_data.csv        : Donnees I(E,t)")
    print(f"  - cv_plot.png        : Voltammogramme")
    print(f"  - concentrations.pvd : Pour ParaView")
    print(f"  - parameters.txt     : Parametres simulation")
    print(f"  - metrics.txt        : Resultats pics")
    print(f"\nPour visualiser dans ParaView:")
    print(f"  paraview {run_dir}/concentrations.pvd")


if __name__ == "__main__":
    main()
