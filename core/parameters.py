"""
parameters.py - Dataclasses pour les parametres de simulation CV
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List
import json
from pathlib import Path


@dataclass
class PhysicalParameters:
    """Parametres physiques de la simulation."""
    D: float = 7.0e-9           # Coefficient de diffusion (m²/s)
    k0: float = 1e-5            # Constante cinetique (m/s)
    alpha: float = 0.5          # Coefficient de transfert
    E0: float = 0.36            # Potentiel formel (V)
    c_bulk: float = 1.0         # Concentration bulk (mol/m³)
    n: int = 1                  # Nombre d'electrons
    T: float = 298.15           # Temperature (K)
    F: float = 96485.0          # Constante de Faraday (C/mol)
    R: float = 8.314            # Constante des gaz (J/mol/K)


@dataclass
class CVParameters:
    """Parametres du signal CV."""
    E_start: float = 0.36       # Potentiel initial (V)
    E_vertex1: float = 0.86     # Vertex anodique (V)
    E_vertex2: float = -0.14    # Vertex cathodique (V)
    scan_rate: float = 0.1      # Vitesse de balayage (V/s)
    n_cycles: int = 2           # Nombre de cycles


@dataclass
class NumericalParameters:
    """Parametres numeriques."""
    dt: float = 0.005           # Pas de temps (s)
    vtk_interval: int = 50      # Intervalle export VTK


@dataclass
class GeometryParameters:
    """Parametres geometriques."""
    we_x: float = -2.5e-3       # Position X du WE (m)
    we_diameter: float = 1.5e-3 # Diametre WE (m)
    ce_x: float = 2.5e-3        # Position X du CE (m)
    ref_x: float = 0.0          # Position X du REF (m)
    depth: float = 1.5e-3       # Profondeur pour 2D->3D (m)


@dataclass
class SimulationConfig:
    """Configuration complete d'une simulation."""
    name: str = "simulation"
    physical: PhysicalParameters = field(default_factory=PhysicalParameters)
    cv: CVParameters = field(default_factory=CVParameters)
    numerical: NumericalParameters = field(default_factory=NumericalParameters)
    geometry: GeometryParameters = field(default_factory=GeometryParameters)
    mesh_path: Optional[str] = None
    output_dir: Optional[str] = None

    def to_dict(self) -> dict:
        """Convertit en dictionnaire."""
        return {
            "name": self.name,
            "physical": asdict(self.physical),
            "cv": asdict(self.cv),
            "numerical": asdict(self.numerical),
            "geometry": asdict(self.geometry),
            "mesh_path": self.mesh_path,
            "output_dir": self.output_dir,
        }

    def save(self, path: Path):
        """Sauvegarde en JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "SimulationConfig":
        """Charge depuis JSON."""
        with open(path, 'r') as f:
            data = json.load(f)

        return cls(
            name=data.get("name", "simulation"),
            physical=PhysicalParameters(**data.get("physical", {})),
            cv=CVParameters(**data.get("cv", {})),
            numerical=NumericalParameters(**data.get("numerical", {})),
            geometry=GeometryParameters(**data.get("geometry", {})),
            mesh_path=data.get("mesh_path"),
            output_dir=data.get("output_dir"),
        )


@dataclass
class ParametricStudy:
    """Definition d'une etude parametrique."""
    name: str
    base_config: SimulationConfig
    parameter_path: str          # Ex: "physical.k0"
    values: List[float] = field(default_factory=list)

    def generate_configs(self) -> List[SimulationConfig]:
        """Genere les configurations pour chaque valeur."""
        configs = []
        parts = self.parameter_path.split(".")

        for i, value in enumerate(self.values):
            # Deep copy de la config de base
            config_dict = self.base_config.to_dict()
            config_dict["name"] = f"{self.name}_{i:03d}"

            # Modifier le parametre
            obj = config_dict
            for part in parts[:-1]:
                obj = obj[part]
            obj[parts[-1]] = value

            # Recreer la config
            config = SimulationConfig(
                name=config_dict["name"],
                physical=PhysicalParameters(**config_dict["physical"]),
                cv=CVParameters(**config_dict["cv"]),
                numerical=NumericalParameters(**config_dict["numerical"]),
                geometry=GeometryParameters(**config_dict["geometry"]),
                mesh_path=config_dict.get("mesh_path"),
                output_dir=config_dict.get("output_dir"),
            )
            configs.append(config)

        return configs
