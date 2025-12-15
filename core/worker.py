"""
worker.py - Execution des simulations en background
"""

import subprocess
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable
from enum import Enum
import threading
import time


class SimulationStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SimulationJob:
    """Represente un job de simulation."""
    job_id: str
    config_path: Path
    output_dir: Path
    status: SimulationStatus = SimulationStatus.PENDING
    progress: float = 0.0
    message: str = ""
    pid: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "config_path": str(self.config_path),
            "output_dir": str(self.output_dir),
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "pid": self.pid,
        }


class SimulationWorker:
    """
    Lance et surveille les simulations CV.

    Les simulations Firedrake doivent tourner dans un subprocess separe
    car Firedrake n'est pas compatible avec le threading de Streamlit.
    """

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.jobs: dict[str, SimulationJob] = {}
        self.runner_script = self.base_dir / "core" / "run_simulation.py"

    def submit_job(self, config_path: Path, output_dir: Path) -> SimulationJob:
        """Soumet un nouveau job de simulation."""
        import uuid
        job_id = str(uuid.uuid4())[:8]

        job = SimulationJob(
            job_id=job_id,
            config_path=config_path,
            output_dir=output_dir,
        )
        self.jobs[job_id] = job

        # Lancer dans un thread
        thread = threading.Thread(target=self._run_job, args=(job,))
        thread.daemon = True
        thread.start()

        return job

    def _run_job(self, job: SimulationJob):
        """Execute le job dans un subprocess."""
        job.status = SimulationStatus.RUNNING
        job.message = "Demarrage..."

        try:
            # Commande pour lancer la simulation
            # Note: on utilise le script runner qui active Firedrake
            cmd = [
                "bash", "-c",
                f"source ~/firedrake/firedrake-env/bin/activate && "
                f"python {self.runner_script} "
                f"--config {job.config_path} "
                f"--output {job.output_dir}"
            ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(self.base_dir),
            )
            job.pid = process.pid

            # Lire la sortie en temps reel
            for line in process.stdout:
                line = line.strip()
                if line:
                    job.message = line
                    # Parser la progression si possible
                    if "%" in line and "[" in line:
                        try:
                            pct = float(line.split("%")[0].split("[")[-1].strip())
                            job.progress = pct / 100.0
                        except:
                            pass

            process.wait()

            if process.returncode == 0:
                job.status = SimulationStatus.COMPLETED
                job.progress = 1.0
                job.message = "Simulation terminee"
            else:
                job.status = SimulationStatus.FAILED
                job.message = f"Erreur (code {process.returncode})"

        except Exception as e:
            job.status = SimulationStatus.FAILED
            job.message = str(e)

    def get_job(self, job_id: str) -> Optional[SimulationJob]:
        """Recupere un job par son ID."""
        return self.jobs.get(job_id)

    def get_all_jobs(self) -> list[SimulationJob]:
        """Retourne tous les jobs."""
        return list(self.jobs.values())

    def cancel_job(self, job_id: str) -> bool:
        """Annule un job en cours."""
        job = self.jobs.get(job_id)
        if job and job.status == SimulationStatus.RUNNING and job.pid:
            try:
                os.kill(job.pid, 9)
                job.status = SimulationStatus.CANCELLED
                job.message = "Annule par l'utilisateur"
                return True
            except:
                pass
        return False


def monitor_progress(output_dir: Path) -> dict:
    """
    Lit le fichier de progression d'une simulation.

    Returns:
        dict avec 'progress', 'time', 'E', 'I', etc.
    """
    progress_file = output_dir / "progress.json"
    cv_data_file = output_dir / "cv_data.csv"

    result = {
        "progress": 0.0,
        "n_points": 0,
        "time": [],
        "E": [],
        "I": [],
    }

    # Lire progression
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                data = json.load(f)
                result["progress"] = data.get("progress", 0.0)
        except:
            pass

    # Lire donnees CV partielles
    if cv_data_file.exists():
        try:
            import numpy as np
            data = np.loadtxt(cv_data_file, delimiter=",", skiprows=1)
            if len(data) > 0:
                result["time"] = data[:, 0].tolist()
                result["E"] = data[:, 1].tolist()
                result["I"] = data[:, 2].tolist()
                result["n_points"] = len(data)
        except:
            pass

    return result
