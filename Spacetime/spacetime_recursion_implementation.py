"""
Spacetime Recursion Core Engine
"""

from src.experiment_runner import build_default_parameters, run_spacetime_experiments


def _format_record(record):
    """Return a human-readable description of an experiment record."""

    return (
        f"domain={record.domain} | sigma={record.sigma:.3f} | tau={record.tau:.3f} | "
        f"PCE={'ok' if record.pce_compliant else 'violated'} | "
        f"torsion_force={record.torsion_force:.3f} | "
        f"lambda4={'yes' if record.lambda_4_detected else 'no'}"
    )


def run_spacetime_engine():
    print("🚀 Running the Spacetime Recursion Engine...")
    records = run_spacetime_experiments(build_default_parameters())

    for record in records:
        print(_format_record(record))
