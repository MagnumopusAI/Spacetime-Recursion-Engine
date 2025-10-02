import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Spacetime'))

from src.deterministic_topological_arbitrage import (
    ArbitrageParameters,
    DeterministicTopologicalArbitrageSystem,
    LsprExecutionEngine,
    MarketEfficiencyKalmanFilter,
    SmugPceSolver,
    TdaSignalGenerationEngine,
)


def test_kalman_filter_prediction_and_update():
    filter_ = MarketEfficiencyKalmanFilter(0.05, 0.02, state=1.2, covariance=0.5)
    predicted = filter_.predict(0.1)
    assert predicted < 1.2
    updated_state, gain = filter_.update(0.8)
    assert 0.0 < gain < 1.0
    assert abs(updated_state - filter_.state) < 1e-12


def test_solver_respects_pce_and_returns_positive_signal():
    params = ArbitrageParameters(drive=3.0, damping=0.8, resistance=0.5)
    solver = SmugPceSolver()
    signal = solver.solve_arbitrage_signal(params)
    assert signal >= 0.0
    position = solver.determine_position_size(signal)
    limit = solver.limit_of_arbitrage(signal)
    assert position >= limit >= 0.0


def test_execution_schedule_preserves_total_signal():
    engine = LsprExecutionEngine()
    schedule = engine.generate_schedule(2.5, [0.1, 0.4, 0.3, 0.2])
    assert np.isclose(sum(schedule), 2.5)
    slippage = engine.measure_slippage(schedule, schedule)
    assert np.isclose(slippage, 0.0)


def test_full_system_cycle_outputs_consistent_dictionary():
    kalman = MarketEfficiencyKalmanFilter(0.05, 0.05, state=1.0, covariance=0.1)
    sensor = TdaSignalGenerationEngine(kalman, persistence_gain=0.2)
    solver = SmugPceSolver()
    actuator = LsprExecutionEngine()
    system = DeterministicTopologicalArbitrageSystem(sensor, solver, actuator)

    bids = [(100.0, 5.0), (99.9, 2.5), (99.8, 1.0)]
    asks = [(100.1, 3.0), (100.2, 1.5)]
    result = system.sense_and_trade(bids, asks, realised_decay=0.9, delta_t=0.05, micro_price_series=[0.2, 0.3, 0.5])

    assert {'signal', 'schedule', 'position_size', 'limit_of_arbitrage', 'kalman_gain', 'parameters'} <= set(result)
    assert len(result['schedule']) == 3
    assert np.isclose(sum(result['schedule']), result['signal'])
