import React, { useState, useEffect, useRef } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
} from 'recharts';

/**
 * User-facing demo component for exploring torsion-induced CP violation.
 * The logic parallels the formulas implemented in ``Spacetime.src.cp_violation``.
 */
export default function TorsionCPSimulation() {
  const [simulationData, setSimulationData] = useState([]);
  const [neutrinoData, setNeutrinoData] = useState([]);
  const [torsionStrength, setTorsionStrength] = useState(0.1);
  const [running, setRunning] = useState(false);
  const [detectionProb, setDetectionProb] = useState(0);
  const intervalRef = useRef(null);

  const calculateTorsionCP = (distance, energy, gT) => {
    const L = distance;
    const E = energy;
    const dm31 = 2.5e-3;
    const delta_torsion = gT * Math.PI * Math.sin(L / 100) * Math.cos(E / 2);
    const delta_total = (197 * Math.PI) / 180 + delta_torsion;
    const prob_standard = 0.5 * (1 - Math.cos((1.27 * dm31 * L) / E));
    const torsion_enhancement = gT * Math.sin(delta_total) * Math.exp(-L / 1000);
    return {
      distance: L,
      energy: E,
      probability: prob_standard + torsion_enhancement,
      torsion_phase: delta_torsion,
      cp_asymmetry: 2 * torsion_enhancement,
    };
  };

  const runSimulation = () => {
    if (running) return;
    setRunning(true);
    const data = [];
    const neutrinoOsc = [];
    for (let i = 0; i < 100; i++) {
      const distance = 50 + i * 10;
      const energy = 0.5 + i * 0.05;
      const result = calculateTorsionCP(distance, energy, torsionStrength);
      data.push({
        distance,
        standard_cp: Math.sin((197 * Math.PI) / 180) * 0.1,
        torsion_cp: result.cp_asymmetry,
        total_asymmetry: Math.sin((197 * Math.PI) / 180) * 0.1 + result.cp_asymmetry,
        detection_significance: Math.abs(result.cp_asymmetry) > 0.01 ? 3.5 : 1.2,
      });
      neutrinoOsc.push({
        energy,
        oscillation_prob: result.probability,
        torsion_phase: (result.torsion_phase * 180) / Math.PI,
        asymmetry: result.cp_asymmetry * 100,
      });
    }
    setSimulationData(data);
    setNeutrinoData(neutrinoOsc);
    const significantPoints = data.filter((d) => d.detection_significance > 3).length;
    setDetectionProb(significantPoints / data.length);
    setTimeout(() => setRunning(false), 1000);
  };

  useEffect(() => {
    runSimulation();
  }, [torsionStrength]);

  const neutronEDMData = [
    { experiment: 'Current Limit', value: 1.8e-26, error: 0.2e-26 },
    { experiment: 'Torsion Prediction', value: 5.2e-28, error: 1.1e-28 },
    { experiment: 'Future Sensitivity', value: 1e-28, error: 0 },
  ];

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-slate-900 to-indigo-900 text-white rounded-lg">
      {/* Omitted rendering details for brevity */}
    </div>
  );
}
