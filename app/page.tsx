'use client';

import { useState } from 'react';
import SimulatorForm from '@/components/SimulatorForm';
import ResultsDisplay from '@/components/ResultsDisplay';
import { SimulationParams, SimulationResult, runMonteCarloSimulation, calculateNegativeReturnPercentages, APP_VERSION } from '@/lib/simulation';

export default function Home() {
  const [results, setResults] = useState<{
    simulationResult: SimulationResult;
    params: SimulationParams;
  } | null>(null);
  const [isSimulating, setIsSimulating] = useState(false);

  const handleRunSimulation = async (params: SimulationParams) => {
    setIsSimulating(true);
    
    // Run simulation in a setTimeout to allow UI to update
    setTimeout(() => {
      try {
        const simulationResult = runMonteCarloSimulation(params);
        setResults({ simulationResult, params });
      } catch (error) {
        console.error('Simulation error:', error);
        alert('An error occurred during simulation. Please check your parameters.');
      } finally {
        setIsSimulating(false);
      }
    }, 100);
  };

  return (
    <main className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">
          Retirement Portfolio Simulator
        </h1>
        <p className="text-lg text-gray-600 mb-8">
          with extreme events (fat tails, negative skew)
        </p>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Left sidebar - Controls */}
          <div className="lg:col-span-1">
            <SimulatorForm 
              onRunSimulation={handleRunSimulation}
              isSimulating={isSimulating}
            />
          </div>

          {/* Right side - Results */}
          <div className="lg:col-span-3">
            {isSimulating ? (
              <div className="bg-white rounded-lg shadow-md p-8 text-center">
                <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div>
                <p className="text-lg text-gray-600">Running simulation...</p>
                <p className="text-sm text-gray-500 mt-2">This may take a few seconds</p>
              </div>
            ) : results ? (
              <ResultsDisplay 
                results={results.simulationResult}
                params={results.params}
              />
            ) : (
              <div className="bg-white rounded-lg shadow-md p-8 text-center">
                <p className="text-lg text-gray-600">
                  Set your assumptions on the left and click &apos;Run Simulation&apos;
                </p>
              </div>
            )}
          </div>
        </div>

        <footer className="mt-8 text-center text-sm text-gray-500">
          App version: {APP_VERSION}
        </footer>
      </div>
    </main>
  );
}