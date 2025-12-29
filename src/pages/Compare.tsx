import React, { useState, useEffect, useRef } from 'react';
import { fetchModels, fetchModelOutputs, fetchNote, saveNote } from '../services/api';
import { ModelData, ModelOutput } from '../types';

export default function Compare() {
  const [models, setModels] = useState<ModelData[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  // Selection state
  const [leftModelId, setLeftModelId] = useState<string>('');
  const [rightModelId, setRightModelId] = useState<string>('');
  const [selectedPrompt, setSelectedPrompt] = useState<string>('');
  const [selectedSeed, setSelectedSeed] = useState<string>('');

  // Data state
  const [leftOutputs, setLeftOutputs] = useState<ModelOutput[]>([]);
  const [rightOutputs, setRightOutputs] = useState<ModelOutput[]>([]);

  // UI state
  const [mode, setMode] = useState<'side-by-side' | 'slider'>('side-by-side');
  const [sliderPosition, setSliderPosition] = useState<number>(50);

  // Notes
  const [notes, setNotes] = useState<string>('');
  const [savingNotes, setSavingNotes] = useState<boolean>(false);

  const containerRef = useRef<HTMLDivElement>(null);

  // Load models on mount
  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      const data = await fetchModels();
      setModels(data);
      if (data.length >= 2) {
        setLeftModelId(data[0].id);
        setRightModelId(data[1].id);
      } else if (data.length === 1) {
        setLeftModelId(data[0].id);
      }
      setLoading(false);
    } catch (err) {
      console.error("Failed to load models", err);
      setLoading(false);
    }
  };

  // Fetch outputs when models change
  useEffect(() => {
    if (leftModelId) loadOutputs(leftModelId, setLeftOutputs);
  }, [leftModelId]);

  useEffect(() => {
    if (rightModelId) loadOutputs(rightModelId, setRightOutputs);
  }, [rightModelId]);

  const loadOutputs = async (modelId: string, setOutputs: React.Dispatch<React.SetStateAction<ModelOutput[]>>) => {
    try {
      const data = await fetchModelOutputs(modelId);
      setOutputs(data);
    } catch (err) {
      console.error(`Failed to load outputs for ${modelId}`, err);
    }
  };

  // Derive common prompts and seeds
  // We want to find prompts/seeds that exist in BOTH models if possible,
  // or at least available in one of them.
  // Actually, for comparison, it's best to show intersection, but we can show union and handle missing images.

  const allPrompts = Array.from(new Set([
    ...leftOutputs.map(o => o.prompt),
    ...rightOutputs.map(o => o.prompt)
  ])).sort();

  const allSeeds = Array.from(new Set([
    ...leftOutputs.map(o => o.seed),
    ...rightOutputs.map(o => o.seed)
  ])).sort((a, b) => a - b);

  // Default selection if not set
  useEffect(() => {
    if (!selectedPrompt && allPrompts.length > 0) {
      setSelectedPrompt(allPrompts[0]);
    }
    if (!selectedSeed && allSeeds.length > 0) {
      setSelectedSeed(allSeeds[0].toString());
    }
  }, [allPrompts, allSeeds, selectedPrompt, selectedSeed]);

  // Load Notes for the pair
  useEffect(() => {
    if (leftModelId && rightModelId) {
      const loadNotesForPair = async () => {
        try {
          // Sort IDs to ensure stable key regardless of Left/Right order
          const ids = [leftModelId, rightModelId].sort().join('_vs_');
          const data = await fetchNote(ids);
          setNotes(data.content || '');
        } catch (err) {
          console.error("Failed to load notes", err);
        }
      };
      loadNotesForPair();
    }
  }, [leftModelId, rightModelId]);

  const handleSaveNotes = async () => {
    if (!leftModelId || !rightModelId) return;
    setSavingNotes(true);
    try {
      const ids = [leftModelId, rightModelId].sort().join('_vs_');
      await saveNote(ids, notes);
    } catch (err) {
      console.error("Failed to save notes", err);
    } finally {
      setSavingNotes(false);
    }
  };

  // Get current images
  const getCurrentImage = (outputs: ModelOutput[]) => {
    return outputs.find(o =>
      o.prompt === selectedPrompt &&
      o.seed.toString() === selectedSeed
    );
  };

  const leftImage = getCurrentImage(leftOutputs);
  const rightImage = getCurrentImage(rightOutputs);

  const getImageUrl = (img: ModelOutput | undefined) => {
      if (!img) return null;
      return `${import.meta.env.VITE_API_BASE?.replace('/api', '') || 'http://localhost:8000'}${img.url}`;
  };

  const leftUrl = getImageUrl(leftImage);
  const rightUrl = getImageUrl(rightImage);

  const [containerWidth, setContainerWidth] = useState<number>(0);

  useEffect(() => {
      const updateWidth = () => {
          if (containerRef.current) {
              setContainerWidth(containerRef.current.clientWidth);
          }
      };

      // Initial check
      updateWidth();

      // Resize observer
      if (containerRef.current) {
          const observer = new ResizeObserver(updateWidth);
          observer.observe(containerRef.current);
          return () => observer.disconnect();
      }
  }, [mode]); // Re-run when mode changes to slider

  // Slider logic
  const handleMouseMove = (e: React.MouseEvent | React.TouchEvent) => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
      const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
      const percentage = (x / rect.width) * 100;
      setSliderPosition(percentage);
  };

  const [isDragging, setIsDragging] = useState(false);

  return (
    <div className="max-w-[1800px] mx-auto px-6 py-8">
      <div className="flex flex-col gap-6">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
            <h2 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
            Model Comparison
            </h2>

            <div className="flex bg-slate-100 dark:bg-slate-800 p-1 rounded-lg">
                <button
                    onClick={() => setMode('side-by-side')}
                    className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${mode === 'side-by-side' ? 'bg-white dark:bg-slate-700 shadow text-indigo-600 dark:text-indigo-400' : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'}`}
                >
                    Side-by-Side
                </button>
                <button
                    onClick={() => setMode('slider')}
                    className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${mode === 'slider' ? 'bg-white dark:bg-slate-700 shadow text-indigo-600 dark:text-indigo-400' : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'}`}
                >
                    Slider
                </button>
            </div>
        </div>

        {/* Controls */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 bg-white dark:bg-slate-800 p-4 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700">
            {/* Model A */}
            <div className="flex flex-col gap-1">
                <label className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Model A (Left)</label>
                <select
                    className="px-3 py-2 bg-slate-50 dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-md text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    value={leftModelId}
                    onChange={(e) => setLeftModelId(e.target.value)}
                >
                    <option value="">Select Model</option>
                    {models.map(m => (
                        <option key={m.id} value={m.id}>{m.name}</option>
                    ))}
                </select>
            </div>

            {/* Model B */}
            <div className="flex flex-col gap-1">
                <label className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Model B (Right)</label>
                <select
                    className="px-3 py-2 bg-slate-50 dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-md text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    value={rightModelId}
                    onChange={(e) => setRightModelId(e.target.value)}
                >
                    <option value="">Select Model</option>
                    {models.map(m => (
                        <option key={m.id} value={m.id}>{m.name}</option>
                    ))}
                </select>
            </div>

            {/* Prompt Selector */}
            <div className="flex flex-col gap-1">
                <label className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Prompt</label>
                <select
                    className="px-3 py-2 bg-slate-50 dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-md text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    value={selectedPrompt}
                    onChange={(e) => setSelectedPrompt(e.target.value)}
                    disabled={allPrompts.length === 0}
                >
                    <option value="">Select Prompt</option>
                    {allPrompts.map((p, i) => (
                        <option key={i} value={p}>{p.substring(0, 50)}{p.length > 50 ? '...' : ''}</option>
                    ))}
                </select>
            </div>

            {/* Seed Selector */}
            <div className="flex flex-col gap-1">
                <label className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Seed</label>
                <select
                    className="px-3 py-2 bg-slate-50 dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-md text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    value={selectedSeed}
                    onChange={(e) => setSelectedSeed(e.target.value)}
                    disabled={allSeeds.length === 0}
                >
                    <option value="">Select Seed</option>
                    {allSeeds.map((s) => (
                        <option key={s} value={s.toString()}>{s}</option>
                    ))}
                </select>
            </div>
        </div>

        {/* Full Prompt Text Display */}
        {selectedPrompt && (
             <div className="bg-slate-50 dark:bg-slate-800/50 p-3 rounded-lg border border-slate-200 dark:border-slate-700 text-sm text-slate-600 dark:text-slate-400 font-mono break-words">
                 {selectedPrompt}
             </div>
        )}

        {/* View Area */}
        <div className="bg-slate-100 dark:bg-slate-900 rounded-xl p-4 min-h-[500px] flex items-center justify-center relative select-none">
            {mode === 'side-by-side' && (
                <div className="grid grid-cols-2 gap-4 w-full h-full">
                    {/* Left Image */}
                    <div className="flex flex-col items-center gap-2">
                        <div className="relative w-full aspect-square bg-white dark:bg-slate-800 rounded-lg overflow-hidden shadow-sm border border-slate-200 dark:border-slate-700">
                             {leftUrl ? (
                                <img src={leftUrl} alt="Model A" className="w-full h-full object-contain" />
                             ) : (
                                <div className="w-full h-full flex items-center justify-center text-slate-400">No Image</div>
                             )}
                             <div className="absolute top-2 left-2 bg-black/60 text-white text-xs px-2 py-1 rounded backdrop-blur-sm">
                                 {models.find(m => m.id === leftModelId)?.name || 'Model A'}
                             </div>
                        </div>
                    </div>

                    {/* Right Image */}
                    <div className="flex flex-col items-center gap-2">
                        <div className="relative w-full aspect-square bg-white dark:bg-slate-800 rounded-lg overflow-hidden shadow-sm border border-slate-200 dark:border-slate-700">
                             {rightUrl ? (
                                <img src={rightUrl} alt="Model B" className="w-full h-full object-contain" />
                             ) : (
                                <div className="w-full h-full flex items-center justify-center text-slate-400">No Image</div>
                             )}
                             <div className="absolute top-2 left-2 bg-black/60 text-white text-xs px-2 py-1 rounded backdrop-blur-sm">
                                 {models.find(m => m.id === rightModelId)?.name || 'Model B'}
                             </div>
                        </div>
                    </div>
                </div>
            )}

            {mode === 'slider' && (
                <div
                    ref={containerRef}
                    className="relative w-full max-w-4xl aspect-square bg-white dark:bg-slate-800 rounded-lg overflow-hidden shadow-lg border border-slate-200 dark:border-slate-700 cursor-ew-resize"
                    onMouseDown={() => setIsDragging(true)}
                    onMouseUp={() => setIsDragging(false)}
                    onMouseLeave={() => setIsDragging(false)}
                    onMouseMove={(e) => {
                        if (isDragging) handleMouseMove(e);
                    }}
                    onTouchStart={() => setIsDragging(true)}
                    onTouchEnd={() => setIsDragging(false)}
                    onTouchMove={(e) => {
                        if (isDragging) handleMouseMove(e);
                    }}
                    onClick={handleMouseMove}
                >
                    {/* Background Image (Right) */}
                     {rightUrl ? (
                        <img
                            src={rightUrl}
                            alt="Model B"
                            className="absolute inset-0 w-full h-full object-contain"
                        />
                     ) : (
                         <div className="absolute inset-0 flex items-center justify-center text-slate-400">Right Image Missing</div>
                     )}

                    {/* Foreground Image (Left) - Clip Path */}
                    <div
                        className="absolute inset-0 overflow-hidden border-r-2 border-white/80 shadow-[2px_0_5px_rgba(0,0,0,0.3)]"
                        style={{ width: `${sliderPosition}%` }}
                    >
                         {leftUrl ? (
                            <img
                                src={leftUrl}
                                alt="Model A"
                                className="absolute inset-0 w-full h-full object-cover object-left"
                                style={{
                                    width: containerWidth || '100vw', // Fallback to viewport width to prevent squish before measure
                                    maxWidth: 'none' // Important to prevent squishing
                                }}
                            />
                         ) : (
                            <div className="w-full h-full bg-slate-200 dark:bg-slate-700 flex items-center justify-center text-slate-400">Left Image Missing</div>
                         )}
                    </div>

                    {/* Labels */}
                    <div className="absolute top-4 left-4 bg-black/60 text-white text-xs px-2 py-1 rounded backdrop-blur-sm pointer-events-none">
                        {models.find(m => m.id === leftModelId)?.name || 'Model A'}
                    </div>
                    <div className="absolute top-4 right-4 bg-black/60 text-white text-xs px-2 py-1 rounded backdrop-blur-sm pointer-events-none">
                        {models.find(m => m.id === rightModelId)?.name || 'Model B'}
                    </div>

                    {/* Handle */}
                    <div
                        className="absolute top-0 bottom-0 w-1 bg-white cursor-ew-resize shadow-[0_0_10px_rgba(0,0,0,0.5)]"
                        style={{ left: `${sliderPosition}%` }}
                    >
                        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-white rounded-full shadow-lg flex items-center justify-center text-slate-600">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="15 18 9 12 15 6"></polyline></svg>
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="9 18 15 12 9 6"></polyline></svg>
                        </div>
                    </div>
                </div>
            )}
        </div>

        {/* Notes Section */}
        <div className="bg-white dark:bg-slate-800 p-4 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700">
            <div className="flex justify-between items-center mb-2">
                <h3 className="text-lg font-semibold text-slate-800 dark:text-slate-100">Comparison Notes</h3>
                <button
                    onClick={handleSaveNotes}
                    disabled={savingNotes}
                    className="px-4 py-1.5 bg-indigo-600 text-white rounded-md text-sm font-medium hover:bg-indigo-700 disabled:opacity-50 transition-colors"
                >
                    {savingNotes ? 'Saving...' : 'Save Notes'}
                </button>
            </div>
            <textarea
                className="w-full h-32 p-3 bg-slate-50 dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-md text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                placeholder="Add notes about these two models..."
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
            />
            <p className="text-xs text-slate-500 mt-2">
                Notes are saved for this specific pair of models ({models.find(m => m.id === leftModelId)?.name} vs {models.find(m => m.id === rightModelId)?.name}).
            </p>
        </div>
      </div>
    </div>
  );
}
