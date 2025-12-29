import React, { useState, useEffect, useRef } from 'react';
import { fetchModels, fetchModelOutputs, fetchNote, saveNote } from '../services/api';
import { ModelData, ModelOutput } from '../types';

export default function Compare() {
  const [models, setModels] = useState<ModelData[]>([]);
  const [modelA, setModelA] = useState<string>('');
  const [modelB, setModelB] = useState<string>('');

  const [outputsA, setOutputsA] = useState<ModelOutput[]>([]);
  const [outputsB, setOutputsB] = useState<ModelOutput[]>([]);

  const [selectedPrompt, setSelectedPrompt] = useState<string>('All');
  const [selectedSeed, setSelectedSeed] = useState<string>('All');

  const [note, setNote] = useState<string>('');
  const [noteSaving, setNoteSaving] = useState(false);

  const [viewMode, setViewMode] = useState<'side-by-side' | 'slider'>('side-by-side');
  const [sliderPosition, setSliderPosition] = useState<number>(50);

  // Loading states
  const [loadingA, setLoadingA] = useState(false);
  const [loadingB, setLoadingB] = useState(false);

  // Initial load
  useEffect(() => {
    fetchModels().then(data => {
      setModels(data);
      if (data.length >= 2) {
        setModelA(data[0].id);
        setModelB(data[1].id);
      } else if (data.length === 1) {
        setModelA(data[0].id);
      }
    });
  }, []);

  // Fetch outputs when models change
  useEffect(() => {
    if (modelA) {
      setLoadingA(true);
      fetchModelOutputs(modelA)
        .then(setOutputsA)
        .finally(() => setLoadingA(false));
    } else {
      setOutputsA([]);
    }
  }, [modelA]);

  useEffect(() => {
    if (modelB) {
      setLoadingB(true);
      fetchModelOutputs(modelB)
        .then(setOutputsB)
        .finally(() => setLoadingB(false));
    } else {
      setOutputsB([]);
    }
  }, [modelB]);

  // Compute common prompts and seeds for filter dropdowns
  // We want to navigate between "common ground" to compare apples to apples
  const promptsA = new Set(outputsA.map(o => o.prompt));
  const promptsB = new Set(outputsB.map(o => o.prompt));
  const commonPrompts = Array.from(promptsA).filter(p => promptsB.has(p));

  const seedsA = new Set(outputsA.map(o => o.seed));
  const seedsB = new Set(outputsB.map(o => o.seed));
  const commonSeeds = Array.from(seedsA).filter(s => seedsB.has(s)).sort((a, b) => a - b);

  // If currently selected filters are invalid for new model selection, reset them
  // Logic: Try to keep selection if possible, else default to first common, else "All"
  useEffect(() => {
    // This effect runs when outputs update.
    // We should ensure selectedPrompt and selectedSeed are valid or "All"
    // Actually, "All" might be confusing for 1:1 comparison.
    // Let's default to the first common prompt/seed if available and current selection is invalid.

    if (outputsA.length > 0 && outputsB.length > 0) {
        if (selectedPrompt === 'All' && commonPrompts.length > 0) {
            setSelectedPrompt(commonPrompts[0]);
        }
        if (selectedSeed === 'All' && commonSeeds.length > 0) {
            setSelectedSeed(commonSeeds[0].toString());
        }
    }
  }, [outputsA, outputsB]); // Don't include commonPrompts/commonSeeds arrays to avoid loops, just depend on data

  // Find the specific images to compare
  const imageA = outputsA.find(o => o.prompt === selectedPrompt && o.seed.toString() === selectedSeed);
  const imageB = outputsB.find(o => o.prompt === selectedPrompt && o.seed.toString() === selectedSeed);

  // Note management
  const noteId = imageA && imageB ? `compare:${modelA}:${modelB}:${imageA.prompt_idx}:${imageA.seed}` : null;

  useEffect(() => {
    if (noteId) {
      setNote(''); // Clear while loading
      fetchNote(noteId).then(data => {
        if (data && data.content) {
            setNote(data.content);
        }
      }).catch(() => {
          // No note found, ignore
      });
    } else {
        setNote('');
    }
  }, [noteId]);

  const handleSaveNote = async () => {
    if (!noteId) return;
    setNoteSaving(true);
    try {
        await saveNote(noteId, { content: note, timestamp: new Date().toISOString() });
    } catch (e) {
        console.error("Failed to save note", e);
    } finally {
        setNoteSaving(false);
    }
  };

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSliderPosition(Number(e.target.value));
  };

  // Helper for URLs
  const getUrl = (url: string) => `${import.meta.env.VITE_API_BASE?.replace('/api', '') || 'http://localhost:8000'}${url}`;

  return (
    <div className="max-w-[1800px] mx-auto px-6 py-8">
      <div className="flex flex-col gap-6">
        <h2 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
          Model Comparison
        </h2>

        {/* Controls */}
        <div className="bg-white dark:bg-slate-800 p-6 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700 flex flex-wrap gap-6 items-end">
            {/* Model A */}
            <div className="flex flex-col gap-2 flex-1 min-w-[200px]">
                <label className="text-sm font-semibold text-indigo-600 dark:text-indigo-400">Model A (Left)</label>
                <select
                    className="w-full px-3 py-2 bg-slate-50 dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-md"
                    value={modelA}
                    onChange={e => setModelA(e.target.value)}
                >
                    {models.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
                </select>
            </div>

            {/* Model B */}
            <div className="flex flex-col gap-2 flex-1 min-w-[200px]">
                <label className="text-sm font-semibold text-pink-600 dark:text-pink-400">Model B (Right)</label>
                <select
                    className="w-full px-3 py-2 bg-slate-50 dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-md"
                    value={modelB}
                    onChange={e => setModelB(e.target.value)}
                >
                    {models.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
                </select>
            </div>

            {/* Prompt Selector */}
            <div className="flex flex-col gap-2 flex-[2] min-w-[300px]">
                <label className="text-sm font-semibold text-slate-600 dark:text-slate-400">Common Prompt</label>
                <select
                    className="w-full px-3 py-2 bg-slate-50 dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-md truncate"
                    value={selectedPrompt}
                    onChange={e => setSelectedPrompt(e.target.value)}
                >
                    {commonPrompts.length === 0 && <option value="All">No common prompts found</option>}
                    {commonPrompts.map((p, i) => (
                        <option key={i} value={p}>{p.substring(0, 80)}{p.length > 80 ? '...' : ''}</option>
                    ))}
                </select>
            </div>

            {/* Seed Selector */}
            <div className="flex flex-col gap-2 flex-none w-[120px]">
                <label className="text-sm font-semibold text-slate-600 dark:text-slate-400">Common Seed</label>
                <select
                    className="w-full px-3 py-2 bg-slate-50 dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-md"
                    value={selectedSeed}
                    onChange={e => setSelectedSeed(e.target.value)}
                >
                    {commonSeeds.length === 0 && <option value="All">None</option>}
                    {commonSeeds.map(s => (
                        <option key={s} value={s}>{s}</option>
                    ))}
                </select>
            </div>

             {/* View Mode Toggle */}
             <div className="flex bg-slate-100 dark:bg-slate-900 rounded-lg p-1 border border-slate-200 dark:border-slate-700">
                <button
                    onClick={() => setViewMode('side-by-side')}
                    className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${viewMode === 'side-by-side' ? 'bg-white dark:bg-slate-700 shadow-sm text-indigo-600 dark:text-indigo-400' : 'text-slate-500 hover:text-slate-700 dark:text-slate-400'}`}
                >
                    Side-by-Side
                </button>
                <button
                    onClick={() => setViewMode('slider')}
                    className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${viewMode === 'slider' ? 'bg-white dark:bg-slate-700 shadow-sm text-indigo-600 dark:text-indigo-400' : 'text-slate-500 hover:text-slate-700 dark:text-slate-400'}`}
                >
                    Slider
                </button>
            </div>
        </div>

        {/* Visualization Area */}
        <div className="bg-slate-100 dark:bg-slate-900/50 rounded-xl border border-slate-200 dark:border-slate-700 p-8 min-h-[500px] flex items-center justify-center relative">

            {(!imageA || !imageB) ? (
                <div className="text-center text-slate-500 dark:text-slate-400">
                    <p className="text-lg">Select matching prompts and seeds to compare.</p>
                    <p className="text-sm mt-2 opacity-70">
                        Model A: {outputsA.length} images | Model B: {outputsB.length} images
                    </p>
                </div>
            ) : (
                <div className="w-full h-full flex flex-col items-center">

                    {viewMode === 'side-by-side' ? (
                        <div className="grid grid-cols-2 gap-4 w-full h-full">
                            <div className="flex flex-col gap-2">
                                <div className="relative aspect-square bg-white dark:bg-black rounded-lg overflow-hidden border-2 border-indigo-200 dark:border-indigo-900/50 shadow-md">
                                    <img src={getUrl(imageA.url)} alt="Model A" className="w-full h-full object-contain" />
                                    <div className="absolute top-2 left-2 bg-indigo-600 text-white text-xs px-2 py-1 rounded shadow-sm opacity-80">Model A</div>
                                </div>
                            </div>
                            <div className="flex flex-col gap-2">
                                <div className="relative aspect-square bg-white dark:bg-black rounded-lg overflow-hidden border-2 border-pink-200 dark:border-pink-900/50 shadow-md">
                                    <img src={getUrl(imageB.url)} alt="Model B" className="w-full h-full object-contain" />
                                    <div className="absolute top-2 right-2 bg-pink-600 text-white text-xs px-2 py-1 rounded shadow-sm opacity-80">Model B</div>
                                </div>
                            </div>
                        </div>
                    ) : (
                        <div className="relative w-full max-w-[800px] aspect-square select-none overflow-hidden rounded-lg shadow-xl border border-slate-300 dark:border-slate-600 bg-black">
                             {/* Base Image (Model B - Right side logic usually) */}
                             <img
                                src={getUrl(imageB.url)}
                                alt="Model B"
                                className="absolute inset-0 w-full h-full object-contain"
                                draggable={false}
                             />

                             {/* Overlay Image (Model A - Left side) */}
                             <div
                                className="absolute inset-0 overflow-hidden"
                                style={{ width: `${sliderPosition}%`, borderRight: '2px solid white' }}
                             >
                                 <img
                                    src={getUrl(imageA.url)}
                                    alt="Model A"
                                    className="absolute top-0 left-0 max-w-none h-full"
                                    style={{ width: `${100 * (100/sliderPosition)}%` }} // Trick to keep aspect ratio? No, we need fixed width
                                    // Actually for object-contain comparison it's tricky if aspects differ.
                                    // Assuming same aspect for same prompt usually.
                                    // Better approach: Set width to container width.
                                 />
                                 {/*
                                    Fixing the overlay image sizing:
                                    If we use object-contain, the image might not fill the container.
                                    Ideally for slider comparison, we assume images are same dimensions.
                                    Let's try absolute positioning with width of the parent container.
                                 */}
                                 <img
                                    src={getUrl(imageA.url)}
                                    alt="Model A"
                                    className="absolute top-0 left-0 w-[800px] h-full object-contain" // Hardcoded width matches max-w?
                                    // React hooks needed to get container width?
                                    // Let's use % but inverse the crop.
                                    style={{ width: `${100 / (sliderPosition/100)}%`, maxWidth: 'none' }}
                                 />
                             </div>

                             {/* Slider Control */}
                             <input
                                type="range"
                                min="0"
                                max="100"
                                value={sliderPosition}
                                onChange={handleSliderChange}
                                className="absolute inset-0 w-full h-full opacity-0 cursor-ew-resize z-20"
                             />

                             {/* Labels */}
                             <div className="absolute bottom-4 left-4 bg-black/50 text-white px-2 py-1 rounded text-sm pointer-events-none z-10">Model A</div>
                             <div className="absolute bottom-4 right-4 bg-black/50 text-white px-2 py-1 rounded text-sm pointer-events-none z-10">Model B</div>

                             {/* Slider Handle Visual */}
                             <div
                                className="absolute top-0 bottom-0 w-1 bg-white shadow-[0_0_10px_rgba(0,0,0,0.5)] z-10 pointer-events-none"
                                style={{ left: `${sliderPosition}%` }}
                             >
                                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-white rounded-full flex items-center justify-center shadow-lg">
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" className="text-slate-800"><path d="m9 18 6-6-6-6"/></svg>
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" className="text-slate-800 rotate-180 absolute"><path d="m9 18 6-6-6-6"/></svg>
                                </div>
                             </div>
                        </div>
                    )}

                    <div className="mt-4 text-center text-slate-600 dark:text-slate-400 font-mono text-sm max-w-2xl">
                        {selectedPrompt}
                    </div>
                </div>
            )}
        </div>

        {/* Notes Section */}
        {imageA && imageB && (
            <div className="bg-white dark:bg-slate-800 p-6 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700">
                <h3 className="text-lg font-semibold text-slate-800 dark:text-slate-100 mb-4">Comparison Notes</h3>
                <div className="flex flex-col gap-3">
                    <textarea
                        className="w-full h-32 p-3 bg-slate-50 dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded-md focus:ring-2 focus:ring-indigo-500 focus:outline-none transition-all"
                        placeholder="Add your thoughts on this comparison..."
                        value={note}
                        onChange={(e) => setNote(e.target.value)}
                    />
                    <div className="flex justify-end">
                        <button
                            onClick={handleSaveNote}
                            disabled={noteSaving}
                            className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-md transition-colors disabled:opacity-50 flex items-center gap-2"
                        >
                            {noteSaving ? 'Saving...' : 'Save Note'}
                        </button>
                    </div>
                </div>
            </div>
        )}
      </div>
    </div>
  );
}
