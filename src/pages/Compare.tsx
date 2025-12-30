import React, { useState, useEffect } from 'react';
import { fetchModels, fetchNote, saveNote } from '../services/api';
import { ModelData } from '../types';
import { useComparisonData } from '../components/compare/useComparisonData';
import { SideBySideView } from '../components/compare/views/SideBySideView';
import { SliderView } from '../components/compare/views/SliderView';
import { ProximityView } from '../components/compare/views/ProximityView';

type ViewMode = 'side-by-side' | 'slider' | 'proximity';

export default function Compare() {
  const [models, setModels] = useState<ModelData[]>([]);
  // Support N models
  const [selectedModelIds, setSelectedModelIds] = useState<string[]>([]);

  const [viewMode, setViewMode] = useState<ViewMode>('side-by-side');
  const [selectedPrompt, setSelectedPrompt] = useState<string>('All');
  const [selectedSeed, setSelectedSeed] = useState<string>('All');

  const [note, setNote] = useState<string>('');
  const [noteSaving, setNoteSaving] = useState(false);

  // Load models on mount
  useEffect(() => {
    fetchModels().then(data => {
      setModels(data);
      // Default to first 2 models if available
      if (data.length >= 2) {
        setSelectedModelIds([data[0].id, data[1].id]);
      } else if (data.length === 1) {
        setSelectedModelIds([data[0].id]);
      }
    });
  }, []);

  // Use Custom Hook for Data
  const {
    commonPrompts,
    commonSeeds,
    getImagesForSelection,
    loadingMap
  } = useComparisonData(models, selectedModelIds);

  // Auto-select first common prompt/seed if current selection is invalid
  useEffect(() => {
     // If we have models selected and common data exists
     if (selectedModelIds.length > 0) {
        const promptValid = selectedPrompt !== 'All' && commonPrompts.includes(selectedPrompt);
        const seedValid = selectedSeed !== 'All' && commonSeeds.map(String).includes(selectedSeed);

        if (!promptValid && commonPrompts.length > 0) {
            setSelectedPrompt(commonPrompts[0]);
        }

        if (!seedValid && commonSeeds.length > 0) {
            setSelectedSeed(commonSeeds[0].toString());
        }
     }
  }, [commonPrompts, commonSeeds, selectedModelIds.length]);

  // Get current images for the view
  const currentImages = getImagesForSelection(selectedPrompt, selectedSeed);
  const currentModelNames = selectedModelIds.map(id => models.find(m => m.id === id)?.name || id);

  // Note management (Composite key)
  // Key needs to be sorted to be consistent regardless of order?
  // Let's sort IDs in the key.
  const noteId = selectedModelIds.length > 0 && selectedPrompt !== 'All' && selectedSeed !== 'All'
    ? `compare:${[...selectedModelIds].sort().join(':')}:${currentImages[0]?.prompt_idx || '0'}:${selectedSeed}`
    : null;

  useEffect(() => {
    if (noteId) {
      setNote('');
      fetchNote(noteId).then(data => {
        if (data && data.content) {
            setNote(data.content);
        }
      }).catch((err) => {
        console.debug(`Failed to fetch note for ${noteId}:`, err);
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

  const toggleModelSelection = (id: string) => {
      setSelectedModelIds(prev => {
          if (prev.includes(id)) {
              // Don't allow deselecting the last one? Or allow empty?
              return prev.filter(m => m !== id);
          } else {
              return [...prev, id];
          }
      });
  };

  return (
    <div className="max-w-[1800px] mx-auto px-6 py-8">
      <div className="flex flex-col gap-6">
        <h2 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
          Model Comparison
        </h2>

        {/* Top Controls Area */}
        <div className="bg-white dark:bg-slate-800 p-6 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700 space-y-6">

            {/* 1. Model Selection (Chips) */}
            <div className="flex flex-col gap-2">
                <label className="text-sm font-semibold text-slate-500 uppercase tracking-wider">Selected Models ({selectedModelIds.length})</label>
                <div className="flex flex-wrap gap-2">
                    {models.map(m => {
                        const isSelected = selectedModelIds.includes(m.id);
                        return (
                            <button
                                key={m.id}
                                onClick={() => toggleModelSelection(m.id)}
                                className={`px-3 py-1.5 rounded-full text-sm font-medium transition-all border ${
                                    isSelected
                                    ? 'bg-indigo-100 text-indigo-700 border-indigo-200 dark:bg-indigo-900/50 dark:text-indigo-300 dark:border-indigo-700'
                                    : 'bg-slate-50 text-slate-600 border-slate-200 hover:bg-slate-100 dark:bg-slate-700/50 dark:text-slate-400 dark:border-slate-600'
                                }`}
                            >
                                {m.name}
                                {isSelected && <span className="ml-2">✓</span>}
                            </button>
                        );
                    })}
                </div>
            </div>

            {/* 2. Parameters & View Mode */}
            <div className="flex flex-wrap gap-6 items-end border-t border-slate-100 dark:border-slate-700 pt-6">
                 {/* Prompt Selector */}
                <div className="flex flex-col gap-2 flex-[2] min-w-[300px]">
                    <label className="text-sm font-semibold text-slate-600 dark:text-slate-400">Common Prompt</label>
                    <select
                        className="w-full px-3 py-2 bg-slate-50 dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-md truncate"
                        value={selectedPrompt}
                        onChange={e => setSelectedPrompt(e.target.value)}
                        disabled={commonPrompts.length === 0}
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
                        disabled={commonSeeds.length === 0}
                    >
                        {commonSeeds.length === 0 && <option value="All">None</option>}
                        {commonSeeds.map(s => (
                            <option key={s} value={s}>{s}</option>
                        ))}
                    </select>
                </div>

                {/* View Mode Switcher */}
                <div className="flex bg-slate-100 dark:bg-slate-900 rounded-lg p-1 border border-slate-200 dark:border-slate-700 ml-auto">
                    {(['side-by-side', 'slider', 'proximity'] as const).map(mode => (
                        <button
                            key={mode}
                            onClick={() => setViewMode(mode)}
                            disabled={mode === 'slider' && selectedModelIds.length !== 2}
                            className={`px-4 py-2 rounded-md text-sm font-medium transition-all capitalize ${
                                viewMode === mode
                                ? 'bg-white dark:bg-slate-700 shadow-sm text-indigo-600 dark:text-indigo-400'
                                : 'text-slate-500 hover:text-slate-700 dark:text-slate-400 disabled:opacity-40 disabled:cursor-not-allowed'
                            }`}
                            title={mode === 'slider' && selectedModelIds.length !== 2 ? "Slider requires exactly 2 models" : ""}
                        >
                            {mode.replace(/-/g, ' ')}
                        </button>
                    ))}
                </div>
            </div>
        </div>

        {/* Visualization Area */}
        <div className="bg-slate-100 dark:bg-slate-900/50 rounded-xl border border-slate-200 dark:border-slate-700 min-h-[500px] flex flex-col relative overflow-hidden">
            {selectedModelIds.length === 0 ? (
                <div className="flex-1 flex items-center justify-center text-slate-500">
                    Select at least one model to compare.
                </div>
            ) : commonPrompts.length === 0 ? (
                <div className="flex-1 flex items-center justify-center text-slate-500">
                    Selected models have no common prompts. Try selecting different models.
                </div>
            ) : (
                <div className="flex-1">
                    {viewMode === 'side-by-side' && (
                        <SideBySideView images={currentImages} modelNames={currentModelNames} />
                    )}
                    {viewMode === 'slider' && (
                        <SliderView images={currentImages} modelNames={currentModelNames} />
                    )}
                    {viewMode === 'proximity' && (
                        <ProximityView images={currentImages} modelNames={currentModelNames} />
                    )}
                </div>
            )}

            {/* Prompt Text Footer */}
            {selectedPrompt !== 'All' && (
                 <div className="p-4 bg-white/50 dark:bg-black/50 backdrop-blur-sm border-t border-slate-200 dark:border-slate-700 text-center">
                    <p className="font-mono text-sm text-slate-700 dark:text-slate-300">{selectedPrompt}</p>
                 </div>
            )}
        </div>

        {/* Notes Section */}
        {noteId && (
            <div className="bg-white dark:bg-slate-800 p-6 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700">
                <div className="flex justify-between items-center mb-4">
                     <h3 className="text-lg font-semibold text-slate-800 dark:text-slate-100">
                        Notes for this Comparison
                     </h3>
                     <span className="text-xs text-slate-400 font-mono bg-slate-100 dark:bg-slate-900 px-2 py-1 rounded">
                        {currentModelNames.length} Models • Seed {selectedSeed}
                     </span>
                </div>

                <div className="flex flex-col gap-3">
                    <textarea
                        className="w-full h-32 p-3 bg-slate-50 dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded-md focus:ring-2 focus:ring-indigo-500 focus:outline-none transition-all"
                        placeholder="Add your thoughts on this specific comparison..."
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
