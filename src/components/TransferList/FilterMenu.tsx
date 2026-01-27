import React, { useRef, useState, useEffect } from 'react';
import { Filter, X, Check } from 'lucide-react';

export interface FilterOptions {
  modelTypes: Set<string>;
  predictionTypes: Set<string>;
  sources: Set<string>;
}

interface FilterMenuProps {
  filters: FilterOptions;
  onChange: (filters: FilterOptions) => void;
  availableModelTypes: string[];
}

export function FilterMenu({ filters, onChange, availableModelTypes }: FilterMenuProps) {
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // Close when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen]);

  const toggleFilter = (category: keyof FilterOptions, value: string) => {
    const newFilters = { ...filters };
    const set = new Set(newFilters[category]);
    
    if (set.has(value)) {
      set.delete(value);
    } else {
      set.add(value);
    }
    
    newFilters[category] = set;
    onChange(newFilters);
  };

  const clearFilters = () => {
    onChange({
      modelTypes: new Set(),
      predictionTypes: new Set(),
      sources: new Set()
    });
    setIsOpen(false);
  };

  const isActive = filters.modelTypes.size > 0 || filters.predictionTypes.size > 0 || filters.sources.size > 0;

  return (
    <div className="relative" ref={menuRef}>
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className={`p-2 border rounded-lg transition-colors flex items-center gap-2 ${
          isActive 
            ? 'bg-blue-50 border-blue-200 text-blue-600 dark:bg-blue-900/30 dark:border-blue-700 dark:text-blue-400' 
            : 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-500 hover:text-blue-500 hover:border-blue-500'
        }`}
        title="Filter Options"
      >
        <Filter size={16} />
        {isActive && <span className="text-xs font-bold w-4 h-4 rounded-full bg-blue-500 text-white flex items-center justify-center -ml-1">!</span>}
      </button>

      {isOpen && (
        <div className="absolute right-0 top-full mt-2 w-64 bg-white dark:bg-slate-800 rounded-xl shadow-xl border border-slate-200 dark:border-slate-700 z-50 overflow-hidden transform origin-top-right animate-in fade-in zoom-in-95 duration-100">
          <div className="p-2 border-b border-slate-100 dark:border-slate-700/50 flex items-center justify-between">
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Filters</span>
            {isActive && (
              <button 
                onClick={clearFilters}
                className="text-[10px] text-red-500 hover:text-red-600 flex items-center gap-1"
              >
                <X size={10} /> Clear
              </button>
            )}
          </div>

          <div className="p-2 space-y-2 max-h-[350px] overflow-y-auto custom-scrollbar">
            {/* Model Types */}
            <div className="space-y-0.5">
              <label className="text-[10px] font-bold text-slate-400 px-2 uppercase opacity-70">Type</label>
              {availableModelTypes.map(type => (
                <button
                  key={type}
                  onClick={() => toggleFilter('modelTypes', type)}
                  className={`w-full flex items-center justify-between px-2 py-1 rounded-md text-xs transition-colors ${
                    filters.modelTypes.has(type)
                      ? 'bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300'
                      : 'hover:bg-slate-50 dark:hover:bg-slate-700/50 text-slate-600 dark:text-slate-300'
                  }`}
                >
                  <span>{type}</span>
                  {filters.modelTypes.has(type) && <Check size={12} />}
                </button>
              ))}
            </div>

            <div className="h-px bg-slate-100 dark:bg-slate-700/50" />

            {/* Prediction Types */}
            <div className="space-y-0.5">
              <label className="text-[10px] font-bold text-slate-400 px-2 uppercase opacity-70">Prediction</label>
              <button
                onClick={() => toggleFilter('predictionTypes', 'epsilon')}
                className={`w-full flex items-center justify-between px-2 py-1 rounded-md text-xs transition-colors ${
                  filters.predictionTypes.has('epsilon')
                    ? 'bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300'
                    : 'hover:bg-slate-50 dark:hover:bg-slate-700/50 text-slate-600 dark:text-slate-300'
                }`}
              >
                <span>Epsilon</span>
                {filters.predictionTypes.has('epsilon') && <Check size={12} />}
              </button>
              <button
                onClick={() => toggleFilter('predictionTypes', 'v_prediction')}
                className={`w-full flex items-center justify-between px-2 py-1 rounded-md text-xs transition-colors ${
                  filters.predictionTypes.has('v_prediction')
                    ? 'bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300'
                    : 'hover:bg-slate-50 dark:hover:bg-slate-700/50 text-slate-600 dark:text-slate-300'
                }`}
              >
                <span>V-Prediction</span>
                {filters.predictionTypes.has('v_prediction') && <Check size={12} />}
              </button>
               <button
                onClick={() => toggleFilter('predictionTypes', 'v_prediction_ztsnr')}
                className={`w-full flex items-center justify-between px-2 py-1 rounded-md text-xs transition-colors ${
                  filters.predictionTypes.has('v_prediction_ztsnr')
                    ? 'bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300'
                    : 'hover:bg-slate-50 dark:hover:bg-slate-700/50 text-slate-600 dark:text-slate-300'
                }`}
              >
                <span>V-Pred + Zero-SNR</span>
                {filters.predictionTypes.has('v_prediction_ztsnr') && <Check size={12} />}
              </button>
            </div>

            <div className="h-px bg-slate-100 dark:bg-slate-700/50" />

            {/* Source */}
            <div className="space-y-0.5">
              <label className="text-[10px] font-bold text-slate-400 px-2 uppercase opacity-70">Source</label>
              <button
                onClick={() => toggleFilter('sources', 'Civitai')}
                className={`w-full flex items-center justify-between px-2 py-1 rounded-md text-xs transition-colors ${
                  filters.sources.has('Civitai')
                    ? 'bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300'
                    : 'hover:bg-slate-50 dark:hover:bg-slate-700/50 text-slate-600 dark:text-slate-300'
                }`}
              >
                <span>Civitai</span>
                {filters.sources.has('Civitai') && <Check size={12} />}
              </button>
              <button
                onClick={() => toggleFilter('sources', 'HuggingFace')}
                className={`w-full flex items-center justify-between px-2 py-1 rounded-md text-xs transition-colors ${
                  filters.sources.has('HuggingFace')
                    ? 'bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300'
                    : 'hover:bg-slate-50 dark:hover:bg-slate-700/50 text-slate-600 dark:text-slate-300'
                }`}
              >
                <span>HuggingFace</span>
                {filters.sources.has('HuggingFace') && <Check size={12} />}
              </button>
              <button
                onClick={() => toggleFilter('sources', 'Local')}
                className={`w-full flex items-center justify-between px-2 py-1 rounded-md text-xs transition-colors ${
                  filters.sources.has('Local')
                    ? 'bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300'
                    : 'hover:bg-slate-50 dark:hover:bg-slate-700/50 text-slate-600 dark:text-slate-300'
                }`}
              >
                <span>Local</span>
                {filters.sources.has('Local') && <Check size={12} />}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
