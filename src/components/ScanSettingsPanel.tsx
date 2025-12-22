import React, { useState, useEffect } from 'react';
import { Settings, ChevronDown, ChevronUp, Play, Square, BarChart2 } from 'lucide-react';

export interface ScanOptionsType {
  sampler: string;
  steps: number;
  guidance_scale: number;
  seed: number;
  images_per_prompt: number;
  num_prompts: number;
  width: number;
  height: number;
}

export interface GenerationStatus {
  is_running: boolean;
  current_model: string | null;
  progress: { current: number; total: number };
}

const SAMPLER_OPTIONS = [
  'euler_a', 'euler', 'ddim', 'pndm', 'lms', 'heun', 
  'dpm_2', 'dpm_2_a', 'dpmsolver', 'dpmsolver++', 'dpmsingle',
  'k_lms', 'k_euler', 'k_euler_a', 'k_dpm_2', 'k_dpm_2_a'
];

interface ScanSettingsPanelProps {
  options: ScanOptionsType;
  onChange: (options: ScanOptionsType) => void;
  onGenerate: () => Promise<void>;
  onAnalyze: () => Promise<void>;
  onCancel: () => Promise<void>;
  status: GenerationStatus;
  onRefreshModels: () => void;
}

export const ScanSettingsPanel: React.FC<ScanSettingsPanelProps> = ({ 
  options, 
  onChange, 
  onGenerate,
  onAnalyze,
  onCancel,
  status,
  onRefreshModels
}) => {
  const [isExpanded, setIsExpanded] = useState(true);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const updateOption = <K extends keyof ScanOptionsType>(key: K, value: ScanOptionsType[K]) => {
    onChange({ ...options, [key]: value });
  };

  const handleGenerate = async () => {
    setIsGenerating(true);
    try {
      await onGenerate();
    } finally {
      setIsGenerating(false);
      onRefreshModels();
    }
  };

  const handleAnalyze = async () => {
    setIsAnalyzing(true);
    try {
      await onAnalyze();
    } finally {
      setIsAnalyzing(false);
      onRefreshModels();
    }
  };

  const handleCancel = async () => {
    await onCancel();
  };

  return (
    <div className="p-6 rounded-3xl shadow-xl shadow-slate-200/50 dark:shadow-black/20 border border-white/60 dark:border-white/5 bg-white/60 dark:bg-slate-800/40 backdrop-blur-xl transition-all">
      <button 
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between mb-4"
      >
        <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 flex items-center gap-2">
          <Settings size={14} /> Sample Settings
        </h2>
        {isExpanded ? <ChevronUp size={16} className="text-slate-400" /> : <ChevronDown size={16} className="text-slate-400" />}
      </button>

      {isExpanded && (
        <div className="space-y-4">
          {/* Sampler */}
          <div>
            <label className="block text-[10px] font-bold text-slate-500 dark:text-slate-400 mb-1 ml-1">SAMPLER</label>
            <select
              value={options.sampler}
              onChange={(e) => updateOption('sampler', e.target.value)}
              disabled={status.is_running}
              className="w-full px-3 py-2 bg-white/50 dark:bg-black/20 border border-slate-200/60 dark:border-white/5 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/30 text-slate-800 dark:text-slate-200 disabled:opacity-50"
            >
              {SAMPLER_OPTIONS.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>

          {/* Two column grid for numbers */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-[10px] font-bold text-slate-500 dark:text-slate-400 mb-1 ml-1">STEPS</label>
              <input
                type="number"
                value={options.steps}
                onChange={(e) => updateOption('steps', parseInt(e.target.value) || 20)}
                min={1}
                max={150}
                disabled={status.is_running}
                className="w-full px-3 py-2 bg-white/50 dark:bg-black/20 border border-slate-200/60 dark:border-white/5 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/30 text-slate-800 dark:text-slate-200 disabled:opacity-50"
              />
            </div>
            <div>
              <label className="block text-[10px] font-bold text-slate-500 dark:text-slate-400 mb-1 ml-1">CFG SCALE</label>
              <input
                type="number"
                value={options.guidance_scale}
                onChange={(e) => updateOption('guidance_scale', parseFloat(e.target.value) || 5.0)}
                min={1}
                max={30}
                step={0.5}
                disabled={status.is_running}
                className="w-full px-3 py-2 bg-white/50 dark:bg-black/20 border border-slate-200/60 dark:border-white/5 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/30 text-slate-800 dark:text-slate-200 disabled:opacity-50"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-[10px] font-bold text-slate-500 dark:text-slate-400 mb-1 ml-1">SEED</label>
              <input
                type="number"
                value={options.seed}
                onChange={(e) => updateOption('seed', parseInt(e.target.value) || 0)}
                disabled={status.is_running}
                className="w-full px-3 py-2 bg-white/50 dark:bg-black/20 border border-slate-200/60 dark:border-white/5 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/30 text-slate-800 dark:text-slate-200 disabled:opacity-50"
              />
            </div>
            <div>
              <label className="block text-[10px] font-bold text-slate-500 dark:text-slate-400 mb-1 ml-1">PROMPTS</label>
              <input
                type="number"
                value={options.num_prompts}
                onChange={(e) => updateOption('num_prompts', parseInt(e.target.value) || 10)}
                min={1}
                max={100}
                disabled={status.is_running}
                className="w-full px-3 py-2 bg-white/50 dark:bg-black/20 border border-slate-200/60 dark:border-white/5 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/30 text-slate-800 dark:text-slate-200 disabled:opacity-50"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-[10px] font-bold text-slate-500 dark:text-slate-400 mb-1 ml-1">WIDTH</label>
              <input
                type="number"
                value={options.width}
                onChange={(e) => updateOption('width', parseInt(e.target.value) || 1024)}
                min={512}
                max={2048}
                step={64}
                disabled={status.is_running}
                className="w-full px-3 py-2 bg-white/50 dark:bg-black/20 border border-slate-200/60 dark:border-white/5 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/30 text-slate-800 dark:text-slate-200 disabled:opacity-50"
              />
            </div>
            <div>
              <label className="block text-[10px] font-bold text-slate-500 dark:text-slate-400 mb-1 ml-1">HEIGHT</label>
              <input
                type="number"
                value={options.height}
                onChange={(e) => updateOption('height', parseInt(e.target.value) || 1536)}
                min={512}
                max={2048}
                step={64}
                disabled={status.is_running}
                className="w-full px-3 py-2 bg-white/50 dark:bg-black/20 border border-slate-200/60 dark:border-white/5 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/30 text-slate-800 dark:text-slate-200 disabled:opacity-50"
              />
            </div>
          </div>

          <div>
            <label className="block text-[10px] font-bold text-slate-500 dark:text-slate-400 mb-1 ml-1">
              IMAGES PER PROMPT <span className="text-blue-500">(for LPIPS diversity)</span>
            </label>
            <input
              type="number"
              value={options.images_per_prompt}
              onChange={(e) => updateOption('images_per_prompt', parseInt(e.target.value) || 1)}
              min={1}
              max={10}
              disabled={status.is_running}
              className="w-full px-3 py-2 bg-white/50 dark:bg-black/20 border border-slate-200/60 dark:border-white/5 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/30 text-slate-800 dark:text-slate-200 disabled:opacity-50"
            />
            <p className="text-[10px] text-slate-400 mt-1 ml-1">Set &gt; 1 to measure intra-prompt diversity</p>
          </div>

          {/* Info summary */}
          <div className="bg-slate-100/50 dark:bg-slate-900/30 rounded-xl p-3 text-xs text-slate-600 dark:text-slate-400">
            <p>
              <strong>Total images per model:</strong> {options.num_prompts} × {options.images_per_prompt} = {options.num_prompts * options.images_per_prompt}
            </p>
            <p className="mt-1">
              <strong>Resolution:</strong> {options.width} × {options.height}
            </p>
          </div>

          {/* Progress indicator */}
          {status.is_running && (
            <div className="bg-blue-50/50 dark:bg-blue-900/20 rounded-xl p-3 text-xs text-blue-700 dark:text-blue-300">
              <p className="font-medium">Generating...</p>
              {status.current_model && <p>Model: {status.current_model}</p>}
              {status.progress.total > 0 && (
                <>
                  <div className="mt-2 h-1.5 bg-blue-200 dark:bg-blue-800 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-blue-500 transition-all duration-300"
                      style={{ width: `${(status.progress.current / status.progress.total) * 100}%` }}
                    />
                  </div>
                  <p className="mt-1">{status.progress.current} / {status.progress.total} images</p>
                </>
              )}
            </div>
          )}

          {/* Buttons */}
          <div className="grid grid-cols-2 gap-3">
            {status.is_running ? (
              <button
                onClick={handleCancel}
                className="col-span-2 py-3 px-4 rounded-xl font-medium text-sm transition-all flex items-center justify-center gap-2 bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 text-white shadow-lg shadow-red-500/20"
              >
                <Square size={16} /> Cancel
              </button>
            ) : (
              <>
                <button
                  onClick={handleGenerate}
                  disabled={isGenerating || isAnalyzing}
                  className="py-3 px-4 rounded-xl font-medium text-sm transition-all flex items-center justify-center gap-2 bg-gradient-to-r from-green-500 to-green-600 hover:from-green-600 hover:to-green-700 text-white shadow-lg shadow-green-500/20 disabled:opacity-50"
                >
                  <Play size={16} /> Generate
                </button>
                <button
                  onClick={handleAnalyze}
                  disabled={isGenerating || isAnalyzing}
                  className="py-3 px-4 rounded-xl font-medium text-sm transition-all flex items-center justify-center gap-2 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white shadow-lg shadow-blue-500/20 disabled:opacity-50"
                >
                  <BarChart2 size={16} /> Analyze
                </button>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export const DEFAULT_SCAN_OPTIONS: ScanOptionsType = {
  sampler: 'euler_a',
  steps: 28,
  guidance_scale: 5.0,
  seed: 218,
  images_per_prompt: 1,
  num_prompts: 10,
  width: 1024,
  height: 1536,
};

