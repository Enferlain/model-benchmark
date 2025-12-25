import React, { useState, useEffect } from 'react';
import { fetchModels, fetchModelOutputs } from '../services/api';

interface Model {
  id: string;
  name: string;
  source: string;
  url: string;
  path: string;
}

interface ModelOutput {
  filename: string;
  url: string;
  prompt: string;
  seed: number;
  prompt_idx: number;
}

export default function Gallery() {
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [outputs, setOutputs] = useState<ModelOutput[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadModels();
  }, []);

  useEffect(() => {
    if (selectedModel) {
      loadOutputs(selectedModel);
    } else {
      setOutputs([]);
    }
  }, [selectedModel]);

  const loadModels = async () => {
    try {
      const data = await fetchModels();
      setModels(data);
      if (data.length > 0) {
        setSelectedModel(data[0].id);
      }
    } catch (err) {
      console.error("Failed to load models", err);
      setError("Failed to load models");
    }
  };

  const loadOutputs = async (modelId: string) => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchModelOutputs(modelId);
      setOutputs(data);
    } catch (err) {
      console.error("Failed to load outputs", err);
      setError("Failed to load outputs");
    } finally {
      setLoading(false);
    }
  };

  // Group by prompt for better viewing
  const groupedOutputs = outputs.reduce((acc, output) => {
    if (!acc[output.prompt]) {
      acc[output.prompt] = [];
    }
    acc[output.prompt].push(output);
    return acc;
  }, {} as Record<string, ModelOutput[]>);

  return (
    <div className="max-w-[1800px] mx-auto px-6 py-8">
      <div className="flex flex-col md:flex-row justify-between items-center mb-8 gap-4">
        <h2 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
          Image Gallery
        </h2>

        <div className="flex items-center gap-2">
           <label className="text-slate-600 dark:text-slate-300 font-medium">Select Model:</label>
           <select
             className="px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500"
             value={selectedModel}
             onChange={(e) => setSelectedModel(e.target.value)}
           >
             {models.length === 0 && <option value="">No models found</option>}
             {models.map(m => (
               <option key={m.id} value={m.id}>{m.name}</option>
             ))}
           </select>
        </div>
      </div>

      {error && (
        <div className="p-4 mb-6 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 rounded-md">
          {error}
        </div>
      )}

      {loading && (
        <div className="flex justify-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
        </div>
      )}

      {!loading && outputs.length === 0 && selectedModel && (
         <div className="text-center py-12 text-slate-500 dark:text-slate-400">
            No images found for this model. Try generating some first.
         </div>
      )}

      {!loading && Object.keys(groupedOutputs).map((prompt, idx) => (
        <div key={idx} className="mb-8">
          <div className="bg-slate-100 dark:bg-slate-800/50 p-4 rounded-lg mb-4">
            <h3 className="text-lg font-semibold text-slate-800 dark:text-slate-200 mb-1">
              Prompt {outputs.find(o => o.prompt === prompt)?.prompt_idx}:
            </h3>
            <p className="text-slate-600 dark:text-slate-400 font-mono text-sm">{prompt}</p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6">
            {groupedOutputs[prompt].map((output) => (
              <div key={output.filename} className="bg-white dark:bg-slate-800 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700 overflow-hidden hover:shadow-md transition-shadow">
                <div className="aspect-[2/3] relative group">
                  <img
                    src={`${import.meta.env.VITE_API_BASE?.replace('/api', '') || 'http://localhost:8000'}${output.url}`}
                    alt={output.prompt}
                    className="w-full h-full object-cover"
                    loading="lazy"
                  />
                  <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                    <a
                      href={`${import.meta.env.VITE_API_BASE?.replace('/api', '') || 'http://localhost:8000'}${output.url}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="px-4 py-2 bg-white text-slate-900 rounded-full text-sm font-medium hover:bg-indigo-50 transition-colors"
                    >
                      View Full
                    </a>
                  </div>
                </div>
                <div className="p-3 border-t border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50">
                   <div className="flex justify-between items-center text-xs text-slate-500 dark:text-slate-400">
                     <span>Seed: <span className="font-mono text-slate-700 dark:text-slate-300">{output.seed}</span></span>
                     <span className="truncate max-w-[100px]" title={output.filename}>{output.filename}</span>
                   </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
