import React, { useState } from 'react';
import { BarChart3 } from 'lucide-react';
import { ModelTable } from '../components/ModelTable';
import { ModelData } from '../types';
import { deleteModel } from '../services/api';

interface AnalyticsProps {
  models: ModelData[];
  setModels: React.Dispatch<React.SetStateAction<ModelData[]>>;
  fetchModels: () => Promise<void>;
}

export default function Analytics({ models, setModels, fetchModels }: AnalyticsProps) {
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const handleDeleteModel = async (id: string, deleteFile: boolean) => {
    // Save previous state for revert
    const previousModels = [...models];

    // Optimistic update
    setModels((prev) => prev.filter((m) => m.id !== id));

    try {
      await deleteModel(id, deleteFile);
      if (selectedId === id) setSelectedId(null);
    } catch (error) {
      console.error("Error deleting model:", error);
      // Revert state on error
      setModels(previousModels);
      // Ideally show a toast here
    }
  };

  return (
    <div className="max-w-[1800px] mx-auto px-6 py-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-slate-900 to-slate-600 dark:from-white dark:to-slate-400 flex items-center gap-3">
          <BarChart3 className="text-blue-500" />
          Benchmark Analytics
        </h1>
        <p className="text-slate-500 dark:text-slate-400 mt-2">
          Detailed performance metrics and management for {models.length} models.
        </p>
      </div>

      <div className="space-y-6">
         {/* We can re-add the chart here later if requested, for now the table is the main focus as requested */}
         <ModelTable 
            models={models} 
            onDelete={handleDeleteModel}
            selectedId={selectedId}
            onSelect={setSelectedId}
         />
      </div>
    </div>
  );
}
