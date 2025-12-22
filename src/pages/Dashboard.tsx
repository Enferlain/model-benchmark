import React, { useState, useCallback, useEffect } from "react";
import {
  Plus,
  Info,
  Loader2,
} from "lucide-react";
import { ScatterPlot } from "../components/ScatterPlot";
import { ModelTable } from "../components/ModelTable";
import { ScanSettingsPanel, ScanOptionsType, DEFAULT_SCAN_OPTIONS } from "../components/ScanSettingsPanel";
import { METRIC_OPTIONS } from "../constants";
import { ModelData, MetricKey } from "../types";
import { useOutletContext } from "react-router-dom";
import { analyzeModelUrl, deleteModel, generateImages, analyzeImages, cancelOperation, getStatus } from "../services/api";

// Context type from the MainLayout if we were using context for shared state,
// but currently props are passed down or managed here.
// For now, we'll manage model state here or lift it up if needed.
// Since Dashboard is the main viewer, it can own the data for now.

interface DashboardProps {
  models: ModelData[];
  setModels: React.Dispatch<React.SetStateAction<ModelData[]>>;
  isLoading: boolean;
  fetchModels: () => Promise<void>;
}

export default function Dashboard({ models, setModels, isLoading, fetchModels }: DashboardProps) {
  const [urlInput, setUrlInput] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [scanOptions, setScanOptions] = useState<ScanOptionsType>(DEFAULT_SCAN_OPTIONS);

  const [xMetricKey, setXMetricKey] = useState<MetricKey>("accuracy");
  const [yMetricKey, setYMetricKey] = useState<MetricKey>("diversity");

  const xMetric =
    METRIC_OPTIONS.find((m) => m.value === xMetricKey) || METRIC_OPTIONS[0];
  const yMetric =
    METRIC_OPTIONS.find((m) => m.value === yMetricKey) || METRIC_OPTIONS[1];

  // Status for generation progress (polled from backend)
  const [generationStatus, setGenerationStatus] = useState({
    is_running: false,
    current_model: null as string | null,
    progress: { current: 0, total: 0 }
  });

  // Poll status when scanning
  useEffect(() => {
    if (!isScanning) return;
    
    const interval = setInterval(async () => {
      try {
        const status = await getStatus();
        setGenerationStatus(status);
        if (!status.is_running) {
          setIsScanning(false);
        }
      } catch (e) {
        console.error('Status poll error:', e);
      }
    }, 1000);
    
    return () => clearInterval(interval);
  }, [isScanning]);

  const handleGenerate = useCallback(async () => {
    setIsScanning(true);
    setGenerationStatus({ is_running: true, current_model: null, progress: { current: 0, total: 0 } });
    try {
      await generateImages(scanOptions);
    } catch (error) {
      console.error('Generate error:', error);
    } finally {
      setIsScanning(false);
      setGenerationStatus(prev => ({ ...prev, is_running: false }));
    }
  }, [scanOptions]);

  const handleAnalyze = useCallback(async () => {
    setIsScanning(true);
    try {
      await analyzeImages(scanOptions);
      await fetchModels();
    } catch (error) {
      console.error('Analyze error:', error);
    } finally {
      setIsScanning(false);
    }
  }, [scanOptions, fetchModels]);

  const handleCancel = useCallback(async () => {
    try {
      await cancelOperation();
    } catch (error) {
      console.error('Cancel error:', error);
    }
  }, []);

  const parseUrl = (
    url: string
  ): { name: string; source: "Civitai" | "HuggingFace" | "Unknown" } | null => {
    try {
      const urlObj = new URL(url);

      if (url.includes("civitai.com")) {
        const parts = urlObj.pathname.split("/").filter(Boolean);
        const namePart =
          parts.length >= 3 ? parts[2] : parts[1] || "Civitai Model";
        return {
          name: namePart
            .replace(/-/g, " ")
            .replace(/\b\w/g, (l) => l.toUpperCase()),
          source: "Civitai",
        };
      }

      if (url.includes("huggingface.co")) {
        const parts = urlObj.pathname.split("/").filter(Boolean);
        const namePart =
          parts.length >= 2 ? `${parts[0]}/${parts[1]}` : "HF Model";
        return {
          name: namePart,
          source: "HuggingFace",
        };
      }

      // Allow generic URLs for testing if needed
      if (url) {
        return { name: "Unknown Model", source: "Unknown" };
      }

      return null;
    } catch (e) {
      return null;
    }
  };

  const handleAddModel = useCallback(async () => {
    const trimmedUrl = urlInput.trim();
    if (!trimmedUrl) return;

    const info = parseUrl(trimmedUrl);
    if (!info) {
      alert("Please enter a valid Civitai or HuggingFace model URL.");
      return;
    }

    setIsProcessing(true);

    try {
      const newModel = await analyzeModelUrl(trimmedUrl, info.name, info.source);
      setModels((prev) => [...prev, newModel]);
      setUrlInput("");
    } catch (error) {
      console.error("Error analyzing model:", error);
      alert("Error connecting to backend or analyzing model.");
    } finally {
      setIsProcessing(false);
    }
  }, [urlInput, setModels]);

  const handleDeleteModel = useCallback(async (id: string) => {
    // Optimistic update
    setModels((prev) => prev.filter((m) => m.id !== id));
    try {
      await deleteModel(id);
    } catch (error) {
      console.error("Error deleting model:", error);
      // Could revert state here
    }
  }, [setModels]);

  return (
      <div className="max-w-[1800px] mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Sidebar / Controls */}
          <div className="lg:col-span-3 space-y-6">
            {/* Input Glass Card */}
            <div className="p-6 rounded-3xl shadow-xl shadow-slate-200/50 dark:shadow-black/20 border border-white/60 dark:border-white/5 bg-white/60 dark:bg-slate-800/40 backdrop-blur-xl transition-all hover:shadow-2xl hover:bg-white/80 dark:hover:bg-slate-800/50">
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-4 flex items-center gap-2">
                <Plus size={14} /> Add Model
              </h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-[10px] font-bold text-slate-500 dark:text-slate-400 mb-2 ml-1 opacity-80">
                    MODEL URL
                  </label>
                  <input
                    type="text"
                    value={urlInput}
                    onChange={(e) => setUrlInput(e.target.value)}
                    placeholder="https://..."
                    className="w-full px-4 py-3 border border-slate-200/60 dark:border-white/5 bg-white/50 dark:bg-black/20 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/30 dark:focus:ring-blue-400/20 transition-all placeholder:text-slate-400/70 dark:placeholder:text-slate-600 text-slate-800 dark:text-slate-200 backdrop-blur-sm"
                    onKeyDown={(e) => e.key === "Enter" && handleAddModel()}
                  />
                </div>
                <button
                  onClick={handleAddModel}
                  disabled={!urlInput || isProcessing}
                  className="w-full bg-blue-600/90 hover:bg-blue-600 dark:bg-blue-500/80 dark:hover:bg-blue-500 text-white disabled:bg-slate-200 dark:disabled:bg-slate-800/50 disabled:text-slate-400 disabled:cursor-not-allowed font-medium py-3 px-4 rounded-xl transition-all duration-300 text-sm flex items-center justify-center gap-2 shadow-lg shadow-blue-500/20 dark:shadow-blue-900/20 backdrop-blur-sm"
                >
                  {isProcessing ? (
                    <>
                      <Loader2 size={16} className="animate-spin" /> Processing
                    </>
                  ) : (
                    <>Fetch & Analyze</>
                  )}
                </button>
              </div>
            </div>

            {/* Sample Settings Panel */}
            <ScanSettingsPanel
              options={scanOptions}
              onChange={setScanOptions}
              onGenerate={handleGenerate}
              onAnalyze={handleAnalyze}
              onCancel={handleCancel}
              status={generationStatus}
              onRefreshModels={fetchModels}
            />

            {/* Plot Settings Glass Card */}
            <div className="p-6 rounded-3xl shadow-xl shadow-slate-200/50 dark:shadow-black/20 border border-white/60 dark:border-white/5 bg-white/60 dark:bg-slate-800/40 backdrop-blur-xl transition-all">
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-4">
                View Settings
              </h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-[10px] font-bold text-slate-500 dark:text-slate-400 mb-2 ml-1 opacity-80">
                    X-AXIS METRIC
                  </label>
                  <div className="relative">
                    <select
                      value={xMetricKey}
                      onChange={(e) =>
                        setXMetricKey(e.target.value as MetricKey)
                      }
                      className="w-full px-4 py-3 bg-white/50 dark:bg-black/20 border border-slate-200/60 dark:border-white/5 rounded-xl text-sm appearance-none focus:outline-none focus:ring-2 focus:ring-blue-500/30 dark:focus:ring-blue-400/20 text-slate-800 dark:text-slate-200 cursor-pointer backdrop-blur-sm"
                    >
                      {METRIC_OPTIONS.map((opt) => (
                        <option
                          key={`x-${opt.value}`}
                          value={opt.value}
                          className="bg-white dark:bg-slate-800"
                        >
                          {opt.label}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                <div>
                  <label className="block text-[10px] font-bold text-slate-500 dark:text-slate-400 mb-2 ml-1 opacity-80">
                    Y-AXIS METRIC
                  </label>
                  <div className="relative">
                    <select
                      value={yMetricKey}
                      onChange={(e) =>
                        setYMetricKey(e.target.value as MetricKey)
                      }
                      className="w-full px-4 py-3 bg-white/50 dark:bg-black/20 border border-slate-200/60 dark:border-white/5 rounded-xl text-sm appearance-none focus:outline-none focus:ring-2 focus:ring-blue-500/30 dark:focus:ring-blue-400/20 text-slate-800 dark:text-slate-200 cursor-pointer backdrop-blur-sm"
                    >
                      {METRIC_OPTIONS.map((opt) => (
                        <option
                          key={`y-${opt.value}`}
                          value={opt.value}
                          className="bg-white dark:bg-slate-800"
                        >
                          {opt.label}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>
            </div>

            {/* Info Card */}
            <div className="p-5 rounded-3xl border border-blue-100/50 dark:border-blue-500/10 bg-blue-50/50 dark:bg-blue-500/5 backdrop-blur-md">
              <div className="flex gap-3">
                <Info
                  className="text-blue-500 dark:text-blue-400 shrink-0 mt-0.5"
                  size={18}
                />
                <div className="space-y-3">
                  <p className="text-sm text-blue-900 dark:text-blue-100 font-medium">
                    Metric Info
                  </p>
                  
                  <div className="space-y-2">
                    <div className="text-xs text-blue-800/80 dark:text-blue-200/80">
                      <span className="font-semibold">{xMetric.label}:</span> {xMetric.description}
                      {xMetric.direction && (
                        <span className="ml-1 opacity-75">
                          ({xMetric.direction === 'higher' ? 'Higher is better ⬆️' : 'Lower is better ⬇️'})
                        </span>
                      )}
                    </div>
                    
                    <div className="text-xs text-blue-800/80 dark:text-blue-200/80">
                      <span className="font-semibold">{yMetric.label}:</span> {yMetric.description}
                      {yMetric.direction && (
                        <span className="ml-1 opacity-75">
                          ({yMetric.direction === 'higher' ? 'Higher is better ⬆️' : 'Lower is better ⬇️'})
                        </span>
                      )}
                    </div>
                  </div>
                  
                  <div className="pt-2 border-t border-blue-200/30 dark:border-blue-500/20">
                     <p className="text-[10px] text-blue-600/60 dark:text-blue-400/50">
                        Running on :8000
                     </p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-9 space-y-8">
            {/* Chart Area */}
            <div className="relative">
              {isLoading ? (
                <div className="h-[650px] flex items-center justify-center text-slate-400 bg-white/50 dark:bg-slate-800/30 rounded-3xl">
                  <div className="text-center">
                    <Loader2 className="animate-spin mx-auto mb-2" />
                    <p>Loading Models...</p>
                  </div>
                </div>
              ) : (
                <ScatterPlot
                  data={models}
                  xMetric={xMetric}
                  yMetric={yMetric}
                  // isDarkMode is now handled by CSS classes mainly, but if ScatterPlot needs it,
                  // we can check document class or pass it.
                  // Original passed `isDarkMode`. We can get it from document for now if needed,
                  // or ignore if ScatterPlot uses CSS vars.
                  // Let's assume we need to pass it for Recharts themes if logic existed.
                  isDarkMode={document.documentElement.classList.contains('dark')}
                />
              )}
            </div>

            {/* Table Area */}
            <div className="pt-2">
              <ModelTable models={models} onDelete={handleDeleteModel} />
            </div>
          </div>
        </div>
      </div>
  );
}
