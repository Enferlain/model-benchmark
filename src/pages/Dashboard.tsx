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
import { downloadModel, getDownloadStatus, deleteModel, generateImages, analyzeImages, cancelOperation, getStatus } from "../services/api";
import { useTheme } from "../context/ThemeContext";
import { Download, HardDrive } from "lucide-react";

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
  const { isDarkMode } = useTheme();
  const [urlInput, setUrlInput] = useState("");
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [scanOptions, setScanOptions] = useState<ScanOptionsType>(DEFAULT_SCAN_OPTIONS);

  const [downloadProgress, setDownloadProgress] = useState({
     current: 0,
     total: 0,
     status: 'idle',
     filename: ''
  });

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

  // Poll download status
  useEffect(() => {
    if (!isDownloading) return;

    const interval = setInterval(async () => {
        try {
            const status = await getDownloadStatus();
            setDownloadProgress({
                current: status.progress,
                total: status.total,
                status: status.status,
                filename: status.current_file
            });

            if (status.status === 'completed' || status.status === 'error') {
                setIsDownloading(false);
                if (status.status === 'completed') {
                    await fetchModels();
                    setUrlInput("");
                } else {
                    setDownloadError(status.error || "Download failed");
                }
            }
        } catch (e) {
            console.error('Download status poll error:', e);
            setDownloadError("Error polling download status");
            setIsDownloading(false);
        }
    }, 1000);

    return () => clearInterval(interval);
  }, [isDownloading, fetchModels]);

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
        const lastPart = parts[parts.length - 1];

        // Check if the URL points directly to a model file
        if (lastPart && /\.(safetensors|ckpt|pt|bin|pth)$/i.test(lastPart)) {
             return {
                 name: lastPart,
                 source: "HuggingFace"
             };
        }

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

  const handleDownloadModel = useCallback(async () => {
    setDownloadError(null);
    const trimmedUrl = urlInput.trim();
    if (!trimmedUrl) return;

    const info = parseUrl(trimmedUrl);
    if (!info) {
      setDownloadError("Please enter a valid Civitai or HuggingFace model URL.");
      return;
    }

    setIsDownloading(true);
    setDownloadProgress({ current: 0, total: 0, status: 'downloading', filename: info.name });

    try {
      await downloadModel(trimmedUrl, info.name, info.source);
    } catch (error: any) {
      console.error("Error starting download:", error);
      setDownloadError(error.message || "Error connecting to backend or starting download.");
      setIsDownloading(false);
    }
  }, [urlInput]);

  const handleDeleteModel = useCallback(async (id: string, deleteFile: boolean) => {
    // Save previous state for revert
    const previousModels = [...models];

    // Optimistic update
    setModels((prev) => prev.filter((m) => m.id !== id));

    try {
      await deleteModel(id, deleteFile);
    } catch (error) {
      console.error("Error deleting model:", error);
      // Revert state on error
      setModels(previousModels);
      alert("Failed to delete model.");
    }
  }, [models, setModels]);

  // Shared selection state for Chart and Table
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);

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
                    onChange={(e) => {
                      setUrlInput(e.target.value);
                      if (downloadError) setDownloadError(null);
                    }}
                    placeholder="https://..."
                    className={`w-full px-4 py-3 border ${downloadError ? 'border-red-500/50 bg-red-500/5' : 'border-slate-200/60 dark:border-white/5 bg-white/50 dark:bg-black/20'} rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/30 dark:focus:ring-blue-400/20 transition-all placeholder:text-slate-400/70 dark:placeholder:text-slate-600 text-slate-800 dark:text-slate-200 backdrop-blur-sm`}
                    onKeyDown={(e) => e.key === "Enter" && handleDownloadModel()}
                  />
                  {downloadError && (
                    <p className="text-red-500 text-[10px] mt-1 ml-1 animate-pulse">
                      {downloadError}
                    </p>
                  )}
                </div>
                {isDownloading && downloadProgress.total > 0 && (
                   <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2.5 mb-1 overflow-hidden">
                      <div
                         className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
                         style={{ width: `${Math.min(100, (downloadProgress.current / downloadProgress.total) * 100)}%` }}
                      ></div>
                   </div>
                )}
                {isDownloading && (
                    <p className="text-[10px] text-slate-400 text-center mb-2">
                        {downloadProgress.total > 0
                            ? `${(downloadProgress.current / 1024 / 1024).toFixed(1)} / ${(downloadProgress.total / 1024 / 1024).toFixed(1)} MB`
                            : "Starting download..."
                        }
                    </p>
                )}
                <button
                  onClick={handleDownloadModel}
                  disabled={!urlInput || isDownloading}
                  className="w-full bg-blue-600/90 hover:bg-blue-600 dark:bg-blue-500/80 dark:hover:bg-blue-500 text-white disabled:bg-slate-200 dark:disabled:bg-slate-800/50 disabled:text-slate-400 disabled:cursor-not-allowed font-medium py-3 px-4 rounded-xl transition-all duration-300 text-sm flex items-center justify-center gap-2 shadow-lg shadow-blue-500/20 dark:shadow-blue-900/20 backdrop-blur-sm"
                >
                  {isDownloading ? (
                    <>
                      <Loader2 size={16} className="animate-spin" /> Downloading...
                    </>
                  ) : (
                    <>
                      <Download size={16} /> Download Model
                    </>
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
                  isDarkMode={isDarkMode}
                  selectedId={selectedModelId}
                  onSelect={setSelectedModelId}
                />
              )}
            </div>

            {/* Table Area */}
            <div className="pt-2">
              <ModelTable 
                 models={models} 
                 onDelete={handleDeleteModel}
                 selectedId={selectedModelId}
                 onSelect={setSelectedModelId}
              />
            </div>
          </div>
        </div>
      </div>
  );
}
