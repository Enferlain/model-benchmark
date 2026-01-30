import { BarChart3, Search, Filter, Globe, Table, LayoutGrid, X } from "lucide-react";
import { useEffect, useState, useMemo } from "react";
import { ModelTable } from "../components/ModelTable";
import { useData } from "../context/DataContext";
import { deleteModel } from "../services/api";
import type { ModelData, BenchmarkRun } from "../types";

export default function Analytics() {
	const { models, runs, refreshModels: fetchModels } = useData();
	const [selectedId, setSelectedId] = useState<string | null>(() => {
		return localStorage.getItem("analytics_selectedId");
	});

	// View Settings
	const [viewMode, setViewMode] = useState<"global" | "run">("global");
	const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
	
	// Filters
	const [searchQuery, setSearchQuery] = useState("");
	const [architectureFilter, setArchitectureFilter] = useState<string>("all");
	const [sourceFilter, setSourceFilter] = useState<string>("all");

	useEffect(() => {
		if (selectedId) {
			localStorage.setItem("analytics_selectedId", selectedId);
		} else {
			localStorage.removeItem("analytics_selectedId");
		}
	}, [selectedId]);

	const handleDeleteModel = async (id: string, deleteFile: boolean) => {
		try {
			await deleteModel(id, deleteFile);
			if (selectedId === id) setSelectedId(null);
			fetchModels();
		} catch (error) {
			console.error("Error deleting model:", error);
		}
	};

	// Get available architectures and sources for filters
	const architectures = useMemo(() => {
		const archs = new Set(models.map(m => m.model_type).filter(Boolean));
		return Array.from(archs).sort();
	}, [models]);

	const sources = useMemo(() => {
		const srcs = new Set(models.map(m => m.source).filter(Boolean));
		return Array.from(srcs).sort();
	}, [models]);

	// Filtered and Transformed Models
	const displayModels = useMemo(() => {
		let result = [...models];

		// 1. Filter by architecture
		if (architectureFilter !== "all") {
			result = result.filter(m => m.model_type === architectureFilter);
		}

		// 2. Filter by source
		if (sourceFilter !== "all") {
			result = result.filter(m => m.source === sourceFilter);
		}

		// 3. Filter by search
		if (searchQuery) {
			const q = searchQuery.toLowerCase();
			result = result.filter(m => 
				m.name.toLowerCase().includes(q) || 
				m.id.toLowerCase().includes(q)
			);
		}

		// 4. Transform for Run Mode vs Global Mode
		if (viewMode === "run" && selectedRunId) {
			const run = runs.find(r => r.id === selectedRunId);
			if (run) {
				// Only keep models that were in this run
				result = result.filter(m => run.results.some(res => res.model_hash === m.id));
				
				// Overlay the run-specific metrics onto the model data
				result = result.map(m => {
					const runResult = run.results.find(res => res.model_hash === m.id);
					return {
						...m,
						// Override global avg/latest with this specific run's metrics
						// We'll pass a special flag to ModelTable to handle this
						_runMetrics: runResult?.metrics,
						_runImageCount: runResult?.image_count
					} as ModelData & { _runMetrics?: Record<string, number>; _runImageCount?: number };
				});
			} else {
				return []; // Run not found
			}
		}

		return result;
	}, [models, runs, viewMode, selectedRunId, searchQuery, architectureFilter, sourceFilter]);

	return (
		<div className="max-w-[1800px] mx-auto px-6 py-8">
			<div className="mb-8">
				<h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-slate-900 to-slate-600 dark:from-white dark:to-slate-400 flex items-center gap-3">
					<BarChart3 className="text-blue-500" />
					Benchmark Analytics
				</h1>
				<p className="text-slate-500 dark:text-slate-400 mt-2">
					{viewMode === "global" 
						? `Global performance aggregates across ${models.length} models.`
						: `Comparative results for Run #${selectedRunId}.`
					}
				</p>
			</div>

			<div className="space-y-6">
				<div className="bg-white dark:bg-slate-800 rounded-2xl shadow-sm border border-slate-200 dark:border-slate-700 overflow-hidden">
					{/* Custom Model Data Header / Toolbar */}
					<div className="px-6 py-4 border-b border-slate-200 dark:border-slate-700 bg-slate-50/50 dark:bg-slate-900/20">
						<div className="flex flex-wrap items-center justify-between gap-4">
							<div className="flex items-center gap-4">
								<h3 className="text-sm font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 flex items-center gap-2">
									<LayoutGrid size={16} />
									Model Data
								</h3>
								
								{/* View Mode Switcher */}
								<div className="flex items-center bg-slate-200/50 dark:bg-slate-800 p-0.5 rounded-lg border border-slate-200 dark:border-slate-700">
									<button
										type="button"
										onClick={() => setViewMode("global")}
										className={`flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-medium transition-all ${
											viewMode === "global" 
												? "bg-white dark:bg-slate-700 shadow-sm text-blue-600 dark:text-blue-400" 
												: "text-slate-500 hover:text-slate-700 dark:hover:text-slate-300"
										}`}
									>
										<Globe size={14} />
										Global Avg
									</button>
									<button
										type="button"
										onClick={() => setViewMode("run")}
										className={`flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-medium transition-all ${
											viewMode === "run" 
												? "bg-white dark:bg-slate-700 shadow-sm text-blue-600 dark:text-blue-400" 
												: "text-slate-500 hover:text-slate-700 dark:hover:text-slate-300"
										}`}
									>
										<Table size={14} />
										Shared Run
									</button>
								</div>

								{/* Run Selector (Conditional) */}
								{viewMode === "run" && (
									<select 
										value={selectedRunId || ""}
										onChange={(e) => setSelectedRunId(Number(e.target.value))}
										className="text-xs bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-1.5 focus:ring-2 focus:ring-blue-500 outline-none min-w-[180px]"
									>
										<option value="">Select a Run...</option>
										{runs.sort((a,b) => b.id - a.id).map(run => (
											<option key={run.id} value={run.id}>
												Run #{run.id} - {new Date(run.timestamp).toLocaleDateString()}
											</option>
										))}
									</select>
								)}
							</div>

							<div className="flex items-center gap-3">
								{/* Search */}
								<div className="relative">
									<Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
									<input 
										type="text"
										placeholder="Search models..."
										value={searchQuery}
										onChange={(e) => setSearchQuery(e.target.value)}
										className="pl-9 pr-4 py-1.5 text-xs bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none w-[200px]"
									/>
									{searchQuery && (
										<button 
											type="button"
											onClick={() => setSearchQuery("")}
											className="absolute right-2 top-1/2 -translate-y-1/2 p-0.5 hover:bg-slate-100 dark:hover:bg-slate-700 rounded"
										>
											<X size={12} className="text-slate-400" />
										</button>
									)}
								</div>

								{/* Filters */}
								<div className="flex items-center gap-2 pl-4 border-l border-slate-200 dark:border-slate-700">
									<Filter size={14} className="text-slate-400" />
									<select 
										value={architectureFilter}
										onChange={(e) => setArchitectureFilter(e.target.value)}
										className="text-[11px] bg-transparent border-none focus:ring-0 cursor-pointer text-slate-600 dark:text-slate-300 font-medium"
									>
										<option value="all">Any Architecture</option>
										{architectures.map(arch => (
											<option key={arch} value={arch}>{arch.toUpperCase()}</option>
										))}
									</select>
									<select 
										value={sourceFilter}
										onChange={(e) => setSourceFilter(e.target.value)}
										className="text-[11px] bg-transparent border-none focus:ring-0 cursor-pointer text-slate-600 dark:text-slate-300 font-medium"
									>
										<option value="all">Any Source</option>
										{sources.map(src => (
											<option key={src} value={src}>{src}</option>
										))}
									</select>
								</div>
							</div>
						</div>
					</div>

					<ModelTable
						models={displayModels}
						onDelete={handleDeleteModel}
						selectedId={selectedId}
						onSelect={setSelectedId}
						isRunSpecific={viewMode === "run"}
						activeRunId={selectedRunId}
					/>
				</div>
			</div>
		</div>
	);
}
