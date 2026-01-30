import {
	Calendar,
	ChevronDown,
	ChevronUp,
	ExternalLink,
	FileText,
	Info,
	MoreVertical,
	Trash2,
	X,
} from "lucide-react";
import { useNavigate } from "react-router-dom";
import React, { useEffect, useMemo, useState } from "react";
import { METRIC_OPTIONS } from "../constants";
import type { MetricKey, MetricOption, ModelData } from "../types";
import { useData } from "../context/DataContext";
import { stringToColor } from "../utils/colorUtils";
import { DeleteConfirmModal } from "./DeleteConfirmModal";

interface ModelTableProps {
	models: ModelData[];
	onDelete: (id: string, deleteFile: boolean) => void;
	selectedId: string | null;
	onSelect: (id: string | null) => void;
	isRunSpecific?: boolean;
	activeRunId?: number | null;
}

type SortDirection = "asc" | "desc" | null;

// Metric Info Modal Component
const MetricInfoModal: React.FC<{
	metric: MetricOption | null;
	onClose: () => void;
}> = ({ metric, onClose }) => {
	if (!metric) return null;

	return (
		<div className="fixed inset-0 z-50 flex items-center justify-center p-4">
			<button
				type="button"
				className="absolute inset-0 bg-black/50 backdrop-blur-sm w-full h-full border-none p-0 m-0"
				onClick={onClose}
				aria-label="Close modal"
			/>
			<div className="relative bg-white dark:bg-slate-800 rounded-2xl shadow-2xl max-w-lg w-full max-h-[80vh] overflow-hidden border border-slate-200 dark:border-slate-700">
				<div className="px-6 py-4 border-b border-slate-200 dark:border-slate-700 flex items-center justify-between">
					<h3 className="text-lg font-semibold text-slate-800 dark:text-slate-100">
						{metric.label}
					</h3>
					<button
						type="button"
						onClick={onClose}
						className="p-1 hover:bg-slate-100 dark:hover:bg-slate-700 rounded-full transition-colors"
						aria-label="Close"
					>
						<X size={20} className="text-slate-500" />
					</button>
				</div>
				<div className="px-6 py-4 overflow-y-auto max-h-[60vh]">
					<div className="flex items-center gap-2 mb-4">
						<span
							className={`px-2 py-1 rounded text-xs font-medium ${
								metric.direction === "higher"
									? "bg-green-100 text-green-700 dark:bg-green-500/20 dark:text-green-300"
									: "bg-amber-100 text-amber-700 dark:bg-amber-500/20 dark:text-amber-300"
							}`}
						>
							{metric.direction === "higher"
								? "↑ Higher is better"
								: "↓ Lower is better"}
						</span>
					</div>
					<p className="text-slate-600 dark:text-slate-300 mb-4">
						{metric.description}
					</p>
					{metric.extendedDescription && (
						<div className="prose prose-sm dark:prose-invert max-w-none text-slate-600 dark:text-slate-300 whitespace-pre-wrap">
							{metric.extendedDescription}
						</div>
					)}
				</div>
			</div>
		</div>
	);
};

// Model History Dropdown Component
const ModelHistory: React.FC<{ model: ModelData }> = ({ model }) => {
	const { runs } = useData();
	const navigate = useNavigate();

	const allModelRuns = useMemo(() => {
		return runs
			.filter((run) => run.results.some((res) => res.model_hash === model.id))
			.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
	}, [runs, model.id]);

	const limitedRuns = useMemo(() => allModelRuns.slice(0, 5), [allModelRuns]);

	if (allModelRuns.length === 0) {
		return (
			<div className="px-12 py-8 text-center text-slate-400 italic text-sm border-t border-slate-100 dark:border-white/5">
				No historical benchmark runs found for this model.
			</div>
		);
	}

	return (
		<div className="px-12 py-6 space-y-4 w-full bg-slate-50/30 dark:bg-slate-900/10 border-t border-slate-100 dark:border-white/5">
			<h4 className="text-[10px] font-bold uppercase tracking-widest text-slate-400 dark:text-slate-500 mb-4 px-2 flex items-center gap-2">
				<Calendar size={12} />
				Benchmark History ({allModelRuns.length})
			</h4>
			<div className="space-y-3">
				{limitedRuns.map((run) => {
					const result = run.results.find((res) => res.model_hash === model.id);
					return (
						<div
							key={run.id}
							className="w-full text-left bg-white/50 dark:bg-slate-800/20 rounded-xl border border-slate-200/40 dark:border-white/5 p-4 flex items-center justify-between group"
						>
							<div className="flex items-center gap-6">
								<div 
									onClick={() => navigate(`/database?tab=runs&runId=${run.id}`)}
									className="cursor-pointer hover:opacity-70"
								>
									<div className="flex items-center gap-2 mb-1">
										<span className="font-mono text-[10px] font-bold text-slate-400">#{run.id}</span>
										<span className="text-xs font-medium text-slate-700 dark:text-slate-200">
											{new Date(run.timestamp).toLocaleString()}
										</span>
									</div>
									<div className="text-[10px] text-slate-400 flex items-center gap-1">
										<FileText size={10} />
										{run.prompt_set_id || "Ad-hoc Run"}
									</div>
								</div>

								<div className="flex items-center gap-6 border-l border-slate-200/40 dark:border-white/5 pl-6">
									{Object.entries(result?.metrics || {}).map(([key, val]) => (
										<div key={key} className="flex flex-col">
											<span className="text-[9px] uppercase font-bold text-slate-400 tracking-tight">{key}</span>
											<span className="text-xs font-mono text-slate-600 dark:text-slate-300">{(val as number).toFixed(3)}</span>
										</div>
									))}
								</div>
							</div>

							<div className="flex items-center gap-4">
								<div className="text-right text-[10px] text-slate-400 opacity-60">
									{run.parameters.width}x{run.parameters.height}, {run.parameters.steps} steps
								</div>
								<button 
									type="button"
									onClick={() => navigate(`/database?tab=runs&runId=${run.id}`)}
									className="p-2 text-slate-400 hover:text-blue-500 transition-colors"
								>
									<ExternalLink size={14} />
								</button>
							</div>
						</div>
					);
				})}
			</div>
			{allModelRuns.length > 5 && (
				<button
					type="button"
					onClick={() => navigate("/database?tab=runs")}
					className="text-[10px] font-semibold text-blue-500 hover:text-blue-600 transition-colors px-2"
				>
					View all {allModelRuns.length} runs in Database →
				</button>
			)}
		</div>
	);
};

export const ModelTable: React.FC<ModelTableProps> = ({
	models,
	onDelete,
	selectedId,
	onSelect,
	isRunSpecific = false,
	activeRunId = null,
}) => {
	const [sortKey, setSortKey] = useState<MetricKey | null>(null);
	const [sortDirection, setSortDirection] = useState<SortDirection>(null);
	const [selectedMetric, setSelectedMetric] = useState<MetricOption | null>(null);
	const [expandedModelIds, setExpandedModelIds] = useState<Set<string>>(new Set());
	const [deleteModal, setDeleteModal] = useState<{isOpen: boolean; modelId: string; modelName: string;}>({
		isOpen: false,
		modelId: "",
		modelName: "",
	});

	const [menuState, setMenuState] = useState<{
		isOpen: boolean;
		x: number;
		y: number;
		modelId: string;
		modelName: string;
	}>({ isOpen: false, x: 0, y: 0, modelId: "", modelName: "" });

	const getMetricValue = useMemo(() => {
		return (model: any, key: string): number => {
			if (model._runMetrics && key in model._runMetrics) return model._runMetrics[key];
			if (model.metrics_avg && key in model.metrics_avg) return model.metrics_avg[key];
			if (model.metrics && key in model.metrics) return model.metrics[key as MetricKey];
			return (model as unknown as Record<string, number>)[key] ?? 0;
		};
	}, []);

	const sortedModels = useMemo(() => {
		if (!sortKey || !sortDirection) return models;
		return [...models].sort((a, b) => {
			const aVal = getMetricValue(a, sortKey);
			const bVal = getMetricValue(b, sortKey);
			const diff = aVal - bVal;
			return sortDirection === "asc" ? diff : -diff;
		});
	}, [models, sortKey, sortDirection, getMetricValue]);

	const handleSort = (key: MetricKey) => {
		if (sortKey === key) {
			if (sortDirection === "asc") setSortDirection("desc");
			else if (sortDirection === "desc") { setSortKey(null); setSortDirection(null); }
			else setSortDirection("asc");
		} else {
			setSortKey(key);
			setSortDirection("asc");
		}
	};

	const toggleModelExpanded = (e: React.MouseEvent, modelId: string) => {
		e.stopPropagation();
		setExpandedModelIds((prev) => {
			const next = new Set(prev);
			if (next.has(modelId)) next.delete(modelId);
			else next.add(modelId);
			return next;
		});
	};

	return (
		<div className="w-full">
			<MetricInfoModal metric={selectedMetric} onClose={() => setSelectedMetric(null)} />
			<DeleteConfirmModal 
				isOpen={deleteModal.isOpen} 
				onClose={() => setDeleteModal(p => ({...p, isOpen: false}))}
				onConfirm={(delFile) => { onDelete(deleteModal.modelId, delFile); setDeleteModal(p => ({...p, isOpen: false})); }}
				modelName={deleteModal.modelName}
			/>

			{/* Action Menu Backdrop */}
			{menuState.isOpen && (
				<button 
					type="button"
					className="fixed inset-0 z-[60] w-full h-full cursor-default border-none p-0 m-0 bg-transparent" 
					onClick={() => setMenuState(p => ({...p, isOpen: false}))}
					aria-label="Close menu"
				>
					<div 
						className="absolute bg-white dark:bg-slate-800 rounded-xl shadow-xl border border-slate-200 dark:border-slate-700 py-1 w-40"
						style={{ top: menuState.y, left: menuState.x - 160 }}
						onClick={e => e.stopPropagation()}
					>
						<button
							type="button"
							onClick={() => {
								setDeleteModal({ isOpen: true, modelId: menuState.modelId, modelName: menuState.modelName });
								setMenuState(p => ({...p, isOpen: false}));
							}}
							className="w-full text-left px-4 py-2.5 text-sm text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/10 flex items-center gap-2"
						>
							<Trash2 size={14} /> Delete
						</button>
					</div>
				</button>
			)}

			<div className="overflow-x-auto min-h-[400px]">
				<table className="w-full text-left text-sm table-fixed min-w-[1000px]">
					<thead className="bg-slate-50/50 dark:bg-slate-900/30 text-slate-500 dark:text-slate-400 border-b border-slate-200 dark:border-white/5">
						<tr>
							<th className="w-12 px-6 py-4"></th>
							<th className="px-6 py-4 font-semibold uppercase tracking-wider text-[11px] w-[25%] text-left relative">
								Model Name
								<div className="absolute right-0 top-1/4 bottom-1/4 w-px bg-slate-200 dark:bg-white/10" />
							</th>
							<th className="px-6 py-4 font-semibold uppercase tracking-wider text-[11px] text-center w-[12%] relative">
								Source
								<div className="absolute right-0 top-1/4 bottom-1/4 w-px bg-slate-200 dark:bg-white/10" />
							</th>
							{METRIC_OPTIONS.map((metric) => (
								<th key={metric.value} className="px-3 py-4 font-semibold uppercase tracking-wider text-[11px] text-center group w-[12%] relative">
									<div className="flex items-center justify-center gap-1 h-full min-w-0">
										{/* Info Icon */}
										<button 
											type="button"
											onClick={() => setSelectedMetric(metric)}
											className="p-1 hover:bg-slate-100 dark:hover:bg-slate-700 rounded transition-colors text-slate-300 dark:text-slate-600 hover:text-blue-500 shrink-0"
											title={`About ${metric.label}`}
										>
											<Info size={14} />
										</button>

										{/* Label + Sort */}
										<button 
											type="button"
											onClick={() => handleSort(metric.value)}
											className={`flex items-center gap-1.5 hover:text-blue-500 transition-colors min-w-0 ${sortKey === metric.value ? "text-blue-600 dark:text-blue-400" : "text-slate-500 dark:text-slate-400"}`}
										>
											<span className="truncate overflow-hidden">{metric.label}</span>
											<div className="flex flex-col -space-y-1 shrink-0">
												<ChevronUp 
													size={10} 
													className={sortKey === metric.value && sortDirection === "asc" ? "text-blue-500" : "opacity-30"} 
												/>
												<ChevronDown 
													size={10} 
													className={sortKey === metric.value && sortDirection === "desc" ? "text-blue-500" : "opacity-30"} 
												/>
											</div>
										</button>
									</div>
									{/* Separator */}
									<div className="absolute right-0 top-1/4 bottom-1/4 w-px bg-slate-200 dark:bg-white/10" />
								</th>
							))}
							<th className="px-6 py-4 font-semibold uppercase tracking-wider text-[11px] text-right w-16"></th>
						</tr>
					</thead>
					<tbody className="divide-y divide-slate-100 dark:divide-white/5">
						{sortedModels.map((model) => {
							const isExpanded = expandedModelIds.has(model.id);
							const isMenuOpen = menuState.isOpen && menuState.modelId === model.id;
							return (
								<React.Fragment key={model.id}>
									<tr 
										className={`group transition-colors border-transparent hover:bg-slate-50/50 dark:hover:bg-white/5 ${selectedId === model.id ? "bg-blue-50/50 dark:bg-blue-900/10" : ""} ${isExpanded ? "bg-slate-50/30 dark:bg-slate-900/5 shadow-inner" : ""}`}
										onClick={() => onSelect(model.id === selectedId ? null : model.id)}
										role="button"
										tabIndex={0}
										onKeyDown={(e) => {
											if (e.key === "Enter" || e.key === " ") {
												onSelect(model.id === selectedId ? null : model.id);
											}
										}}
									>
										<td className="px-6 py-4 text-center">
											<button
												type="button"
												onClick={(e) => toggleModelExpanded(e, model.id)}
												className="p-1 hover:bg-slate-200 dark:hover:bg-slate-700 rounded transition-colors text-slate-400"
											>
												{isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
											</button>
										</td>
										<td className="px-6 py-4">
											<div className="flex items-center gap-2 overflow-hidden">
												<div className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: stringToColor(model.id) }}></div>
												<span className="font-medium text-slate-700 dark:text-slate-200 truncate font-semibold" title={model.name}>{model.name}</span>
												{model.is_missing && <span className="text-[9px] bg-red-100 text-red-600 px-1 rounded font-bold uppercase">offline</span>}
												{!isRunSpecific && model.run_count && model.run_count > 0 ? (
													<span className="text-[9px] bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 px-1.5 py-0.5 rounded font-bold border border-slate-200 dark:border-white/5">
														{model.run_count} {model.run_count === 1 ? "RUN" : "RUNS"}
													</span>
												) : null}
											</div>
										</td>
										<td className="px-6 py-4 text-center">
											<span className="px-2 py-0.5 rounded-full text-[10px] bg-slate-100 dark:bg-slate-800 text-slate-500 border border-slate-200 dark:border-white/5 font-medium">
												{model.source}
											</span>
										</td>
										{METRIC_OPTIONS.map((metric) => {
											const val = getMetricValue(model, metric.value);
											return (
												<td key={metric.value} className="px-6 py-4 text-center font-mono text-xs">
													<div className="flex flex-col">
														<span className={val > 0 ? "text-slate-600 dark:text-slate-300 text-sm" : "text-slate-300 dark:text-slate-700 opacity-30"}>
															{val > 0 ? val.toFixed(3) : "—"}
														</span>
													</div>
												</td>
											);
										})}
										<td className="px-6 py-4 text-right">
											<button
												type="button"
												onClick={(e) => {
													e.stopPropagation();
													const rect = e.currentTarget.getBoundingClientRect();
													setMenuState({
														isOpen: true,
														x: rect.right,
														y: rect.bottom + 5,
														modelId: model.id,
														modelName: model.name,
													});
												}}
												className={`p-1.5 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition-all hover:bg-slate-100 dark:hover:bg-slate-700 rounded-full inline-block ${
													isMenuOpen ? "bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-200" : ""
												}`}
											>
												<MoreVertical size={16} />
											</button>
										</td>
									</tr>
									{isExpanded && (
										<tr>
											<td colSpan={METRIC_OPTIONS.length + 4} className="p-0 border-none">
												<ModelHistory model={model} />
											</td>
										</tr>
									)}
								</React.Fragment>
							);
						})}
					</tbody>
				</table>
			</div>
		</div>
	);
};
