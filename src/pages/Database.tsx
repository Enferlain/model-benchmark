import {
	Archive,
	ArrowUpDown,
	Calendar,
	ChevronDown,
	ChevronRight,
	Database as DatabaseIcon,
	FileText,
	MoreVertical,
	RefreshCw,
	Search,
	Table,
	Trash2,
} from "lucide-react";
import React, { useCallback, useEffect, useMemo, useState } from "react";
import { DeleteConfirmModal } from "../components/DeleteConfirmModal";
import { useData } from "../context/DataContext";
import { deleteModel, scanModels } from "../services/api";
import type { ModelData } from "../types";

interface ModelResultData {
	model_hash: string;
	model_name: string;
	metrics: Record<string, number>;
	image_count: number;
}

interface BenchmarkRun {
	id: number;
	timestamp: string;
	parameters: any;
	prompts: string[];
	prompt_set_id: string | null;
	results: ModelResultData[];
}

type SortField = "name" | "model_type" | "prediction_type";

export default function Database() {
	const {
		models,
		runs,
		isLoadingModels: isLoading,
		refreshAll: loadData,
	} = useData();

	const [activeTab, setActiveTab] = useState<"models" | "runs">(
		() =>
			(localStorage.getItem("database_activeTab") as "models" | "runs") ||
			"models",
	);

	// Expanded runs state
	const [expandedRunIds, setExpandedRunIds] = useState<Set<number>>(new Set());

	// Sorting State
	const [sortField, setSortField] = useState<SortField>(
		() => (localStorage.getItem("database_sortField") as SortField) || "name",
	);
	const [sortDirection, setSortDirection] = useState<"asc" | "desc">(
		() =>
			(localStorage.getItem("database_sortDirection") as "asc" | "desc") ||
			"asc",
	);
	const [searchQuery, setSearchQuery] = useState(
		() => localStorage.getItem("database_searchQuery") || "",
	);

	// Persistence Effects
	useEffect(() => {
		localStorage.setItem("database_activeTab", activeTab);
	}, [activeTab]);

	useEffect(() => {
		localStorage.setItem("database_sortField", sortField);
	}, [sortField]);

	useEffect(() => {
		localStorage.setItem("database_sortDirection", sortDirection);
	}, [sortDirection]);

	useEffect(() => {
		localStorage.setItem("database_searchQuery", searchQuery);
	}, [searchQuery]);

	// Actions State
	const [deleteModal, setDeleteModal] = useState<{
		isOpen: boolean;
		modelId: string;
		modelName: string;
	}>({
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
	}>({
		isOpen: false,
		x: 0,
		y: 0,
		modelId: "",
		modelName: "",
	});

	// Close menu on scroll or resize
	useEffect(() => {
		const closeMenu = () => {
			if (menuState.isOpen)
				setMenuState((prev) => ({ ...prev, isOpen: false }));
		};
		window.addEventListener("scroll", closeMenu, true);
		window.addEventListener("resize", closeMenu);
		return () => {
			window.removeEventListener("scroll", closeMenu, true);
			window.removeEventListener("resize", closeMenu);
		};
	}, [menuState.isOpen]);

	const handleActionClick = (e: React.MouseEvent, model: ModelData) => {
		e.stopPropagation();
		const rect = e.currentTarget.getBoundingClientRect();
		setMenuState({
			isOpen: true,
			x: rect.left - 120,
			y: rect.bottom + 5,
			modelId: model.id,
			modelName: model.name,
		});
	};

	const handleDeleteModel = async (id: string) => {
		try {
			await deleteModel(id, false);
			loadData();
		} catch (error) {
			console.error("Error deleting model:", error);
		}
	};

	const handleSort = (field: SortField) => {
		if (sortField === field) {
			setSortDirection(sortDirection === "asc" ? "desc" : "asc");
		} else {
			setSortField(field);
			setSortDirection("asc");
		}
	};

	const sortedModels = useMemo(() => {
		let result = [...models];

		if (searchQuery) {
			const q = searchQuery.toLowerCase();
			result = result.filter(
				(m) =>
					m.name.toLowerCase().includes(q) ||
					m.filename?.toLowerCase().includes(q) ||
					m.hash?.toLowerCase().includes(q) ||
					m.path?.toLowerCase().includes(q),
			);
		}

		return result.sort((a, b) => {
			if (a.is_missing && !b.is_missing) return 1;
			if (!a.is_missing && b.is_missing) return -1;

			let aValue = (a[sortField as keyof ModelData] as string) || "";
			let bValue = (b[sortField as keyof ModelData] as string) || "";

			if (aValue === undefined || aValue === null) aValue = "";
			if (bValue === undefined || bValue === null) bValue = "";

			const comparison = aValue.localeCompare(bValue);
			return sortDirection === "asc" ? comparison : -comparison;
		});
	}, [models, sortField, sortDirection, searchQuery]);

	const formatDate = (dateStr: string) => {
		return new Date(dateStr).toLocaleString();
	};

	const toggleRunExpanded = (runId: number) => {
		setExpandedRunIds((prev) => {
			const next = new Set(prev);
			if (next.has(runId)) {
				next.delete(runId);
			} else {
				next.add(runId);
			}
			return next;
		});
	};

	const SortIcon = ({ field }: { field: SortField }) => {
		if (sortField !== field)
			return <ArrowUpDown size={14} className="text-slate-300" />;
		return (
			<ArrowUpDown
				size={14}
				className={
					sortDirection === "asc"
						? "text-indigo-600 rotate-180"
						: "text-indigo-600"
				}
			/>
		);
	};

	return (
		<div className="max-w-[1800px] mx-auto px-6 py-8">
			<div className="flex flex-col mb-8">
				<h2 className="text-2xl font-bold text-slate-800 dark:text-slate-100 flex items-center gap-2">
					<DatabaseIcon className="w-6 h-6 text-indigo-500" />
					Database Management
				</h2>
				<p className="text-slate-500 dark:text-slate-400 mt-1">
					View and manage local models and historical benchmark runs.
				</p>
			</div>

			<div className="flex items-center gap-4 mb-6 border-b border-slate-200 dark:border-slate-700">
				<button
					type="button"
					onClick={() => setActiveTab("models")}
					className={`flex items-center gap-2 px-4 py-3 border-b-2 transition-colors ${
						activeTab === "models"
							? "border-indigo-500 text-indigo-600 dark:text-indigo-400 font-medium"
							: "border-transparent text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200"
					}`}
				>
					<Archive size={18} />
					Models
					<span className="bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 px-2 py-0.5 rounded-full text-xs">
						{models.length}
					</span>
				</button>
				<button
					type="button"
					onClick={() => setActiveTab("runs")}
					className={`flex items-center gap-2 px-4 py-3 border-b-2 transition-colors ${
						activeTab === "runs"
							? "border-indigo-500 text-indigo-600 dark:text-indigo-400 font-medium"
							: "border-transparent text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200"
					}`}
				>
					<Table size={18} />
					Benchmark Runs
					<span className="bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 px-2 py-0.5 rounded-full text-xs">
						{runs.length}
					</span>
				</button>

				<div className="ml-auto flex items-center gap-2">
					<div className="relative mr-2">
						<Search
							size={16}
							className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"
						/>
						<input
							type="text"
							placeholder="Search models..."
							value={searchQuery}
							onChange={(e) => setSearchQuery(e.target.value)}
							className="pl-9 pr-4 py-1.5 text-sm rounded-md border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
						/>
					</div>
					<button
						type="button"
						onClick={async () => {
							try {
								await scanModels();
								await loadData();
							} catch (e) {
								console.error("Scan failed", e);
							}
						}}
						disabled={isLoading}
						className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-indigo-50 hover:bg-indigo-100 text-indigo-700 dark:bg-indigo-900/20 dark:hover:bg-indigo-900/40 dark:text-indigo-300 text-sm font-medium transition-colors"
						title="Scan disk for new models and update database"
					>
						<RefreshCw size={14} className={isLoading ? "animate-spin" : ""} />
						Rescan Models
					</button>

					<div className="h-6 w-px bg-slate-200 dark:bg-slate-700 mx-2"></div>

					<button
						type="button"
						onClick={loadData}
						disabled={isLoading}
						className="p-2 text-slate-500 hover:text-indigo-600 dark:text-slate-400 dark:hover:text-indigo-400 transition-colors"
						title="Refresh Data"
					>
						<RefreshCw size={18} className={isLoading ? "animate-spin" : ""} />
					</button>
				</div>
			</div>

			<div className="bg-white dark:bg-slate-800 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700 overflow-hidden">
				{isLoading &&
				(activeTab === "models" ? models.length === 0 : runs.length === 0) ? (
					<div className="p-12 flex justify-center text-slate-400">
						Loading...
					</div>
				) : (
					<>
						{activeTab === "models" && (
							<div className="overflow-x-auto min-h-[400px]">
								<table className="w-full text-left text-sm table-fixed">
									<thead className="bg-slate-50 dark:bg-slate-700/50 text-slate-500 dark:text-slate-400 border-b border-slate-200 dark:border-slate-700">
										<tr>
											<th
												className="px-6 py-4 font-medium cursor-pointer hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors group w-[35%]"
												onClick={() => handleSort("name")}
											>
												<div className="flex items-center gap-2">
													Model Name
													<SortIcon field="name" />
												</div>
											</th>
											<th
												className="px-6 py-4 font-medium cursor-pointer hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors group w-[12%]"
												onClick={() => handleSort("model_type")}
											>
												<div className="flex items-center gap-2">
													Type
													<SortIcon field="model_type" />
												</div>
											</th>
											<th
												className="px-6 py-4 font-medium cursor-pointer hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors group w-[15%]"
												onClick={() => handleSort("prediction_type")}
											>
												<div className="flex items-center gap-2">
													Prediction
													<SortIcon field="prediction_type" />
												</div>
											</th>
											<th className="px-6 py-4 font-medium w-[15%]">
												Hash / ID
											</th>
											<th className="px-6 py-4 font-medium w-[13%]">Metrics</th>
											<th className="px-6 py-4 font-medium w-[10%] text-center">
												Actions
											</th>
										</tr>
									</thead>
									<tbody className="divide-y divide-slate-100 dark:divide-slate-700">
										{sortedModels.map((model) => (
											<tr
												key={model.id}
												className={`hover:bg-slate-50 dark:hover:bg-slate-700/30 transition-colors ${model.is_missing ? "opacity-60 grayscale" : ""}`}
											>
												<td className="px-6 py-4 font-medium text-slate-900 dark:text-slate-200">
													<div className="flex items-center gap-2">
														{model.name}
														{model.is_missing && (
															<span className="px-1.5 py-0.5 rounded text-[10px] bg-slate-200 text-slate-600 dark:bg-slate-700 dark:text-slate-400 font-bold border border-slate-300 dark:border-slate-600">
																OFFLINE
															</span>
														)}
													</div>
													<div
														className="text-xs font-normal text-slate-500 truncate block"
														title={model.path}
													>
														{model.path}
													</div>
												</td>
												<td className="px-6 py-4">
													<span
														className={`inline-flex items-center px-2 py-1 rounded text-xs font-medium ${
															model.model_type === "sdxl"
																? "bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300"
																: "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300"
														}`}
													>
														{model.model_type || "Unknown"}
													</span>
												</td>
												<td className="px-6 py-4 text-slate-600 dark:text-slate-400">
													<div className="flex items-center gap-2">
														<span>{model.prediction_type || "epsilon"}</span>
														{model.ztsnr && (
															<span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-indigo-100 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-300">
																ZTSNR
															</span>
														)}
													</div>
												</td>
												<td className="px-6 py-4 font-mono text-xs text-slate-500">
													<div
														className="truncate max-w-[140px]"
														title={model.hash || model.id}
													>
														{model.hash || model.id}
													</div>
												</td>
												<td className="px-6 py-4 text-slate-600 dark:text-slate-400">
													<div className="flex gap-4">
														<span title="Image Count">
															{model.image_count} imgs
														</span>
														{model.metrics &&
															Object.keys(model.metrics).length > 0 && (
																<span className="text-green-600 dark:text-green-400">
																	Has Metrics
																</span>
															)}
													</div>
												</td>
												<td className="px-6 py-4 text-center">
													<button
														type="button"
														onClick={(e) => handleActionClick(e, model)}
														className="p-2 text-slate-400 hover:text-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 dark:hover:text-slate-200 rounded-full transition-all"
														title="Actions"
													>
														<MoreVertical size={16} />
													</button>
												</td>
											</tr>
										))}
										{models.length === 0 && (
											<tr>
												<td
													colSpan={6}
													className="px-6 py-12 text-center text-slate-500 italic"
												>
													No models found in database.
												</td>
											</tr>
										)}
									</tbody>
								</table>
							</div>
						)}

						{activeTab === "runs" && (
							<div className="overflow-x-auto">
								<table className="w-full text-left text-sm">
									<thead className="bg-slate-50 dark:bg-slate-700/50 text-slate-500 dark:text-slate-400 border-b border-slate-200 dark:border-slate-700">
										<tr>
											<th className="px-6 py-4 font-medium">Run ID</th>
											<th className="px-6 py-4 font-medium">Timestamp</th>
											<th className="px-6 py-4 font-medium">Prompt Set</th>
											<th className="px-6 py-4 font-medium">Parameters</th>
											<th className="px-6 py-4 font-medium">Details</th>
										</tr>
									</thead>
									<tbody className="divide-y divide-slate-100 dark:divide-slate-700">
										{runs.map((run) => {
											const isExpanded = expandedRunIds.has(run.id);
											return (
												<React.Fragment key={run.id}>
													<tr
														className="hover:bg-slate-50 dark:hover:bg-slate-700/30 transition-colors cursor-pointer"
														onClick={() => toggleRunExpanded(run.id)}
													>
														<td className="px-6 py-4 font-mono text-xs text-slate-500">
															<div className="flex items-center gap-2">
																{isExpanded ? (
																	<ChevronDown size={16} />
																) : (
																	<ChevronRight size={16} />
																)}
																#{run.id}
															</div>
														</td>
														<td className="px-6 py-4 text-slate-700 dark:text-slate-300">
															<div className="flex items-center gap-2">
																<Calendar
																	size={14}
																	className="text-slate-400"
																/>
																{formatDate(run.timestamp)}
															</div>
														</td>
														<td className="px-6 py-4 text-slate-600 dark:text-slate-400">
															<div className="flex items-center gap-2">
																<FileText
																	size={14}
																	className="text-slate-400"
																/>
																{run.prompt_set_id
																	? `Set: ${run.prompt_set_id}`
																	: `Ad-hoc (${run.prompts?.length || 0} prompts)`}
															</div>
														</td>
														<td className="px-6 py-4 text-slate-600 dark:text-slate-400 text-xs">
															<div className="grid grid-cols-2 gap-x-4 gap-y-1">
																<span>Steps: {run.parameters?.steps}</span>
																<span>
																	CFG: {run.parameters?.guidance_scale}
																</span>
																<span>Seed: {run.parameters?.seed}</span>
																<span>
																	Dim: {run.parameters?.width}x
																	{run.parameters?.height}
																</span>
															</div>
														</td>
														<td className="px-6 py-4">
															<span className="text-indigo-600 dark:text-indigo-400 text-xs font-medium">
																{isExpanded ? "Hide" : "View"} (
																{run.results?.length || 0} models)
															</span>
														</td>
													</tr>
													{isExpanded && (
														<tr>
															<td
																colSpan={5}
																className="px-6 py-4 bg-slate-50 dark:bg-slate-800/50"
															>
																<div className="space-y-4">
																	<div>
																		<h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
																			Models & Metrics
																		</h4>
																		{run.results && run.results.length > 0 ? (
																			<table className="w-full text-xs border border-slate-200 dark:border-slate-700 rounded">
																				<thead className="bg-slate-100 dark:bg-slate-700">
																					<tr>
																						<th className="px-3 py-2 text-left font-medium">
																							Model
																						</th>
																						<th className="px-3 py-2 text-left font-medium">
																							Accuracy
																						</th>
																						<th className="px-3 py-2 text-left font-medium">
																							Diversity
																						</th>
																						<th className="px-3 py-2 text-left font-medium">
																							Images
																						</th>
																					</tr>
																				</thead>
																				<tbody className="divide-y divide-slate-200 dark:divide-slate-700">
																					{run.results.map((res) => (
																						<tr key={res.model_hash}>
																							<td className="px-3 py-2 text-slate-800 dark:text-slate-200">
																								{res.model_name}
																							</td>
																							<td className="px-3 py-2 text-slate-600 dark:text-slate-400">
																								{res.metrics?.accuracy?.toFixed(
																									3,
																								) ?? "-"}
																							</td>
																							<td className="px-3 py-2 text-slate-600 dark:text-slate-400">
																								{res.metrics?.diversity?.toFixed(
																									3,
																								) ?? "-"}
																							</td>
																							<td className="px-3 py-2 text-slate-600 dark:text-slate-400">
																								{res.image_count}
																							</td>
																						</tr>
																					))}
																				</tbody>
																			</table>
																		) : (
																			<p className="text-slate-500 italic text-xs">
																				No model results recorded for this run.
																			</p>
																		)}
																	</div>

																	<div>
																		<h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
																			Prompts ({run.prompts?.length || 0})
																		</h4>
																		<div className="text-xs text-slate-600 dark:text-slate-400 space-y-1 max-h-32 overflow-y-auto max-w-full">
																			{run.prompts
																				?.slice(0, 5)
																				.map((prompt, idx) => (
																					<div
																						key={idx}
																						className="break-words"
																					>
																						• {prompt}
																					</div>
																				))}
																			{(run.prompts?.length || 0) > 5 && (
																				<div className="text-slate-400 italic">
																					+{run.prompts.length - 5} more
																				</div>
																			)}
																		</div>
																	</div>
																</div>
															</td>
														</tr>
													)}
												</React.Fragment>
											);
										})}
										{runs.length === 0 && (
											<tr>
												<td
													colSpan={5}
													className="px-6 py-12 text-center text-slate-500 italic"
												>
													No benchmark runs recorded yet.
												</td>
											</tr>
										)}
									</tbody>
								</table>
							</div>
						)}
					</>
				)}
			</div>

			{menuState.isOpen && (
				<div className="fixed inset-0 z-[60] flex items-start justify-start">
					<div
						className="absolute inset-0"
						onClick={() => setMenuState((prev) => ({ ...prev, isOpen: false }))}
					></div>

					<div
						className="absolute bg-white dark:bg-slate-800 rounded-xl shadow-xl border border-slate-200 dark:border-slate-700 py-1 w-40 animate-in fade-in zoom-in-95 duration-100 origin-top-right"
						style={{ top: menuState.y, left: menuState.x }}
					>
						<button
							type="button"
							onClick={() => {
								setDeleteModal({
									isOpen: true,
									modelId: menuState.modelId,
									modelName: menuState.modelName,
								});
								setMenuState((prev) => ({ ...prev, isOpen: false }));
							}}
							className="w-full text-left px-4 py-2.5 text-sm text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/10 flex items-center gap-2"
						>
							<Trash2 size={14} />
							Delete
						</button>
					</div>
				</div>
			)}

			<DeleteConfirmModal
				isOpen={deleteModal.isOpen}
				onClose={() => setDeleteModal({ ...deleteModal, isOpen: false })}
				modelName={deleteModal.modelName}
				onConfirm={() => {
					handleDeleteModel(deleteModal.modelId);
					setDeleteModal({ ...deleteModal, isOpen: false });
				}}
			/>
		</div>
	);
}
