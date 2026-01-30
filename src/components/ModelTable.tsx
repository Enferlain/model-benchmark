import {
	ChevronDown,
	ChevronUp,
	ExternalLink,
	Info,
	MoreVertical,
	Trash2,
	X,
} from "lucide-react";
import type React from "react";
import { useEffect, useMemo, useState } from "react";
import { METRIC_OPTIONS } from "../constants";
import type { MetricKey, MetricOption, ModelData } from "../types";
import { stringToColor } from "../utils/colorUtils";
import { DeleteConfirmModal } from "./DeleteConfirmModal";

interface ModelTableProps {
	models: ModelData[];
	onDelete: (id: string, deleteFile: boolean) => void;
	selectedId: string | null;
	onSelect: (id: string | null) => void;
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
			{/* Backdrop */}
			<button
				type="button"
				className="absolute inset-0 bg-black/50 backdrop-blur-sm w-full h-full border-none p-0 m-0"
				onClick={onClose}
				aria-label="Close modal"
			/>
			{/* Modal */}
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
						<div className="prose prose-sm dark:prose-invert max-w-none">
							{metric.extendedDescription.split("\n\n").map((paragraph) => (
								<p
									key={paragraph.substring(0, 50)}
									className="text-slate-600 dark:text-slate-300 whitespace-pre-wrap text-sm leading-relaxed mb-3"
								>
									{paragraph.split("**").map((part, j) =>
										j % 2 === 1 ? (
											<strong
												key={part}
												className="text-slate-800 dark:text-slate-100"
											>
												{part}
											</strong>
										) : (
											part
										),
									)}
								</p>
							))}
						</div>
					)}
				</div>
			</div>
		</div>
	);
};

export const ModelTable: React.FC<ModelTableProps> = ({
	models,
	onDelete,
	selectedId,
	onSelect,
}) => {
	const [sortKey, setSortKey] = useState<MetricKey | null>(null);
	const [sortDirection, setSortDirection] = useState<SortDirection>(null);
	const [selectedMetric, setSelectedMetric] = useState<MetricOption | null>(
		null,
	);

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

	const getMetricValue = useMemo(() => {
		return (model: ModelData, key: MetricKey): number => {
			// Prefer metrics dict, fallback to direct properties for backwards compatibility
			if (model.metrics && key in model.metrics) {
				return model.metrics[key];
			}
			// Fallback to direct property access
			return (model as unknown as Record<string, number>)[key] ?? 0;
		};
	}, []);

	const handleSort = (key: MetricKey) => {
		if (sortKey === key) {
			// Cycle: asc -> desc -> null
			if (sortDirection === "asc") {
				setSortDirection("desc");
			} else if (sortDirection === "desc") {
				setSortKey(null);
				setSortDirection(null);
			} else {
				setSortDirection("asc");
			}
		} else {
			setSortKey(key);
			setSortDirection("asc");
		}
	};

	const handleActionClick = (e: React.MouseEvent, model: ModelData) => {
		e.stopPropagation();
		const rect = e.currentTarget.getBoundingClientRect();
		// Position menu to the left of the button, slightly down
		setMenuState({
			isOpen: true,
			x: rect.left - 120, // adjust width of menu
			y: rect.bottom + 5,
			modelId: model.id,
			modelName: model.name,
		});
	};

	const sortedModels = useMemo(() => {
		if (!sortKey || !sortDirection) return models;

		return [...models].sort((a, b) => {
			const aVal = getMetricValue(a, sortKey);
			const bVal = getMetricValue(b, sortKey);
			const diff = aVal - bVal;
			return sortDirection === "asc" ? diff : -diff;
		});
	}, [models, sortKey, sortDirection, getMetricValue]);

	return (
		<>
			<div className="rounded-3xl shadow-xl shadow-slate-200/50 dark:shadow-black/20 border border-white/60 dark:border-white/5 bg-white/60 dark:bg-slate-800/40 backdrop-blur-xl overflow-hidden transition-all">
				<div className="px-6 py-5 border-b border-slate-200/50 dark:border-white/5 flex items-center justify-between">
					<h3 className="font-medium text-slate-800 dark:text-slate-200">
						Model Data
					</h3>
				</div>
				<div className="overflow-x-auto min-h-[400px]">
					<table className="w-full text-left text-sm">
						<thead className="bg-slate-50/50 dark:bg-slate-900/30 border-b border-slate-200/50 dark:border-white/5 text-slate-500 dark:text-slate-400 backdrop-blur-sm">
							<tr>
								<th className="px-6 py-4 font-semibold uppercase tracking-wider text-[11px] text-center w-4"></th>
								<th className="px-6 py-4 font-semibold uppercase tracking-wider text-[11px]">
									Model Name
								</th>
								<th className="px-6 py-4 font-semibold uppercase tracking-wider text-[11px] text-center">
									Source
								</th>
								{METRIC_OPTIONS.map((metric) => (
									<th
										key={metric.value}
										className="px-6 py-4 font-semibold uppercase tracking-wider text-[11px] text-center"
									>
										<div className="flex items-center justify-center gap-1">
											<button
												type="button"
												onClick={(e) => {
													e.stopPropagation();
													setSelectedMetric(metric);
												}}
												className="p-0.5 hover:bg-slate-200 dark:hover:bg-slate-600 rounded transition-colors"
												title="Learn about this metric"
											>
												<Info
													size={12}
													className="text-slate-400 hover:text-blue-500"
												/>
											</button>
											<button
												type="button"
												className="hover:text-slate-700 dark:hover:text-slate-200 transition-colors select-none"
												onClick={() => handleSort(metric.value)}
												title={metric.description}
											>
												{metric.label.split(" ")[0]}
											</button>
											<button
												type="button"
												className="flex flex-col -space-y-1 hover:bg-slate-200 dark:hover:bg-slate-700/50 p-0.5 rounded transition-colors"
												onClick={() => handleSort(metric.value)}
												aria-label={`Sort by ${metric.label}`}
											>
												<ChevronUp
													size={12}
													className={
														sortKey === metric.value && sortDirection === "asc"
															? "text-blue-500"
															: "opacity-30"
													}
												/>
												<ChevronDown
													size={12}
													className={
														sortKey === metric.value && sortDirection === "desc"
															? "text-blue-500"
															: "opacity-30"
													}
												/>
											</button>
										</div>
									</th>
								))}
								<th className="px-6 py-4 font-semibold uppercase tracking-wider text-[11px] text-center">
									Actions
								</th>
							</tr>
						</thead>
						<tbody className="divide-y divide-slate-200/50 dark:divide-white/5">
							{sortedModels.map((model) => (
								<tr
									key={model.id}
									className={`cursor-pointer transition-colors group ${
										selectedId === model.id
											? "bg-blue-50 dark:bg-blue-900/20"
											: "hover:bg-white/50 dark:hover:bg-white/5"
									}`}
									onClick={() =>
										onSelect(selectedId === model.id ? null : model.id)
									}
								>
									<td className="px-6 py-4 text-center">
										<div
											className="w-3 h-3 rounded-full mx-auto"
											style={{
												backgroundColor: stringToColor(model.hash || model.id),
											}}
										></div>
									</td>
									<td className="px-6 py-4">
										<div className="flex items-center gap-2">
											{/* Status Indicator */}
											{model.is_missing && (
												<span className="px-1.5 py-0.5 bg-red-100 text-red-600 text-[10px] rounded font-bold border border-red-200">
													offline
												</span>
											)}
											<span
												className={`font-medium ${selectedId === model.id ? "text-blue-700 dark:text-blue-300" : "text-slate-800 dark:text-slate-200"}`}
											>
												{model.name}
											</span>
											<a
												href={model.url}
												target="_blank"
												rel="noopener noreferrer"
												className="text-slate-400 hover:text-blue-600 dark:hover:text-blue-400 opacity-0 group-hover:opacity-100 transition-opacity"
											>
												<ExternalLink size={12} />
											</a>
										</div>
									</td>
									<td className="px-6 py-4 text-center">
										<span
											className={`inline-flex items-center px-2.5 py-1 rounded-full text-[10px] font-medium border shadow-sm ${
												model.source === "Civitai"
													? "bg-blue-50/80 text-blue-700 border-blue-100 dark:bg-blue-500/10 dark:text-blue-300 dark:border-blue-500/20"
													: model.source === "HuggingFace"
														? "bg-pink-50/80 text-pink-700 border-pink-100 dark:bg-pink-500/10 dark:text-pink-300 dark:border-pink-500/20"
														: "bg-slate-50/80 text-slate-700 border-slate-100 dark:bg-slate-500/10 dark:text-slate-300 dark:border-slate-500/20"
											}`}
										>
											{model.source}
										</span>
									</td>
									{METRIC_OPTIONS.map((metric) => (
										<td
											key={metric.value}
											className="px-6 py-4 text-center font-mono text-slate-600 dark:text-slate-400 text-xs"
										>
											{getMetricValue(model, metric.value).toFixed(3)}
										</td>
									))}
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
										colSpan={METRIC_OPTIONS.length + 4}
										className="px-6 py-12 text-center text-slate-400"
									>
										No models added. Add a URL or Import Locally to see
										benchmarks.
									</td>
								</tr>
							)}
						</tbody>
					</table>
				</div>
			</div>

			{/* Metric Info Modal */}
			<MetricInfoModal
				metric={selectedMetric}
				onClose={() => setSelectedMetric(null)}
			/>

			{/* Action Menu (Fixed Position) */}
			{menuState.isOpen && (
				<div className="fixed inset-0 z-[60] flex items-start justify-start">
					{/* Invisible Full Screen Closer */}
					<button
						type="button"
						className="absolute inset-0 w-full h-full border-none p-0 m-0 bg-transparent"
						onClick={() => setMenuState((prev) => ({ ...prev, isOpen: false }))}
						aria-label="Close menu"
					/>

					{/* The Menu */}
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

			{/* Delete Modal */}
			<DeleteConfirmModal
				isOpen={deleteModal.isOpen}
				onClose={() => setDeleteModal({ ...deleteModal, isOpen: false })}
				modelName={deleteModal.modelName}
				onConfirm={() => {
					onDelete(deleteModal.modelId, false); // Always false
					setDeleteModal({ ...deleteModal, isOpen: false });
				}}
			/>
		</>
	);
};
