import {
	Check,
	History,
	Library,
	ListChecks,
	Search,
	Settings2,
} from "lucide-react";
import { type FC, type ReactNode, useEffect, useRef, useState } from "react";
import type { BenchmarkRun } from "../../types";
import type { Preset } from "../TransferList/PresetMenu";

export interface ChartSource {
	type: "all" | "queue" | "run" | "preset";
	id?: string | number;
}

interface DatasetSelectorProps {
	activeSource: ChartSource;
	onSelect: (source: ChartSource) => void;
	searchQuery: string;
	onSearchChange: (query: string) => void;
	runs: BenchmarkRun[];
	presets: Preset[];
	isDarkMode?: boolean;
}

export function DatasetSelector({
	activeSource,
	onSelect,
	searchQuery,
	onSearchChange,
	runs,
	presets,
	isDarkMode,
}: DatasetSelectorProps) {
	const [isOpen, setIsOpen] = useState(false);
	const menuRef = useRef<HTMLDivElement>(null);

	// Close on click outside
	useEffect(() => {
		function handleClickOutside(event: MouseEvent) {
			if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
				setIsOpen(false);
			}
		}
		if (isOpen) {
			document.addEventListener("mousedown", handleClickOutside);
		}
		return () => document.removeEventListener("mousedown", handleClickOutside);
	}, [isOpen]);

	const getSourceLabel = () => {
		if (activeSource.type === "all") return "All Models";
		if (activeSource.type === "queue") return "Current Queue";
		if (activeSource.type === "run") {
			const run = runs.find((r) => r.id === activeSource.id);
			return run ? `Run #${run.id}` : "Benchmark Run";
		}
		if (activeSource.type === "preset") {
			const preset = presets.find((p) => p.id === activeSource.id);
			return preset ? preset.name : "Preset";
		}
		return "Select Dataset";
	};

	return (
		<div className="relative" ref={menuRef}>
			<button
				type="button"
				onClick={() => setIsOpen(!isOpen)}
				className={`flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs font-semibold transition-all ${
					isOpen
						? "bg-blue-500 text-white border-blue-600 shadow-lg shadow-blue-500/20"
						: isDarkMode
							? "bg-slate-800/50 border-white/10 text-slate-300 hover:bg-slate-700/50"
							: "bg-white border-slate-200 text-slate-600 hover:border-blue-500 hover:text-blue-500"
				}`}
			>
				<Settings2 size={14} />
				<span>{getSourceLabel()}</span>
			</button>

			{isOpen && (
				<div className="absolute top-full left-0 mt-2 w-72 bg-white dark:bg-slate-900 rounded-2xl shadow-2xl border border-slate-200 dark:border-white/10 z-[10001] overflow-hidden transform origin-top-left animate-in fade-in zoom-in-95 duration-100">
					{/* Search Header */}
					<div className="p-3 border-b border-slate-100 dark:border-white/5 bg-slate-50/50 dark:bg-white/5">
						<div className="relative">
							<Search
								className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"
								size={14}
							/>
							<input
								type="text"
								value={searchQuery}
								onChange={(e) => onSearchChange(e.target.value)}
								placeholder="Search models in chart..."
								className="w-full pl-9 pr-4 py-2 bg-white dark:bg-slate-800 border border-slate-200 dark:border-white/10 rounded-xl text-xs focus:outline-none focus:ring-2 focus:ring-blue-500/30 dark:text-white"
							/>
						</div>
					</div>

					<div className="max-h-[400px] overflow-y-auto p-2 space-y-4">
						{/* Views - Always show standard views unless search is very specific? 
                           Let's filter standard views too if they don't match.
                        */}
						{(() => {
							const query = searchQuery.toLowerCase().trim();
							const showAll = !query || "all models".includes(query);
							const showQueue = !query || "current queue".includes(query);

							if (!showAll && !showQueue) return null;

							return (
								<div>
									<h4 className="px-2 mb-1.5 text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-widest">
										Standard Views
									</h4>
									<div className="space-y-0.5">
										{showAll && (
											<SourceItem
												label="All Models"
												icon={<Library size={14} />}
												isActive={activeSource.type === "all"}
												onClick={() => {
													onSelect({ type: "all" });
													onSearchChange("");
													setIsOpen(false);
												}}
											/>
										)}
										{showQueue && (
											<SourceItem
												label="Current Queue"
												icon={<ListChecks size={14} />}
												isActive={activeSource.type === "queue"}
												onClick={() => {
													onSelect({ type: "queue" });
													onSearchChange("");
													setIsOpen(false);
												}}
											/>
										)}
									</div>
								</div>
							);
						})()}

						{/* Runs */}
						{(() => {
							const query = searchQuery.toLowerCase().trim();
							const filteredRuns = runs.filter(
								(run) =>
									`run #${run.id}`.includes(query) ||
									new Date(run.timestamp)
										.toLocaleDateString()
										.toLowerCase()
										.includes(query) ||
									run.results.some((res) =>
										res.model_name.toLowerCase().includes(query),
									),
							);

							if (filteredRuns.length === 0) return null;

							return (
								<div>
									<h4 className="px-2 mb-1.5 text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-widest flex items-center justify-between">
										<span>Benchmark Runs</span>
										<History size={10} />
									</h4>
									<div className="space-y-0.5">
										{filteredRuns.map((run) => (
											<SourceItem
												key={String(run.id)}
												label={`Run #${run.id} (${new Date(run.timestamp).toLocaleDateString()})`}
												isActive={
													activeSource.type === "run" &&
													activeSource.id === run.id
												}
												onClick={() => {
													onSelect({ type: "run", id: run.id });
													onSearchChange("");
													setIsOpen(false);
												}}
												sublabel={`${run.results.length} models`}
											/>
										))}
									</div>
								</div>
							);
						})()}

						{/* Presets */}
						{(() => {
							const query = searchQuery.toLowerCase().trim();
							const filteredPresets = presets.filter((preset) =>
								preset.name.toLowerCase().includes(query),
							);

							if (filteredPresets.length === 0) return null;

							return (
								<div>
									<h4 className="px-2 mb-1.5 text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-widest">
										Saved Presets
									</h4>
									<div className="space-y-0.5">
										{filteredPresets.map((preset) => (
											<SourceItem
												key={String(preset.id)}
												label={preset.name}
												isActive={
													activeSource.type === "preset" &&
													activeSource.id === preset.id
												}
												onClick={() => {
													onSelect({ type: "preset", id: preset.id });
													onSearchChange("");
													setIsOpen(false);
												}}
												sublabel={`${preset.modelIds.length} models`}
											/>
										))}
									</div>
								</div>
							);
						})()}

						{searchQuery.trim() &&
							runs.length === 0 &&
							presets.length === 0 &&
							!(
								"all models".includes(searchQuery.toLowerCase()) ||
								"current queue".includes(searchQuery.toLowerCase())
							) && (
								<div className="py-8 text-center text-slate-400 text-xs italic">
									No sources match "{searchQuery}"
								</div>
							)}
					</div>
				</div>
			)}
		</div>
	);
}

interface SourceItemProps {
	label: string;
	sublabel?: string;
	icon?: ReactNode;
	isActive: boolean;
	onClick: () => void;
}

const SourceItem: FC<SourceItemProps> = ({
	label,
	sublabel,
	icon,
	isActive,
	onClick,
}) => {
	return (
		<button
			type="button"
			onClick={onClick}
			className={`w-full flex items-center justify-between px-3 py-2 rounded-xl text-xs transition-all ${
				isActive
					? "bg-blue-500/10 text-blue-600 dark:bg-blue-500/20 dark:text-blue-400 font-semibold"
					: "text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-white/5"
			}`}
		>
			<div className="flex items-center gap-2.5 truncate">
				{icon && <span className="opacity-70">{icon}</span>}
				<div className="truncate text-left">
					<div className="truncate">{label}</div>
					{sublabel && (
						<div
							className={`text-[9px] ${isActive ? "text-blue-500/70" : "text-slate-400"}`}
						>
							{sublabel}
						</div>
					)}
				</div>
			</div>
			{isActive && <Check size={14} className="shrink-0" />}
		</button>
	);
};
