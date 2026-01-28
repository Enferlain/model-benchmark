import { FolderOpen, Plus, Trash2 } from "lucide-react";
import type React from "react";
import { useEffect, useRef, useState } from "react";

export interface Preset {
	id: string;
	name: string;
	modelIds: string[];
	createdAt: number;
}

interface PresetMenuProps {
	currentIds: string[];
	onLoad: (ids: string[]) => void;
}

const STORAGE_KEY = "model_benchmark_presets";

export function PresetMenu({ currentIds, onLoad }: PresetMenuProps) {
	const [isOpen, setIsOpen] = useState(false);
	const [showSaveInput, setShowSaveInput] = useState(false);
	const [newPresetName, setNewPresetName] = useState("");
	const [presets, setPresets] = useState<Preset[]>([]);
	const menuRef = useRef<HTMLDivElement>(null);

	// Load presets from local storage on mount
	useEffect(() => {
		try {
			const stored = localStorage.getItem(STORAGE_KEY);
			if (stored) {
				setPresets(JSON.parse(stored));
			}
		} catch (e) {
			console.error("Failed to load presets", e);
		}
	}, []);

	// Save presets to local storage whenever they change
	const savePresetsToStorage = (newPresets: Preset[]) => {
		setPresets(newPresets);
		localStorage.setItem(STORAGE_KEY, JSON.stringify(newPresets));
	};

	// Close when clicking outside
	useEffect(() => {
		function handleClickOutside(event: MouseEvent) {
			if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
				setIsOpen(false);
				setShowSaveInput(false);
			}
		}
		if (isOpen) {
			document.addEventListener("mousedown", handleClickOutside);
		}
		return () => document.removeEventListener("mousedown", handleClickOutside);
	}, [isOpen]);

	const handleSave = () => {
		if (!newPresetName.trim()) return;

		const newPreset: Preset = {
			id: crypto.randomUUID(),
			name: newPresetName.trim(),
			modelIds: [...currentIds],
			createdAt: Date.now(),
		};

		savePresetsToStorage([...presets, newPreset]);
		setNewPresetName("");
		setShowSaveInput(false);
	};

	const handleDelete = (e: React.MouseEvent, id: string) => {
		e.stopPropagation();
		savePresetsToStorage(presets.filter((p) => p.id !== id));
	};

	const handleLoad = (ids: string[]) => {
		onLoad(ids);
		setIsOpen(false);
	};

	return (
		<div className="relative" ref={menuRef}>
			<button
				type="button"
				onClick={() => setIsOpen(!isOpen)}
				className={`p-2 border rounded-lg transition-colors flex items-center gap-2 ${
					isOpen
						? "bg-blue-50 border-blue-200 text-blue-600 dark:bg-blue-900/30 dark:border-blue-700 dark:text-blue-400"
						: "bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-500 hover:text-blue-500 hover:border-blue-500"
				}`}
				title="Presets"
			>
				<FolderOpen size={16} />
			</button>

			{isOpen && (
				<div className="absolute right-0 top-full mt-2 w-64 bg-white dark:bg-slate-800 rounded-xl shadow-xl border border-slate-200 dark:border-slate-700 z-50 overflow-hidden transform origin-top-right animate-in fade-in zoom-in-95 duration-100">
					<div className="p-2 border-b border-slate-100 dark:border-slate-700/50 flex items-center justify-between">
						<span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
							Saved Queues
						</span>
						{!showSaveInput && (
							<button
								type="button"
								onClick={() => setShowSaveInput(true)}
								className="text-[10px] text-blue-500 hover:text-blue-600 font-medium flex items-center gap-1"
							>
								<Plus size={12} /> New
							</button>
						)}
					</div>

					<div className="p-2 flex flex-col gap-1 max-h-[300px] overflow-y-auto custom-scrollbar">
						{showSaveInput && (
							<div className="p-2 bg-slate-50 dark:bg-slate-700/50 rounded-lg mb-2 border border-blue-100 dark:border-blue-500/30">
								<input
									type="text"
									value={newPresetName}
									onChange={(e) => setNewPresetName(e.target.value)}
									placeholder="Preset name..."
									className="w-full text-xs bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-600 rounded px-2 py-1 mb-2 focus:outline-none focus:ring-1 focus:ring-blue-500"
									onKeyDown={(e) => e.key === "Enter" && handleSave()}
								/>
								<div className="flex gap-2 justify-end">
									<button
										type="button"
										onClick={() => setShowSaveInput(false)}
										className="text-[10px] text-slate-500 hover:text-slate-700 px-2 py-1"
									>
										Cancel
									</button>
									<button
										type="button"
										onClick={handleSave}
										disabled={!newPresetName.trim()}
										className="text-[10px] bg-blue-500 hover:bg-blue-600 text-white rounded px-3 py-1 font-medium disabled:opacity-50"
									>
										Save
									</button>
								</div>
							</div>
						)}

						{presets.length === 0 && !showSaveInput ? (
							<div className="text-center py-6 text-slate-400 text-xs italic">
								No saved presets
							</div>
						) : (
							presets.map((preset) => (
								<div
									key={preset.id}
									className="w-full group flex items-center justify-between px-2 py-2 rounded-lg text-xs hover:bg-slate-50 dark:hover:bg-slate-700/50 text-slate-600 dark:text-slate-300 transition-colors text-left"
								>
									<button
										type="button"
										onClick={() => handleLoad(preset.modelIds)}
										className="flex-1 text-left"
									>
										<div className="font-medium">{preset.name}</div>
										<div className="text-[9px] text-slate-400 mt-0.5">
											{preset.modelIds.length} models
										</div>
									</button>

									<button
										type="button"
										onClick={(e) => handleDelete(e, preset.id)}
										className="p-1.5 rounded opacity-0 group-hover:opacity-100 hover:bg-red-50 hover:text-red-500 transition-all text-slate-400"
										title="Delete Preset"
									>
										<Trash2 size={12} />
									</button>
								</div>
							))
						)}
					</div>
				</div>
			)}
		</div>
	);
}
