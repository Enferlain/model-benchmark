import { useEffect, useState } from "react";
import { useComparisonData } from "../components/compare/useComparisonData";
import { GridView } from "../components/compare/views/GridView";
import { ProximityView } from "../components/compare/views/ProximityView";
import { SideBySideView } from "../components/compare/views/SideBySideView";
import { SliderView } from "../components/compare/views/SliderView";
import { fetchModels, fetchNote, saveNote } from "../services/api";
import type { ModelData } from "../types";

type ViewMode = "side-by-side" | "slider" | "proximity" | "grid";

export default function Compare() {
	const [models, setModels] = useState<ModelData[]>([]);

	// Initialize state from LocalStorage if available
	const [viewMode, setViewMode] = useState<ViewMode>(() => {
		return (
			(localStorage.getItem("compare_viewMode") as ViewMode) || "side-by-side"
		);
	});

	const [selectedPrompt, setSelectedPrompt] = useState<string>(() => {
		return localStorage.getItem("compare_selectedPrompt") || "All";
	});

	const [selectedSeed, setSelectedSeed] = useState<string>(() => {
		return localStorage.getItem("compare_selectedSeed") || "All";
	});

	// Support N models
	const [selectedModelIds, setSelectedModelIds] = useState<string[]>([]);

	const [note, setNote] = useState<string>("");
	const [noteSaving, setNoteSaving] = useState(false);

	// Load models on mount
	useEffect(() => {
		fetchModels().then((data) => {
			setModels(data);

			// Try to load selection from storage
			const storedIds = localStorage.getItem("compare_selectedModelIds");
			let applied = false;

			if (storedIds) {
				try {
					const parsed = JSON.parse(storedIds);
					if (Array.isArray(parsed)) {
						// Validate IDs exist
						const validIds = parsed.filter((id) =>
							data.find((m) => m.id === id),
						);
						if (validIds.length > 0) {
							setSelectedModelIds(validIds);
							applied = true;
						}
					}
				} catch (e) {
					console.error("Failed to parse stored model IDs", e);
				}
			}

			// Default if no storage or invalid
			if (!applied) {
				if (data.length >= 2) {
					setSelectedModelIds([data[0].id, data[1].id]);
				} else if (data.length === 1) {
					setSelectedModelIds([data[0].id]);
				}
			}
		});
	}, []);

	// Persistence Effects
	useEffect(() => {
		localStorage.setItem("compare_viewMode", viewMode);
	}, [viewMode]);

	useEffect(() => {
		if (selectedModelIds.length > 0) {
			localStorage.setItem(
				"compare_selectedModelIds",
				JSON.stringify(selectedModelIds),
			);
		}
	}, [selectedModelIds]);

	useEffect(() => {
		localStorage.setItem("compare_selectedPrompt", selectedPrompt);
	}, [selectedPrompt]);

	useEffect(() => {
		localStorage.setItem("compare_selectedSeed", selectedSeed);
	}, [selectedSeed]);

	// Use Custom Hook for Data
	const {
		commonPrompts,
		commonSeeds,
		getImagesForSelection,
		getAllImagesForPrompt,
		loadingMap,
	} = useComparisonData(models, selectedModelIds);

	// Auto-select first common prompt/seed if current selection is invalid
	useEffect(() => {
		// If we have models selected and common data exists
		if (selectedModelIds.length > 0) {
			// We delay this check slightly or ensure it runs after data is loaded to avoid
			// overwriting the restored state with specific logic too aggressively?
			// Actually, the hook updates commonPrompts.
			// If the RESTORED prompt is valid, it stays. If not, this replaces it. Perfect.

			const promptValid =
				selectedPrompt !== "All" && commonPrompts.includes(selectedPrompt);
			const seedValid =
				selectedSeed !== "All" &&
				commonSeeds.map(String).includes(selectedSeed);

			// Only overwrite if INVALID (and we have options)
			// AND specifically if we are not 'All' or if we want to force selection?
			// The original logic was: "If invalid... set to first".

			if (!promptValid && commonPrompts.length > 0) {
				setSelectedPrompt(commonPrompts[0]);
			}

			if (!seedValid && commonSeeds.length > 0) {
				// For Grid View, we generally want a specific seed.
				// If we are in 'proximity' mode, the UI forces 'All', but state might remain.
				// We'll just reset it to the first valid seed to be safe.
				setSelectedSeed(commonSeeds[0].toString());
			}
		}
	}, [
		commonPrompts,
		commonSeeds,
		selectedModelIds.length,
		selectedPrompt,
		selectedSeed,
	]);

	// Get current images for the view
	const currentImages = getImagesForSelection(selectedPrompt, selectedSeed);
	const currentModelNames = selectedModelIds.map(
		(id) => models.find((m) => m.id === id)?.name || id,
	);

	// Note management (Composite key)
	// Key needs to be sorted to be consistent regardless of order?
	// Let's sort IDs in the key.
	const noteId =
		selectedModelIds.length > 0 &&
		selectedPrompt !== "All" &&
		selectedSeed !== "All"
			? `compare:${[...selectedModelIds].sort().join(":")}:${currentImages[0]?.prompt_idx || "0"}:${selectedSeed}`
			: null;

	useEffect(() => {
		if (noteId) {
			setNote("");
			fetchNote(noteId)
				.then((data) => {
					if (data?.content) {
						setNote(data.content);
					}
				})
				.catch((err) => {
					console.debug(`Failed to fetch note for ${noteId}:`, err);
				});
		} else {
			setNote("");
		}
	}, [noteId]);

	const handleSaveNote = async () => {
		if (!noteId) return;
		setNoteSaving(true);
		try {
			await saveNote(noteId, {
				content: note,
				timestamp: new Date().toISOString(),
			});
		} catch (e) {
			console.error("Failed to save note", e);
		} finally {
			setNoteSaving(false);
		}
	};

	const toggleModelSelection = (id: string) => {
		setSelectedModelIds((prev) => {
			if (prev.includes(id)) {
				// Don't allow deselecting the last one? Or allow empty?
				return prev.filter((m) => m !== id);
			} else {
				return [...prev, id];
			}
		});
	};

	return (
		<div className="max-w-[1800px] mx-auto px-4 py-4">
			<div className="flex flex-col gap-4">
				<h2 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
					Model Comparison
				</h2>

				{/* Top Controls Area */}
				<div className="bg-white dark:bg-slate-800 p-4 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700 space-y-4">
					{/* 1. Model Selection (Chips) */}
					<div className="flex flex-col gap-2">
						<label className="text-sm font-semibold text-slate-500 uppercase tracking-wider">
							Selected Models ({selectedModelIds.length})
						</label>
						<div className="flex flex-wrap gap-2">
							{models.map((m) => {
								const isSelected = selectedModelIds.includes(m.id);
								return (
									<button
										key={m.id}
										onClick={() => toggleModelSelection(m.id)}
										className={`px-3 py-1.5 rounded-full text-sm font-medium transition-all border ${
											isSelected
												? "bg-indigo-100 text-indigo-700 border-indigo-200 dark:bg-indigo-900/50 dark:text-indigo-300 dark:border-indigo-700"
												: "bg-slate-50 text-slate-600 border-slate-200 hover:bg-slate-100 dark:bg-slate-700/50 dark:text-slate-400 dark:border-slate-600"
										}`}
									>
										{m.name}
										{isSelected && <span className="ml-2">✓</span>}
									</button>
								);
							})}
						</div>
					</div>

					{/* 2. Parameters & View Mode */}
					<div className="flex flex-wrap gap-4 items-end border-t border-slate-100 dark:border-slate-700 pt-4">
						{/* Prompt Selector */}
						<div className="flex flex-col gap-2 flex-[2] min-w-[300px]">
							<label className="text-sm font-semibold text-slate-600 dark:text-slate-400">
								Common Prompt
							</label>
							<select
								className="w-full px-3 py-2 bg-slate-50 dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-md truncate disabled:opacity-50"
								value={selectedPrompt}
								onChange={(e) => setSelectedPrompt(e.target.value)}
								disabled={commonPrompts.length === 0 || viewMode === "grid"}
							>
								{viewMode === "grid" ? (
									<option>All Prompts (Grid View)</option>
								) : (
									<>
										{commonPrompts.length === 0 && (
											<option value="All">No common prompts found</option>
										)}
										{commonPrompts.map((p, i) => (
											<option key={i} value={p}>
												{p.substring(0, 80)}
												{p.length > 80 ? "..." : ""}
											</option>
										))}
									</>
								)}
							</select>
						</div>

						{/* Seed Selector */}
						<div className="flex flex-col gap-2 flex-none w-[120px]">
							<label className="text-sm font-semibold text-slate-600 dark:text-slate-400">
								Common Seed
							</label>
							<select
								className="w-full px-3 py-2 bg-slate-50 dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-md disabled:opacity-50 disabled:cursor-not-allowed"
								value={viewMode === "proximity" ? "All" : selectedSeed}
								onChange={(e) => setSelectedSeed(e.target.value)}
								disabled={commonSeeds.length === 0 || viewMode === "proximity"}
							>
								{viewMode === "proximity" ? (
									<option value="All">All Seeds (Proximity View)</option>
								) : (
									<>
										{commonSeeds.length === 0 && (
											<option value="All">None</option>
										)}
										{/* Grid view usually wants a specific seed, so 'All' might be confusing, but we handle it in render by picking first */}
										{viewMode === "grid" && selectedSeed === "All" && (
											<option value="All">Pick a seed</option>
										)}
										{commonSeeds.map((s) => (
											<option key={s} value={s}>
												{s}
											</option>
										))}
									</>
								)}
							</select>
						</div>

						{/* View Mode Switcher */}
						<div className="flex bg-slate-100 dark:bg-slate-900 rounded-lg p-1 border border-slate-200 dark:border-slate-700 ml-auto">
							{(["side-by-side", "slider", "grid", "proximity"] as const).map(
								(mode) => (
									<button
										key={mode}
										onClick={() => setViewMode(mode)}
										disabled={
											mode === "slider" && selectedModelIds.length !== 2
										}
										className={`px-4 py-2 rounded-md text-sm font-medium transition-all capitalize ${
											viewMode === mode
												? "bg-white dark:bg-slate-700 shadow-sm text-indigo-600 dark:text-indigo-400"
												: "text-slate-500 hover:text-slate-700 dark:text-slate-400 disabled:opacity-40 disabled:cursor-not-allowed"
										}`}
										title={
											mode === "slider" && selectedModelIds.length !== 2
												? "Slider requires exactly 2 models"
												: ""
										}
									>
										{mode.replace(/-/g, " ")}
									</button>
								),
							)}
						</div>
					</div>
				</div>

				{/* Visualization Area */}
				<div className="bg-slate-100 dark:bg-slate-900/50 rounded-xl border border-slate-200 dark:border-slate-700 min-h-[500px] flex flex-col relative overflow-hidden">
					{selectedModelIds.length === 0 ? (
						<div className="flex-1 flex items-center justify-center text-slate-500">
							Select at least one model to compare.
						</div>
					) : commonPrompts.length === 0 ? (
						<div className="flex-1 flex items-center justify-center text-slate-500">
							Selected models have no common prompts. Try selecting different
							models.
						</div>
					) : (
						<div className="flex-1 flex flex-col min-h-0">
							{viewMode === "side-by-side" && (
								<SideBySideView
									images={currentImages}
									modelNames={currentModelNames}
								/>
							)}
							{viewMode === "slider" && (
								<SliderView
									images={currentImages}
									modelNames={currentModelNames}
								/>
							)}
							{viewMode === "grid" && (
								<GridView
									prompts={commonPrompts}
									modelNames={currentModelNames}
									getImagesForSelection={getImagesForSelection}
									seed={
										selectedSeed === "All"
											? commonSeeds[0]?.toString() || "0"
											: selectedSeed
									}
								/>
							)}
							{viewMode === "proximity" && (
								<ProximityView
									groups={getAllImagesForPrompt(selectedPrompt).map((g) => ({
										id: models.find((m) => m.id === g.modelId)?.id || g.modelId,
										modelName:
											models.find((m) => m.id === g.modelId)?.name || g.modelId,
										images: g.images,
									}))}
								/>
							)}
						</div>
					)}

					{/* Prompt Text Footer */}
					{selectedPrompt !== "All" && viewMode !== "grid" && (
						<div className="p-4 bg-white/50 dark:bg-black/50 backdrop-blur-sm border-t border-slate-200 dark:border-slate-700 text-center">
							<p className="font-mono text-sm text-slate-700 dark:text-slate-300">
								{selectedPrompt}
							</p>
						</div>
					)}
				</div>

				{/* Notes Section */}
				{noteId && (
					<div className="bg-white dark:bg-slate-800 p-6 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700">
						<div className="flex justify-between items-center mb-4">
							<h3 className="text-lg font-semibold text-slate-800 dark:text-slate-100">
								Notes for this Comparison
							</h3>
							<span className="text-xs text-slate-400 font-mono bg-slate-100 dark:bg-slate-900 px-2 py-1 rounded">
								{currentModelNames.length} Models • Seed {selectedSeed}
							</span>
						</div>

						<div className="flex flex-col gap-3">
							<textarea
								className="w-full h-32 p-3 bg-slate-50 dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded-md focus:ring-2 focus:ring-indigo-500 focus:outline-none transition-all"
								placeholder="Add your thoughts on this specific comparison..."
								value={note}
								onChange={(e) => setNote(e.target.value)}
							/>
							<div className="flex justify-end">
								<button
									onClick={handleSaveNote}
									disabled={noteSaving}
									className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-md transition-colors disabled:opacity-50 flex items-center gap-2"
								>
									{noteSaving ? "Saving..." : "Save Note"}
								</button>
							</div>
						</div>
					</div>
				)}
			</div>
		</div>
	);
}
