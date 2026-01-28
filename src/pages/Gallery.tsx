import type React from "react";
import { useCallback, useEffect, useState } from "react";
import { getImageUrl } from "../components/compare/utils";
import { SkeletonGrid } from "../components/SkeletonGrid";
import { useData } from "../context/DataContext";
import { fetchModelOutputs } from "../services/api";
import type { ModelOutput } from "../types";

export default function Gallery() {
	const {
		models,
		allPrompts,
		selectedModel,
		setSelectedModel,
		selectedPrompt,
		setSelectedPrompt,
		selectedSeed,
		setSelectedSeed,
		outputCache,
		setOutputCache,
		errors,
	} = useData();

	const [outputs, setOutputs] = useState<ModelOutput[]>([]);
	const [loading, setLoading] = useState<boolean>(false);
	const [localError, setLocalError] = useState<string | null>(null);

	// Lightbox state
	const [lightboxOpen, setLightboxOpen] = useState<boolean>(false);
	const [currentImageIndex, setCurrentImageIndex] = useState<number>(0);

	// Derived state for filters
	const uniqueSeeds = Array.from(new Set(outputs.map((o) => o.seed))).sort(
		(a, b) => (a as number) - (b as number),
	);

	// Filtered outputs
	const filteredOutputs = outputs.filter((output) => {
		if (selectedPrompt !== "All" && output.prompt !== selectedPrompt)
			return false;
		if (selectedSeed !== "All" && output.seed.toString() !== selectedSeed)
			return false;
		return true;
	});

	// Flat list for lightbox navigation
	const lightboxImages = filteredOutputs;

	const loadOutputs = useCallback(
		async (modelId: string) => {
			if (outputCache[modelId]) {
				setOutputs(outputCache[modelId]);
				return;
			}

			setLoading(true);
			setOutputs([]);
			setLocalError(null);
			try {
				const data = await fetchModelOutputs(modelId);
				setOutputs(data);
				setOutputCache((prev) => ({ ...prev, [modelId]: data }));
			} catch (err) {
				console.error("Failed to load outputs", err);
				setLocalError("Failed to load outputs");
			} finally {
				setLoading(false);
			}
		},
		[outputCache, setOutputCache],
	);

	// Improved Auto-Selection & Validation
	useEffect(() => {
		if (models.length > 0) {
			const isValid = models.some((m) => m.id === selectedModel);
			if (!selectedModel || !isValid) {
				setSelectedModel(models[0].id);
			}
		}
	}, [models, selectedModel, setSelectedModel]);

	useEffect(() => {
		if (selectedModel) {
			loadOutputs(selectedModel);
		} else {
			setOutputs([]);
		}
	}, [selectedModel, loadOutputs]);

	const openLightbox = (index: number) => {
		setCurrentImageIndex(index);
		setLightboxOpen(true);
	};

	const closeLightbox = useCallback(() => {
		setLightboxOpen(false);
	}, []);

	const nextImage = useCallback(
		(e?: React.MouseEvent) => {
			e?.stopPropagation();
			setCurrentImageIndex((prev) => (prev + 1) % lightboxImages.length);
		},
		[lightboxImages.length],
	);

	const prevImage = useCallback(
		(e?: React.MouseEvent) => {
			e?.stopPropagation();
			setCurrentImageIndex(
				(prev) => (prev - 1 + lightboxImages.length) % lightboxImages.length,
			);
		},
		[lightboxImages.length],
	);

	useEffect(() => {
		const handleKeyDown = (e: KeyboardEvent) => {
			if (!lightboxOpen) return;
			if (e.key === "Escape") closeLightbox();
			if (e.key === "ArrowRight") nextImage();
			if (e.key === "ArrowLeft") prevImage();
		};
		window.addEventListener("keydown", handleKeyDown);
		return () => window.removeEventListener("keydown", handleKeyDown);
	}, [lightboxOpen, closeLightbox, nextImage, prevImage]);

	const groupedOutputs = filteredOutputs.reduce(
		(acc, output) => {
			if (!acc[output.prompt]) {
				acc[output.prompt] = [];
			}
			acc[output.prompt].push(output);
			return acc;
		},
		{} as Record<string, ModelOutput[]>,
	);

	const activeError = errors.models || localError;

	return (
		<div className="max-w-[1800px] mx-auto px-6 py-8">
			<div className="flex flex-col xl:flex-row justify-between items-start xl:items-center mb-8 gap-4">
				<h2 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
					Image Gallery
				</h2>

				<div className="flex flex-wrap items-center gap-4 bg-white dark:bg-slate-800 p-4 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700">
					<div className="flex flex-col gap-1">
						<label className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
							Model
						</label>
						<select
							className="px-3 py-2 bg-slate-50 dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-md text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 min-w-[200px]"
							value={selectedModel}
							onChange={(e) => setSelectedModel(e.target.value)}
						>
							{models.length === 0 && <option value="">No models found</option>}
							{models.map((m) => (
								<option key={m.id} value={m.id}>
									{m.name}
								</option>
							))}
						</select>
					</div>

					<div className="flex flex-col gap-1">
						<label className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
							Prompt
						</label>
						<select
							className="px-3 py-2 bg-slate-50 dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-md text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 max-w-[300px]"
							value={selectedPrompt}
							onChange={(e) => setSelectedPrompt(e.target.value)}
							disabled={allPrompts.length === 0}
						>
							<option value="All">All Prompts ({allPrompts.length})</option>
							{allPrompts.map((p, i) => (
								<option key={p.id || i} value={p.text}>
									{p.text.substring(0, 50)}
									{p.text.length > 50 ? "..." : ""}
								</option>
							))}
						</select>
					</div>

					<div className="flex flex-col gap-1">
						<label className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
							Seed
						</label>
						<select
							className="px-3 py-2 bg-slate-50 dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-md text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500"
							value={selectedSeed}
							onChange={(e) => setSelectedSeed(e.target.value)}
							disabled={outputs.length === 0}
						>
							<option value="All">All Seeds</option>
							{uniqueSeeds.map((s) => (
								<option key={s} value={s.toString()}>
									{s}
								</option>
							))}
						</select>
					</div>

					<div className="ml-auto text-sm text-slate-500 dark:text-slate-400">
						Showing {filteredOutputs.length} images
					</div>
				</div>
			</div>

			{activeError && (
				<div className="p-4 mb-6 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 rounded-md border border-red-200 dark:border-red-800 font-medium">
					⚠️ {activeError}
				</div>
			)}

			{loading && (
				<div className="py-8 animate-fadeIn">
					<SkeletonGrid />
				</div>
			)}

			{!loading &&
				filteredOutputs.length === 0 &&
				selectedModel &&
				!activeError && (
					<div className="text-center py-12 text-slate-500 dark:text-slate-400 bg-slate-50 dark:bg-slate-800/50 rounded-lg border border-dashed border-slate-300 dark:border-slate-700">
						<p className="text-lg">No images found matching current filters.</p>
						{(selectedPrompt !== "All" || selectedSeed !== "All") && (
							<button
								type="button"
								onClick={() => {
									setSelectedPrompt("All");
									setSelectedSeed("All");
								}}
								className="mt-4 px-4 py-2 text-indigo-600 dark:text-indigo-400 hover:underline"
							>
								Clear Filters
							</button>
						)}
					</div>
				)}

			{!loading &&
				Object.keys(groupedOutputs).map((prompt, idx) => (
					<div key={idx} className="mb-8 animate-fadeIn">
						<div className="bg-slate-100 dark:bg-slate-800/50 p-4 rounded-lg mb-4 border-l-4 border-indigo-500">
							<h3 className="text-lg font-semibold text-slate-800 dark:text-slate-200 mb-1">
								Prompt {outputs.find((o) => o.prompt === prompt)?.prompt_idx}:
							</h3>
							<p className="text-slate-600 dark:text-slate-400 font-mono text-sm">
								{prompt}
							</p>
						</div>

						<div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6 gap-6">
							{groupedOutputs[prompt].map((output) => {
								const globalIndex = lightboxImages.findIndex(
									(o) => o.filename === output.filename,
								);

								return (
									<div
										key={output.filename}
										className="bg-white dark:bg-slate-800 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700 overflow-hidden hover:shadow-lg transition-all cursor-pointer group"
										onClick={() => openLightbox(globalIndex)}
									>
										<div className="aspect-[2/3] relative overflow-hidden bg-slate-100 dark:bg-slate-900">
											<img
												src={getImageUrl(output.url, output.mtime)}
												alt={output.prompt}
												className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
												loading="lazy"
											/>
											<div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
												<span className="text-white bg-black/50 px-3 py-1 rounded-full text-sm backdrop-blur-sm">
													Click to Enlarge
												</span>
											</div>
										</div>
										<div className="p-3 border-t border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50">
											<div className="flex justify-between items-center text-xs text-slate-500 dark:text-slate-400">
												<span>
													Seed:{" "}
													<span className="font-mono text-slate-700 dark:text-slate-300">
														{output.seed}
													</span>
												</span>
												<span
													className="truncate max-w-[100px] opacity-70"
													title={output.filename}
												>
													{output.filename}
												</span>
											</div>
										</div>
									</div>
								);
							})}
						</div>
					</div>
				))}

			{lightboxOpen && lightboxImages.length > 0 && (
				<div
					className="fixed inset-0 z-50 bg-black/95 flex items-center justify-center backdrop-blur-sm"
					onClick={closeLightbox}
				>
					<button
						type="button"
						className="absolute top-4 right-4 text-white/70 hover:text-white p-2 z-50"
						onClick={closeLightbox}
					>
						<svg
							xmlns="http://www.w3.org/2000/svg"
							width="32"
							height="32"
							viewBox="0 0 24 24"
							fill="none"
							stroke="currentColor"
							strokeWidth="2"
							strokeLinecap="round"
							strokeLinejoin="round"
						>
							<line x1="18" y1="6" x2="6" y2="18"></line>
							<line x1="6" y1="6" x2="18" y2="18"></line>
						</svg>
					</button>

					<button
						type="button"
						className="absolute left-4 top-1/2 -translate-y-1/2 text-white/70 hover:text-white p-4 z-50 hover:bg-white/10 rounded-full transition-colors"
						onClick={prevImage}
					>
						<svg
							xmlns="http://www.w3.org/2000/svg"
							width="40"
							height="40"
							viewBox="0 0 24 24"
							fill="none"
							stroke="currentColor"
							strokeWidth="2"
							strokeLinecap="round"
							strokeLinejoin="round"
						>
							<polyline points="15 18 9 12 15 6"></polyline>
						</svg>
					</button>

					<button
						type="button"
						className="absolute right-4 top-1/2 -translate-y-1/2 text-white/70 hover:text-white p-4 z-50 hover:bg-white/10 rounded-full transition-colors"
						onClick={nextImage}
					>
						<svg
							xmlns="http://www.w3.org/2000/svg"
							width="40"
							height="40"
							viewBox="0 0 24 24"
							fill="none"
							stroke="currentColor"
							strokeWidth="2"
							strokeLinecap="round"
							strokeLinejoin="round"
						>
							<polyline points="9 18 15 12 9 6"></polyline>
						</svg>
					</button>

					<div
						className="relative max-w-[90vw] max-h-[90vh] flex flex-col items-center"
						onClick={(e) => e.stopPropagation()}
					>
						<img
							src={getImageUrl(
								lightboxImages[currentImageIndex].url,
								lightboxImages[currentImageIndex].mtime,
							)}
							alt={lightboxImages[currentImageIndex].prompt}
							className="max-w-full max-h-[85vh] object-contain shadow-2xl rounded-sm"
						/>
						<div className="mt-4 text-white text-center w-full max-w-4xl">
							<p className="text-sm text-gray-400 mb-1 uppercase tracking-widest text-xs">
								{currentImageIndex + 1} / {lightboxImages.length} • Seed{" "}
								{lightboxImages[currentImageIndex].seed}
							</p>
							<p
								className="font-medium text-lg leading-snug line-clamp-2"
								title={lightboxImages[currentImageIndex].prompt}
							>
								{lightboxImages[currentImageIndex].prompt}
							</p>
						</div>
					</div>
				</div>
			)}
		</div>
	);
}
