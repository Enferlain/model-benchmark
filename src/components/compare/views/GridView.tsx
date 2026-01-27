import type React from "react";
import { useState } from "react";
import type { ModelOutput } from "../../../types";
import { getImageUrl } from "../utils";

interface Props {
	prompts: string[];
	modelNames: string[];
	getImagesForSelection: (
		prompt: string,
		seed: string,
	) => (ModelOutput | undefined)[];
	seed: string;
}

export const GridView: React.FC<Props> = ({
	prompts,
	modelNames,
	getImagesForSelection,
	seed,
}) => {
	const [expandedImage, setExpandedImage] = useState<ModelOutput | null>(null);
	const [zoomLevel, setZoomLevel] = useState<number>(150); // Default width in px
	const [fitToScreen, setFitToScreen] = useState<boolean>(true);

	return (
		<div className="flex flex-col h-full w-full overflow-hidden bg-zinc-50 dark:bg-zinc-950">
			<div className="flex-1 overflow-auto">
				<table
					className={`w-full border-collapse ${fitToScreen ? "table-fixed" : "table-fixed"}`}
				>
					<thead className="sticky top-0 z-20 bg-white dark:bg-zinc-900 shadow-sm">
						<tr>
							<th
								className="p-4 text-left bg-white dark:bg-zinc-900 border-b border-r border-zinc-200 dark:border-zinc-800 z-30 align-top"
								style={{ width: "200px", minWidth: "200px" }}
							>
								<div className="flex flex-col gap-2">
									<div className="flex items-center justify-between">
										<span className="text-xs font-bold uppercase tracking-wider text-zinc-500">
											Prompt
										</span>
										<button
											onClick={() => setFitToScreen(!fitToScreen)}
											className={`text-[9px] px-1.5 py-0.5 rounded border transition-colors ${
												fitToScreen
													? "bg-indigo-100 text-indigo-700 border-indigo-200"
													: "bg-zinc-100 text-zinc-500 border-zinc-200 hover:bg-zinc-200"
											}`}
											title="Fit columns to screen"
										>
											{fitToScreen ? "Fit: ON" : "Fit: OFF"}
										</button>
									</div>

									{/* Zoom Control (Only visible when NOT fitting) */}
									{!fitToScreen && (
										<div className="flex items-center gap-2 mt-1">
											<span className="text-[10px] text-zinc-400">Size</span>
											<input
												type="range"
												min="80"
												max="400"
												step="10"
												value={zoomLevel}
												onChange={(e) => setZoomLevel(Number(e.target.value))}
												className="w-20 h-1 bg-zinc-200 rounded-lg appearance-none cursor-pointer dark:bg-zinc-700"
											/>
										</div>
									)}
								</div>
							</th>
							{modelNames.map((name, i) => (
								<th
									key={i}
									className="p-4 text-center bg-white dark:bg-zinc-900 border-b border-zinc-200 dark:border-zinc-800"
									style={
										fitToScreen
											? { width: "auto" }
											: { width: zoomLevel, minWidth: zoomLevel }
									}
								>
									<span
										className="text-sm font-semibold text-zinc-800 dark:text-zinc-200 block truncate"
										title={name}
									>
										{name}
									</span>
								</th>
							))}
						</tr>
					</thead>
					<tbody className="divide-y divide-zinc-200 dark:divide-zinc-800">
						{prompts.map((prompt, pIdx) => {
							const rowImages = getImagesForSelection(prompt, seed);
							return (
								<tr
									key={pIdx}
									className="group hover:bg-zinc-100 dark:hover:bg-zinc-900/50 transition-colors"
								>
									{/* Sticky Prompt Cell 
                      height: 1px on td is a trick to let the row height be determined by siblings (images),
                      allowing the child h-full to fill that available space and scroll.
                  */}
									<td
										className="p-4 align-top sticky left-0 bg-zinc-50 dark:bg-zinc-950 group-hover:bg-zinc-100 dark:group-hover:bg-zinc-900/50 border-r border-zinc-200 dark:border-zinc-800 z-10"
										style={{ width: "200px", minWidth: "200px", height: "1px" }}
									>
										<div className="h-full overflow-y-auto max-h-full pr-1">
											<p
												className="text-xs text-zinc-600 dark:text-zinc-400 font-mono"
												title={prompt}
											>
												{prompt}
											</p>
										</div>
									</td>

									{/* Image Cells */}
									{rowImages.map((img, mIdx) => (
										<td
											key={mIdx}
											className="p-2 align-top"
											// Width handled by header in table-fixed, but good to ensure consistency
										>
											<div
												className="w-full relative rounded overflow-hidden bg-zinc-200 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700"
												// Maintain aspect ratio, or allow height to grow?
												// If fitting, height will shrink.
												style={{ aspectRatio: "2/3" }}
											>
												{img ? (
													<div
														className="w-full h-full cursor-pointer hover:opacity-90 transition-opacity"
														onClick={() => setExpandedImage(img)}
													>
														<img
															src={getImageUrl(img.url, img.mtime)}
															alt={`${modelNames[mIdx]}`}
															className="w-full h-full object-cover"
															loading="lazy"
														/>
														{/* Floating Model Name Badge */}
														<div className="absolute top-2 left-1/2 -translate-x-1/2 px-2 py-1 bg-black/60 backdrop-blur rounded-full opacity-0 group-hover:opacity-100 transition-opacity z-10 pointer-events-none whitespace-nowrap max-w-[90%] overflow-hidden text-ellipsis">
															<span className="text-[10px] text-white font-medium">
																{modelNames[mIdx]}
															</span>
														</div>

														{/* Seed Badge */}
														<div className="absolute bottom-1 right-1 px-1.5 py-0.5 bg-black/60 backdrop-blur text-[9px] text-white rounded font-mono opacity-0 group-hover:opacity-100 transition-opacity">
															{img.seed}
														</div>
													</div>
												) : (
													<div className="flex items-center justify-center h-full text-zinc-400 text-xs text-center p-2">
														No Data
													</div>
												)}
											</div>
										</td>
									))}
								</tr>
							);
						})}
					</tbody>
				</table>
			</div>

			{/* Lightbox for Grid */}
			{expandedImage && (
				<div
					className="fixed inset-0 z-[200] bg-black/90 backdrop-blur-sm flex items-center justify-center p-8"
					onClick={() => setExpandedImage(null)}
				>
					<div className="relative max-w-5xl max-h-full flex flex-col items-center justify-center p-4">
						<img
							src={getImageUrl(expandedImage.url, expandedImage.mtime)}
							alt="Expanded"
							className="max-h-[80vh] w-auto object-contain rounded-lg shadow-2xl flex-1 min-h-0"
							onClick={(e) => e.stopPropagation()}
						/>
						<div
							className="mt-4 text-center flex-shrink-0 bg-black/60 backdrop-blur-md rounded-xl p-3 text-white max-w-full overflow-y-auto max-h-[15vh]"
							onClick={(e) => e.stopPropagation()}
						>
							<p className="font-mono text-sm opacity-90">
								{expandedImage.prompt}
							</p>
							<p className="text-white/60 text-xs mt-1">
								Seed: {expandedImage.seed}
							</p>
						</div>
					</div>
				</div>
			)}
		</div>
	);
};
