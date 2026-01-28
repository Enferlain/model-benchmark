import {
	AlertCircle,
	Minus,
	Search,
	Shrink,
	ThumbsDown,
	X,
} from "lucide-react";
import type React from "react";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

interface ArenaBattleProps {
	prompt: string;
	imageA: string;
	imageB: string;
	refImage?: string;
	onVote: (vote: "A" | "B" | "Tie" | "BothBad") => void;
	key?: React.Key;
}

export function ArenaBattle({
	prompt,
	imageA,
	imageB,
	refImage,
	onVote,
}: ArenaBattleProps) {
	const [isRefExpanded, setIsRefExpanded] = useState(false);
	const [showLightbox, setShowLightbox] = useState<string | null>(null);
	const [aspectRatio, setAspectRatio] = useState<number>(0); // Initialize as 0 to wait for first load

	const handleImageLoad = (
		e: React.SyntheticEvent<HTMLImageElement>,
		isPrimary = false,
	) => {
		const img = e.currentTarget;
		if (isPrimary && img.naturalWidth && img.naturalHeight) {
			const ratio = img.naturalWidth / img.naturalHeight;
			// Only update if significantly different to avoid flicker
			if (Math.abs(aspectRatio - ratio) > 0.01) {
				setAspectRatio(ratio);
			}
		}
	};

	const arenaRef = useRef<HTMLDivElement>(null);
	const imageRef = useRef<HTMLDivElement>(null);
	const [containerHeight, setContainerHeight] = useState(0);
	const [measuredImageHeight, setMeasuredImageHeight] = useState<number | null>(
		null,
	);

	// Use ResizeObserver to track the actual rendered height of the images
	useLayoutEffect(() => {
		if (!imageRef.current) return;

		const observer = new ResizeObserver((entries) => {
			for (const entry of entries) {
				if (entry.contentRect.height > 0) {
					setMeasuredImageHeight(entry.contentRect.height);
				}
			}
		});

		observer.observe(imageRef.current);
		return () => observer.disconnect();
	}, []);

	useEffect(() => {
		const updateHeight = () => {
			if (arenaRef.current) {
				// The height of the image area is roughly the arena height minus padding/Vote buttons
				// We'll use the bounding rect of the arena container
				setContainerHeight(arenaRef.current.offsetHeight);
			}
		};

		updateHeight();
		window.addEventListener("resize", updateHeight);
		return () => window.removeEventListener("resize", updateHeight);
	}, []);

	// Calculate precise pixel width of the images
	// images scale to fit container height (minus padding and button area which is ~100px)
	const availableHeight = Math.max(100, containerHeight - 120);
	const imgW = aspectRatio > 0 ? availableHeight * aspectRatio : 400;
	// Since both use the same aspect ratio for now (simplified)
	const imgW_B = imgW;

	// Calculate container styles based on aspect ratio
	const imageContainerStyle = useMemo(() => {
		if (aspectRatio === 0) return {};
		return {
			aspectRatio: `${aspectRatio}`,
		};
	}, [aspectRatio]);

	return (
		<div className="flex flex-col h-full w-full max-w-[1600px] mx-auto gap-4 min-h-0">
			{/* 1. Compact Prompt Area */}
			<div className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm p-2 px-4 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700 text-center relative z-10 flex flex-col items-center gap-0.5 flex-none">
				<h3 className="text-[10px] uppercase tracking-widest font-bold text-slate-400">
					Prompt
				</h3>
				<p className="text-sm lg:text-base text-slate-800 dark:text-slate-100 font-medium leading-tight max-w-4xl">
					{prompt}
				</p>
			</div>

			{/* 2. Unified Battle Arena (Columns for vertical Image + Vote alignment) */}
			<div
				ref={arenaRef}
				className="flex-1 flex items-stretch gap-[17px] relative min-h-0 px-2 overflow-hidden py-2"
			>
				{/* Model A Column */}
				<div className="flex-1 flex justify-end min-w-0 z-10">
					{/* Vertical Stack: Both image and button centered in a unit pulled to the divider */}
					<div
						className="flex flex-col items-center min-h-0 h-full max-h-full"
						style={{ width: `${imgW}px` }}
					>
						{/* Image Holder */}
						<div className="flex-1 flex items-center justify-center min-h-0 w-full">
							<div
								ref={imageRef}
								className="relative rounded-xl overflow-hidden bg-black/5 shadow-inner border border-slate-200 dark:border-slate-700 group h-fit max-h-full"
								style={imageContainerStyle}
							>
								<button
									type="button"
									onClick={() => setShowLightbox(imageA)}
									className="w-full h-full p-0 border-none bg-transparent cursor-zoom-in flex items-center justify-center"
									aria-label="Zoom Model A result"
								>
									<img
										src={imageA}
										onLoad={(e) => handleImageLoad(e, true)}
										alt="Model A Output"
										className="max-w-full max-h-full object-contain hover:opacity-95 transition-opacity"
									/>
								</button>
								<div className="absolute top-4 left-4 bg-black/50 backdrop-blur text-white px-3 py-1 rounded-full text-xs font-bold pointer-events-none z-10 opacity-0 group-hover:opacity-100 transition-opacity">
									A
								</div>
							</div>
						</div>
						{/* Vote A Button Area */}
						<div className="flex-none py-4 px-2 flex justify-center">
							<div className="w-fit transition-transform duration-500 ease-in-out">
								<VoteButton
									onClick={() => onVote("A")}
									color="indigo"
									icon={
										<div className="font-black text-sm text-indigo-500 group-hover:text-white transition-colors">
											A
										</div>
									}
									label="Vote A"
								/>
							</div>
						</div>
					</div>
				</div>

				{/* Reference Image Column */}
				<div
					className="transition-all duration-500 ease-[cubic-bezier(0.4,0,0.2,1)] flex justify-center min-w-0"
					style={{
						flex: isRefExpanded ? "1 1 0%" : "0 0 56px",
						width: isRefExpanded ? "auto" : "56px",
						zIndex: isRefExpanded ? 20 : 10,
					}}
				>
					<div
						className="flex flex-col items-center min-h-0 h-full max-h-full transition-all duration-500 ease-[cubic-bezier(0.4,0,0.2,1)]"
						style={{ width: isRefExpanded ? `${imgW}px` : "100%" }}
					>
						{/* Reference Holder - Unified Container */}
						<div className="flex-1 flex items-center justify-center relative w-full min-h-0">
							<div
								className={`
									relative w-full transition-all duration-500 ease-[cubic-bezier(0.4,0,0.2,1)]
									rounded-xl overflow-hidden group max-h-full border border-dashed
									${
										isRefExpanded
											? "bg-transparent border-transparent cursor-default"
											: "bg-white/50 dark:bg-slate-900/50 border-slate-300 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-700/50 cursor-pointer"
									}
								`}
								style={
									isRefExpanded
										? { ...imageContainerStyle }
										: {
												height: measuredImageHeight
													? `${measuredImageHeight}px`
													: "400px",
												width: "56px",
											}
								}
							>
								{/* Shrunk Content Layer */}
								<button
									type="button"
									onClick={() => !isRefExpanded && setIsRefExpanded(true)}
									onKeyDown={(e) => {
										if (
											!isRefExpanded &&
											(e.key === "Enter" || e.key === " ")
										) {
											setIsRefExpanded(true);
										}
									}}
									disabled={isRefExpanded}
									aria-label="Show Reference Image"
									className={`
										absolute inset-0 flex flex-col items-center justify-center gap-2
										transition-all duration-500 ease-[cubic-bezier(0.4,0,0.2,1)]
										${isRefExpanded ? "opacity-0 scale-95 pointer-events-none" : "opacity-100 scale-100"}
									`}
								>
									<div className="w-10 h-10 rounded bg-slate-200 dark:bg-slate-700 overflow-hidden relative border border-slate-300 dark:border-slate-600 shrink-0">
										{refImage ? (
											<img
												src={refImage}
												alt="Reference"
												className="w-full h-full object-cover opacity-50 group-hover:opacity-100 transition-opacity"
											/>
										) : (
											<div className="w-full h-full flex items-center justify-center text-slate-400">
												<Search size={14} />
											</div>
										)}
									</div>
									<span className="text-[10px] font-black text-slate-400 group-hover:text-indigo-500 transition-colors uppercase vertical-text">
										REF
									</span>
								</button>

								{/* Expanded Content Layer */}
								<div
									className={`
										w-full h-full flex items-center justify-center
										transition-all duration-500 ease-[cubic-bezier(0.4,0,0.2,1)]
										${isRefExpanded ? "opacity-100 scale-100" : "opacity-0 scale-105 pointer-events-none"}
									`}
								>
									{/* Shrink Button Overlay */}
									<button
										type="button"
										onClick={(e) => {
											e.stopPropagation();
											setIsRefExpanded(false);
										}}
										className="absolute top-4 right-4 z-30 w-8 h-8 md:w-10 md:h-10 bg-slate-900/40 hover:bg-slate-900/60 backdrop-blur-md text-white border border-white/20 rounded-full shadow-lg transition-all duration-300 active:scale-90 flex items-center justify-center opacity-0 group-hover:opacity-100 group/btn"
										title="Collapse Reference"
									>
										<Shrink
											size={18}
											className="group-hover/btn:scale-110 transition-transform"
										/>
									</button>

									{refImage ? (
										<button
											type="button"
											onClick={() => setShowLightbox(refImage)}
											className="w-full h-full p-0 border-none bg-transparent cursor-zoom-in flex items-center justify-center"
											aria-label="Zoom reference image"
										>
											<img
												src={refImage}
												onLoad={(e) => handleImageLoad(e, false)}
												alt="Reference preview"
												className="max-w-full max-h-full object-contain"
											/>
										</button>
									) : (
										<div className="w-full h-full flex flex-col items-center justify-center text-slate-400 p-4 text-center">
											<AlertCircle size={32} className="mb-2 opacity-50" />
											<span className="text-sm">No reference</span>
										</div>
									)}
								</div>
							</div>
						</div>
						{/* Tie/Bad Actions (Absolute positioning to prevent pushing images apart) */}
						<div className="flex-none py-4 px-2 flex justify-center w-full">
							<div className="flex items-center gap-2">
								<VoteButton
									onClick={() => onVote("Tie")}
									color="slate"
									icon={<Minus size={18} />}
									label="Tie"
								/>
								<VoteButton
									onClick={() => onVote("BothBad")}
									color="red"
									icon={<ThumbsDown size={18} />}
									label="Bad"
								/>
							</div>
						</div>
					</div>
				</div>

				{/* Model B Column */}
				<div className="flex-1 flex justify-start min-w-0 z-10">
					{/* Vertical Stack: Both image and button centered in a unit pulled to the divider */}
					<div
						className="flex flex-col items-center min-h-0 h-full max-h-full"
						style={{ width: `${imgW_B}px` }}
					>
						{/* Image Holder */}
						<div className="flex-1 flex items-center justify-center min-h-0 w-full">
							<div
								className="relative rounded-xl overflow-hidden bg-black/5 shadow-inner border border-slate-200 dark:border-slate-700 group h-fit max-h-full"
								style={imageContainerStyle}
							>
								<button
									type="button"
									onClick={() => setShowLightbox(imageB)}
									className="w-full h-full p-0 border-none bg-transparent cursor-zoom-in flex items-center justify-center"
									aria-label="Zoom Model B result"
								>
									<img
										src={imageB}
										onLoad={(e) => handleImageLoad(e, false)}
										alt="Model B Output"
										className="max-w-full max-h-full object-contain hover:opacity-95 transition-opacity"
									/>
								</button>
								<div className="absolute top-4 right-4 bg-black/50 backdrop-blur text-white px-3 py-1 rounded-full text-xs font-bold pointer-events-none z-10 opacity-0 group-hover:opacity-100 transition-opacity">
									B
								</div>
							</div>
						</div>
						{/* Vote B Button Area */}
						<div className="flex-none py-4 px-2 flex justify-center">
							<div className="w-fit transition-transform duration-500 ease-in-out">
								<VoteButton
									onClick={() => onVote("B")}
									color="indigo"
									icon={
										<div className="font-black text-sm text-indigo-500 group-hover:text-white transition-colors">
											B
										</div>
									}
									label="Vote B"
								/>
							</div>
						</div>
					</div>
				</div>
			</div>

			{/* Lightbox */}
			{showLightbox && (
				<div className="fixed inset-0 z-50 bg-black/95 flex items-center justify-center p-8 backdrop-blur-md animate-in fade-in duration-300">
					<button
						type="button"
						className="absolute inset-0 w-full h-full cursor-default"
						onClick={() => setShowLightbox(null)}
						aria-label="Close lightbox"
					/>
					<div className="relative max-w-full max-h-full flex items-center justify-center">
						<img
							src={showLightbox}
							alt="Lightbox view"
							className="max-w-[calc(100vw-4rem)] max-h-[calc(100vh-4rem)] rounded shadow-2xl animate-in zoom-in-95 duration-300 object-contain"
						/>
						<button
							type="button"
							onClick={() => setShowLightbox(null)}
							className="absolute -top-12 right-0 text-white/70 hover:text-white transition-colors bg-white/10 p-2 rounded-full backdrop-blur-lg"
						>
							<X size={24} />
						</button>
					</div>
				</div>
			)}

			<style>{`
				.vertical-text {
					writing-mode: vertical-rl;
					text-orientation: mixed;
				}
			`}</style>
		</div>
	);
}

function VoteButton({
	onClick,
	color,
	icon,
	label,
}: {
	onClick: () => void;
	color: "indigo" | "slate" | "red";
	icon: React.ReactNode;
	label: string;
}) {
	const colors = {
		indigo:
			"bg-indigo-100 text-indigo-700 hover:bg-indigo-600 hover:text-white border-indigo-200 dark:bg-indigo-900/30 dark:text-indigo-300 dark:border-indigo-700 dark:hover:bg-indigo-600 dark:hover:text-white",
		slate:
			"bg-slate-100 text-slate-700 hover:bg-slate-600 hover:text-white border-slate-200 dark:bg-slate-800 dark:text-slate-300 dark:border-slate-700 dark:hover:bg-slate-600 dark:hover:text-white",
		red: "bg-rose-50 text-rose-700 hover:bg-rose-600 hover:text-white border-rose-200 dark:bg-rose-900/20 dark:text-rose-300 dark:border-rose-800 dark:hover:bg-rose-600 dark:hover:text-white",
	};

	return (
		<button
			type="button"
			onClick={onClick}
			className={`
                ${colors[color]}
                flex flex-col items-center justify-center w-20 h-14 rounded-lg border-2 transition-all active:scale-95 shadow-sm group
            `}
		>
			<div className="mb-0.5">{icon}</div>
			<span className="text-[10px] font-black uppercase tracking-wider">
				{label}
			</span>
		</button>
	);
}
