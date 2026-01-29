import {
	AlertCircle,
	Minus,
	Search,
	Shrink,
	ThumbsDown,
	X,
} from "lucide-react";
import type React from "react";
import { useEffect, useRef, useState } from "react";

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
	const [ratioA, setRatioA] = useState<number>(0);
	const [ratioB, setRatioB] = useState<number>(0);
	const [ratioRef, setRatioRef] = useState<number>(0);

	const handleImageLoad = (
		e: React.SyntheticEvent<HTMLImageElement>,
		type: "A" | "B" | "Ref",
	) => {
		const img = e.currentTarget;
		if (img.naturalWidth && img.naturalHeight) {
			const ratio = img.naturalWidth / img.naturalHeight;
			if (type === "A" && Math.abs(ratioA - ratio) > 0.01) setRatioA(ratio);
			if (type === "B" && Math.abs(ratioB - ratio) > 0.01) setRatioB(ratio);
			if (type === "Ref" && Math.abs(ratioRef - ratio) > 0.01) setRatioRef(ratio);
		}
	};

	const arenaRef = useRef<HTMLDivElement>(null);
	const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });


	useEffect(() => {
		const updateSize = () => {
			if (arenaRef.current) {
				setContainerSize({
					width: arenaRef.current.offsetWidth,
					height: arenaRef.current.offsetHeight,
				});
			}
		};

		updateSize();
		window.addEventListener("resize", updateSize);
		return () => window.removeEventListener("resize", updateSize);
	}, []);

	// Calculate precise pixel width of the images
	// 1. Target widths based on available height (images scale to fit container height)
	const availableHeight = Math.max(100, containerSize.height - 120);
	
	// Fallback to 1.0 (square) if ratio is not yet loaded
	const effectiveRatioA = ratioA || 1.0;
	const effectiveRatioB = ratioB || 1.0;
	const effectiveRatioRef = ratioRef || 1.0;

	const targetW_A = availableHeight * effectiveRatioA;
	const targetW_B = availableHeight * effectiveRatioB;
	// Reference is either the bar (56px) or expanded (ratio-based)
	const targetW_Ref = isRefExpanded ? (availableHeight * effectiveRatioRef) : 56;

	// 2. Width-based scaling (ensure total width fits in the container)
	// Gaps are 17px between columns (2 gaps total). Padding is 40px total (20px each side).
	const horizontalGaps = 34;
	const padding = 40;
	const availableWidth = Math.max(200, containerSize.width - horizontalGaps - padding);
	
	const totalTargetW = targetW_A + targetW_B + targetW_Ref;
	
	// If total width exceeds available width, scale everything down proportionally
	const scale = totalTargetW > availableWidth ? availableWidth / totalTargetW : 1;

	const imgW_A = targetW_A * scale;
	const imgW_B = targetW_B * scale;
	const imgW_Ref = targetW_Ref * scale;

	// The height of the images and the reference bar should match perfectly
	const imgH = availableHeight * scale;

	// Helper to get image container style based on ratio
	const getImageContainerStyle = (ratio: number) => {
		if (ratio === 0) return {};
		return {
			aspectRatio: `${ratio}`,
		};
	};

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
				className="flex-1 flex items-stretch justify-center gap-[17px] relative min-h-0 px-2 overflow-hidden py-2"
			>
				{/* Model A Column */}
				<div
					className="flex-none flex justify-end min-w-0 z-10 transition-all duration-500 ease-[cubic-bezier(0.4,0,0.2,1)]"
					style={{ width: `${imgW_A}px` }}
				>
					{/* Vertical Stack: Both image and button centered in a unit pulled to the divider */}
					<div
						className="flex flex-col items-center min-h-0 h-full max-h-full w-full"
					>
						{/* Image Holder */}
						<div className="flex-1 flex items-center justify-center min-h-0 w-full">
							<div
								className="relative rounded-xl overflow-hidden bg-black/5 shadow-inner border border-slate-200 dark:border-slate-700 group h-fit max-h-full"
								style={getImageContainerStyle(ratioA)}
							>
								<button
									type="button"
									onClick={() => setShowLightbox(imageA)}
									className="w-full h-full p-0 border-none bg-transparent cursor-zoom-in flex items-center justify-center"
									aria-label="Zoom Model A result"
								>
									<img
										src={imageA}
										onLoad={(e) => handleImageLoad(e, "A")}
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
						flex: "none",
						width: `${imgW_Ref}px`,
						zIndex: isRefExpanded ? 20 : 10,
					}}
				>
					<div
						className="flex flex-col items-center min-h-0 h-full max-h-full transition-all duration-500 ease-[cubic-bezier(0.4,0,0.2,1)] w-full"
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
										? getImageContainerStyle(ratioRef)
										: {
												height: `${imgH}px`,
												width: "100%",
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
												onLoad={(e) => handleImageLoad(e, "Ref")}
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
				<div
					className="flex-none flex justify-start min-w-0 z-10 transition-all duration-500 ease-[cubic-bezier(0.4,0,0.2,1)]"
					style={{ width: `${imgW_B}px` }}
				>
					{/* Vertical Stack: Both image and button centered in a unit pulled to the divider */}
					<div
						className="flex flex-col items-center min-h-0 h-full max-h-full w-full"
					>
						{/* Image Holder */}
						<div className="flex-1 flex items-center justify-center min-h-0 w-full">
							<div
								className="relative rounded-xl overflow-hidden bg-black/5 shadow-inner border border-slate-200 dark:border-slate-700 group h-fit max-h-full"
								style={getImageContainerStyle(ratioB)}
							>
								<button
									type="button"
									onClick={() => setShowLightbox(imageB)}
									className="w-full h-full p-0 border-none bg-transparent cursor-zoom-in flex items-center justify-center"
									aria-label="Zoom Model B result"
								>
									<img
										src={imageB}
										onLoad={(e) => handleImageLoad(e, "B")}
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
