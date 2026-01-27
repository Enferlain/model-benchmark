import {
	AlertCircle,
	Minus,
	Search,
	Shrink,
	ThumbsDown,
	X,
} from "lucide-react";
import type React from "react";
import { useState, useMemo, useEffect } from "react";

interface ArenaBattleProps {
	prompt: string;
	imageA: string;
	imageB: string;
	refImage?: string;
	onVote: (vote: "A" | "B" | "Tie" | "BothBad") => void;
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
	const [aspectRatio, setAspectRatio] = useState<number>(1); // Default to square until loaded

	const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
		const img = e.currentTarget;
		if (img.naturalWidth && img.naturalHeight) {
			const ratio = img.naturalWidth / img.naturalHeight;
			// Only update if significantly different to avoid flicker
			if (Math.abs(aspectRatio - ratio) > 0.01) {
				setAspectRatio(ratio);
			}
		}
	};

	// Reset ratio when battle changes to prevent "stuck" aspect ratios
	useEffect(() => {
		setAspectRatio(0);
	}, [imageA, imageB, refImage]);

	// Calculate container styles based on aspect ratio
	const imageContainerStyle = useMemo(() => {
		if (aspectRatio === 0) return {};
		return {
			aspectRatio: `${aspectRatio}`,
		};
	}, [aspectRatio]);

	return (
		<div className="flex flex-col h-full w-full max-w-[1600px] mx-auto gap-4 transition-all duration-500 ease-in-out min-h-0">
			{/* 1. Compact Prompt Area */}
			<div className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm p-2 px-4 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700 text-center relative z-10 flex flex-col items-center gap-0.5 flex-none">
				<h3 className="text-[10px] uppercase tracking-widest font-bold text-slate-400">
					Prompt
				</h3>
				<p className="text-sm lg:text-base text-slate-800 dark:text-slate-100 font-medium leading-tight max-w-4xl">
					{prompt}
				</p>
			</div>

			{/* 2. Battle Arena (Images) - Fair 1:1:1 Distribution with Perfect Hugging */}
			<div className="flex-1 flex items-center justify-center gap-4 relative min-h-0 px-2 overflow-hidden">
				{/* Model A Container */}
				<div className="flex-1 flex h-full items-center justify-end min-w-0">
					<div
						className="relative transition-all duration-500 ease-in-out rounded-xl overflow-hidden bg-black/5 shadow-inner border border-slate-200 dark:border-slate-700 group h-fit max-h-full"
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
								onLoad={handleImageLoad}
								alt="Model A Output"
								className="w-full h-full object-contain hover:opacity-95 transition-opacity"
							/>
						</button>
						<div className="absolute top-4 left-4 bg-black/50 backdrop-blur text-white px-3 py-1 rounded-full text-xs font-bold pointer-events-none z-10 opacity-0 group-hover:opacity-100 transition-opacity">
							A
						</div>
					</div>
				</div>

				{/* Reference Container (Expandable) */}
				<div
					className={`relative transition-all duration-500 ease-in-out flex items-center justify-center h-full min-w-0 ${
						isRefExpanded ? "flex-1 z-20" : "w-14 min-w-[56px] flex-none z-10"
					}`}
				>
					{/* Toggle Button for Shrunk State (Visible when shrunk) */}
					<button
						type="button"
						onClick={() => setIsRefExpanded(true)}
						className={`absolute inset-0 w-full h-full bg-white/50 dark:bg-slate-900/50 rounded-xl border border-dashed border-slate-300 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-700/50 flex flex-col items-center justify-center group transition-all duration-500 ease-in-out ${
							isRefExpanded ? "opacity-0 pointer-events-none scale-95" : "opacity-100 scale-100"
						}`}
						title="Show Reference Image"
					>
						<div className="flex flex-col items-center gap-2">
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
						</div>
					</button>

					{/* Expanded Reference Card (Visible when expanded) */}
					<div
						className={`relative transition-all duration-500 ease-in-out rounded-xl overflow-hidden bg-white dark:bg-slate-800 shadow-xl border border-indigo-500/30 group h-fit max-h-full ${
							isRefExpanded ? "opacity-100 scale-100" : "opacity-0 pointer-events-none scale-95 absolute"
						}`}
						style={imageContainerStyle}
					>
						{/* Shrink Button Overlay */}
						<button
							type="button"
							onClick={(e) => {
								e.stopPropagation();
								setIsRefExpanded(false);
							}}
							className="absolute top-4 right-4 z-30 w-10 h-10 bg-black/20 hover:bg-black/40 backdrop-blur-md text-white border border-white/20 rounded-full shadow-lg transition-all duration-300 active:scale-90 flex items-center justify-center opacity-0 group-hover:opacity-100"
							title="Collapse Reference"
						>
							<Shrink size={18} />
						</button>

						<div className="w-full h-full flex items-center justify-center">
							{refImage ? (
								<button
									type="button"
									onClick={() => setShowLightbox(refImage)}
									className="w-full h-full p-0 border-none bg-transparent cursor-zoom-in flex items-center justify-center"
									aria-label="Zoom reference image"
								>
									<img
										src={refImage}
										onLoad={handleImageLoad}
										alt="Reference preview"
										className="w-full h-full object-contain"
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

				{/* Model B Container */}
				<div className="flex-1 flex h-full items-center justify-start min-w-0">
					<div
						className="relative transition-all duration-500 ease-in-out rounded-xl overflow-hidden bg-black/5 shadow-inner border border-slate-200 dark:border-slate-700 group h-fit max-h-full"
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
								onLoad={handleImageLoad}
								alt="Model B Output"
								className="w-full h-full object-contain hover:opacity-95 transition-opacity"
							/>
						</button>
						<div className="absolute top-4 right-4 bg-black/50 backdrop-blur text-white px-3 py-1 rounded-full text-xs font-bold pointer-events-none z-10 opacity-0 group-hover:opacity-100 transition-opacity">
							B
						</div>
					</div>
				</div>
			</div>

			{/* 3. Compact Voting Controls */}
			<div className="flex justify-center items-center gap-3 py-2 flex-none">
				<VoteButton
					onClick={() => onVote("A")}
					color="indigo"
					icon={<div className="font-black text-sm">A</div>}
					label="Model A"
				/>
				<VoteButton
					onClick={() => onVote("B")}
					color="indigo"
					icon={<div className="font-black text-sm">B</div>}
					label="Model B"
				/>
				<div className="w-px h-8 bg-slate-200 dark:bg-slate-700 mx-1" />
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
                flex flex-col items-center justify-center w-20 h-14 rounded-lg border-2 transition-all active:scale-95 shadow-sm
            `}
		>
			<div className="mb-1">{icon}</div>
			<span className="text-xs font-bold uppercase tracking-wide">{label}</span>
		</button>
	);
}
