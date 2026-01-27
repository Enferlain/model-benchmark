import {
	AlertCircle,
	Minus,
	Search,
	Shrink,
	ThumbsDown,
	X,
} from "lucide-react";
import type React from "react";
import { useState } from "react";

interface ArenaBattleProps {
	prompt: string;
	imageA: string;
	imageB: string;
	refImage?: string; // Optional reference image URL
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

	return (
		<div className="flex flex-col h-full w-full max-w-7xl mx-auto gap-6 transition-all duration-500">
			{/* 1. Prompt Area */}
			<div className="bg-white dark:bg-slate-800 p-4 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 text-center relative z-10">
				<h3 className="text-sm uppercase tracking-wider font-bold text-slate-400 mb-1">
					Prompt
				</h3>
				<p className="text-lg text-slate-800 dark:text-slate-100 font-medium leading-relaxed">
					{prompt}
				</p>
			</div>

			{/* 2. Battle Arena (Images) */}
			<div className="flex-1 min-h-[400px] flex items-stretch justify-center gap-4 relative">
				{/* Model A */}
				<div
					className={`flex-1 relative group rounded-xl overflow-hidden bg-black/5 shadow-inner border border-slate-200 dark:border-slate-700 transition-all duration-500 ${isRefExpanded ? "flex-[1]" : "flex-[10]"}`}
				>
					<img
						src={imageA}
						alt="Model A Output"
						className="w-full h-full object-contain cursor-zoom-in hover:opacity-95 transition-opacity"
						onClick={() => setShowLightbox(imageA)}
					/>
					<div className="absolute top-4 left-4 bg-black/50 backdrop-blur text-white px-3 py-1 rounded-full text-xs font-bold pointer-events-none">
						A
					</div>
				</div>

				{/* Reference Image (Expandable Center) */}
				<div
					className={`relative transition-all duration-500 flex flex-col items-center justify-start ${
						isRefExpanded
							? "flex-[1] max-w-[33%] bg-white dark:bg-slate-800 rounded-xl shadow-xl z-20 border border-indigo-500/30"
							: "w-16 flex-none pt-0 -mt-2 z-20"
					}`}
				>
					{/* Toggle Button / Mini View */}
					<button
						onClick={() => setIsRefExpanded(!isRefExpanded)}
						className={`transition-all duration-300 flex flex-col items-center gap-1 group ${
							isRefExpanded
								? "absolute top-2 right-2 p-1.5 hover:bg-slate-100 dark:hover:bg-slate-700 rounded-full text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
								: "bg-white dark:bg-slate-800 p-1.5 rounded-b-xl shadow-md border border-t-0 border-slate-200 dark:border-slate-700 hover:pt-3 hover:pb-2"
						}`}
						title={
							isRefExpanded ? "Collapse Reference" : "Show Reference Image"
						}
					>
						{isRefExpanded ? (
							<Shrink size={18} />
						) : (
							<>
								<div className="w-10 h-10 rounded bg-slate-200 dark:bg-slate-700 overflow-hidden relative border border-slate-300 dark:border-slate-600">
									{refImage ? (
										<img
											src={refImage}
											className="w-full h-full object-cover"
										/>
									) : (
										<div className="w-full h-full flex items-center justify-center text-slate-400">
											<Search size={14} />
										</div>
									)}
								</div>
								<span className="text-[10px] font-bold text-slate-400 group-hover:text-indigo-500 transition-colors uppercase tracking-tight">
									Ref
								</span>
							</>
						)}
					</button>

					{/* Expanded Content */}
					{isRefExpanded && (
						<div className="w-full h-full p-4 pt-12 flex flex-col">
							<div className="w-full flex-1 rounded-lg overflow-hidden bg-slate-100 dark:bg-black/20 relative">
								{refImage ? (
									<img
										src={refImage}
										className="w-full h-full object-contain cursor-zoom-in"
										onClick={() => setShowLightbox(refImage)}
									/>
								) : (
									<div className="w-full h-full flex flex-col items-center justify-center text-slate-400 p-4 text-center">
										<AlertCircle size={32} className="mb-2 opacity-50" />
										<span className="text-sm">
											No reference image available
										</span>
									</div>
								)}
							</div>
							<p className="text-center text-xs text-slate-400 mt-2 font-medium uppercase tracking-wider">
								Ground Truth
							</p>
						</div>
					)}
				</div>

				{/* Model B */}
				<div
					className={`flex-1 relative group rounded-xl overflow-hidden bg-black/5 shadow-inner border border-slate-200 dark:border-slate-700 transition-all duration-500 ${isRefExpanded ? "flex-[1]" : "flex-[10]"}`}
				>
					<img
						src={imageB}
						alt="Model B Output"
						className="w-full h-full object-contain cursor-zoom-in hover:opacity-95 transition-opacity"
						onClick={() => setShowLightbox(imageB)}
					/>
					<div className="absolute top-4 right-4 bg-black/50 backdrop-blur text-white px-3 py-1 rounded-full text-xs font-bold pointer-events-none">
						B
					</div>
				</div>
			</div>

			{/* 3. Voting Controls */}
			<div className="flex justify-center items-center gap-4 py-4">
				<VoteButton
					onClick={() => onVote("A")}
					color="indigo"
					icon={<div className="font-black text-lg">A</div>}
					label="Better"
				/>
				<VoteButton
					onClick={() => onVote("B")}
					color="indigo"
					icon={<div className="font-black text-lg">B</div>}
					label="Better"
				/>
				<div className="w-px h-10 bg-slate-200 dark:bg-slate-700 mx-2" />
				<VoteButton
					onClick={() => onVote("Tie")}
					color="slate"
					icon={<Minus size={24} />}
					label="Tie"
				/>
				<VoteButton
					onClick={() => onVote("BothBad")}
					color="red"
					icon={<ThumbsDown size={20} />}
					label="Both Bad"
				/>
			</div>

			{/* Simple Lightbox */}
			{showLightbox && (
				<div
					className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-8 backdrop-blur-sm animate-in fade-in duration-200"
					onClick={() => setShowLightbox(null)}
				>
					<img
						src={showLightbox}
						className="max-w-full max-h-full rounded-md shadow-2xl"
					/>
					<button className="absolute top-8 right-8 text-white/50 hover:text-white">
						<X size={32} />
					</button>
				</div>
			)}
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
			onClick={onClick}
			className={`
                ${colors[color]}
                flex flex-col items-center justify-center w-24 h-20 rounded-xl border-2 transition-all active:scale-95 shadow-sm
            `}
		>
			<div className="mb-1">{icon}</div>
			<span className="text-xs font-bold uppercase tracking-wide">{label}</span>
		</button>
	);
}
