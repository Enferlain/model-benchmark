import { Swords, Trophy } from "lucide-react";
import type React from "react";
import { useEffect, useState } from "react";
import { ArenaBattle } from "../components/arena/ArenaBattle";
import { ArenaLeaderboard } from "../components/arena/ArenaLeaderboard";
import { fetchModels } from "../services/api";
import type { ModelData } from "../types";

type ArenaTab = "battle" | "leaderboard";

export default function Arena() {
	const [activeTab, setActiveTab] = useState<ArenaTab>("battle");
	const [_models, setModels] = useState<ModelData[]>([]);

	// Mock State for Battle View
	const [currentRound, setCurrentRound] = useState(0);

	// Dummy data for visualization until we have a real backend endpoint for random pairs
	const dummyPrompts = [
		"A futuristic city with flying cars and neon lights, cyberpunk style, highly detailed",
		"A serene landscape with a mountain lake at sunset, photorealistic",
		"A cute robot holding a flower, 3d render, octane render",
		"Portrait of a warrior princess with intricate armor, fantasy art",
	];

	const [currentPrompt, setCurrentPrompt] = useState(dummyPrompts[0]);
	// Mock image URLs - In a real app these would come from the assets folder based on selected models
	const mockImages = [
		"https://placehold.co/1024x1024/222222/FFF.png?text=Model+A+Result",
		"https://placehold.co/1024x1024/2a2a2a/FFF.png?text=Model+B+Result",
		"https://placehold.co/1024x1024/1a1a1a/FFF.png?text=Next+Pair+A",
		"https://placehold.co/1024x1024/333333/FFF.png?text=Next+Pair+B",
	];

	useEffect(() => {
		fetchModels().then(setModels);
	}, []);

	const handleVote = (vote: string) => {
		console.log(`Voted: ${vote} for prompt: ${currentPrompt}`);
		// Simulate loading next round
		setTimeout(() => {
			setCurrentRound((prev) => prev + 1);
			setCurrentPrompt(dummyPrompts[(currentRound + 1) % dummyPrompts.length]);
		}, 400); // Small delay for effect
	};

	return (
		<div className="max-w-[1800px] mx-auto px-6 py-6 h-[calc(100vh-64px)] flex flex-col">
			{/* Header & Tabs */}
			<div className="flex items-center justify-between mb-6 flex-none">
				<h2 className="text-2xl font-bold text-slate-800 dark:text-slate-100 flex items-center gap-3">
					<div className="p-2 bg-indigo-600 rounded-lg text-white">
						<Swords size={24} />
					</div>
					Model Arena
				</h2>

				<div className="flex bg-slate-100 dark:bg-slate-900 p-1 rounded-lg border border-slate-200 dark:border-slate-700">
					<TabButton
						active={activeTab === "battle"}
						onClick={() => setActiveTab("battle")}
						icon={<Swords size={18} />}
						label="Battle"
					/>
					<TabButton
						active={activeTab === "leaderboard"}
						onClick={() => setActiveTab("leaderboard")}
						icon={<Trophy size={18} />}
						label="Leaderboard"
					/>
				</div>
			</div>

			{/* Content Area */}
			<div className="flex-1 flex flex-col min-h-0">
				{activeTab === "battle" ? (
					<div
						key={currentRound}
						className="flex-1 animate-in fade-in slide-in-from-bottom-2 duration-500"
					>
						<ArenaBattle
							prompt={currentPrompt}
							imageA={mockImages[(currentRound * 2) % mockImages.length]}
							imageB={mockImages[(currentRound * 2 + 1) % mockImages.length]}
							refImage={
								currentRound % 2 === 0
									? undefined
									: "https://placehold.co/200x200/444/FFF.png?text=Ref"
							}
							onVote={handleVote}
						/>
					</div>
				) : (
					<div className="animate-in fade-in zoom-in-95 duration-300">
						<ArenaLeaderboard />
					</div>
				)}
			</div>
		</div>
	);
}

function TabButton({
	active,
	onClick,
	icon,
	label,
}: {
	active: boolean;
	onClick: () => void;
	icon: React.ReactNode;
	label: string;
}) {
	return (
		<button
			onClick={onClick}
			className={`
                flex items-center gap-2 px-4 py-2 rounded-md font-bold text-sm transition-all
                ${
									active
										? "bg-white dark:bg-slate-700 text-indigo-600 dark:text-indigo-400 shadow-sm"
										: "text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200"
								}
            `}
		>
			{icon}
			{label}
		</button>
	);
}
