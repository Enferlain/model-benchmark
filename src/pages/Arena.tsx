import { Swords } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { ArenaBattle } from "../components/arena/ArenaBattle";
import { ArenaLeaderboard } from "../components/arena/ArenaLeaderboard";
import { getImageUrl } from "../components/compare/utils";
import { useData } from "../context/DataContext";
import { fetchModelOutputs } from "../services/api";
import type { ModelOutput } from "../types";

type ArenaTab = "battle" | "leaderboard";

interface BattleState {
	prompt: string;
	imageA: string;
	imageB: string;
	refImage?: string;
	modelAId: string;
	modelBId: string;
	isLoading: boolean;
	error: string | null;
	demoIndex?: number;
}

const MOCK_BATTLE_SETS: BattleState[] = [
	{
		prompt:
			"Square Aspect Ratio: A serene mountain lake at sunrise, reflection in the water.",
		imageA:
			"https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&q=80&w=1024&h=1024",
		imageB:
			"https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&q=80&w=1024&h=1024",
		refImage:
			"https://images.unsplash.com/photo-1439853949127-fa647821eba0?auto=format&fit=crop&q=80&w=1024&h=1024",
		modelAId: "mock-square-a",
		modelBId: "mock-square-b",
		isLoading: false,
		error: null,
	},
	{
		prompt:
			"Landscape Aspect Ratio: A cinematic shot of a futuristic cyberpunk city with neon lights, 3:2 style.",
		imageA:
			"https://images.unsplash.com/photo-1605142859862-978be7eba909?auto=format&fit=crop&q=80&w=1500&h=1000",
		imageB:
			"https://images.unsplash.com/photo-1614728263952-84ea256f9679?auto=format&fit=crop&q=80&w=1500&h=1000",
		refImage:
			"https://images.unsplash.com/photo-1534067783941-51c9c23ecefd?auto=format&fit=crop&q=80&w=1500&h=1000",
		modelAId: "mock-landscape-a",
		modelBId: "mock-landscape-b",
		isLoading: false,
		error: null,
	},
	{
		prompt:
			"Portrait Aspect Ratio: A highly detailed fashion portrait of a model with dramatic lighting, 2:3 style.",
		imageA:
			"https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&q=80&w=1000&h=1500",
		imageB:
			"https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?auto=format&fit=crop&q=80&w=1000&h=1500",
		refImage:
			"https://images.unsplash.com/photo-1531746020798-e6953c6e8e04?auto=format&fit=crop&q=80&w=1000&h=1500",
		modelAId: "mock-portrait-a",
		modelBId: "mock-portrait-b",
		isLoading: false,
		error: null,
	},
	{
		prompt:
			"phoebe (wuthering waves), 1girl, arm support, black bow, purple eyes, shirt lift, sitting, underboob, very long hair, white shirt, white skirt",
		imageA:
			"https://images.unsplash.com/photo-1544005313-94ddf0286df2?q=80&w=1024&h=1536&auto=format&fit=crop",
		imageB:
			"https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?q=80&w=1024&h=1536&auto=format&fit=crop",
		refImage: "/assets/image_prompts/00dd01d80e7eb5a103584889cef7f776.jpg",
		modelAId: "real-ref-1",
		modelBId: "real-ref-1-alt",
		isLoading: false,
		error: null,
	},
	{
		prompt:
			"original, honnryou hanaru, 1girl, black hair, blue eyes, holding umbrella, long hair, street, white dress, white umbrella",
		imageA:
			"https://images.unsplash.com/photo-1518005020251-58c9cbb07657?q=80&w=1536&h=1024&auto=format&fit=crop",
		imageB:
			"https://images.unsplash.com/photo-1472214103451-9374bd1c798e?q=80&w=1536&h=1024&auto=format&fit=crop",
		refImage: "/assets/image_prompts/00e1be4187dd98b2241942998c83d3be.jpg",
		modelAId: "real-ref-2",
		modelBId: "real-ref-2-alt",
		isLoading: false,
		error: null,
	},
	{
		prompt:
			"original, sheya tin, 1boy, baseball cap, beach, denim shorts, male focus, ocean, solo, t-shirt, toy car, volleyball net, white shirt",
		imageA:
			"https://images.unsplash.com/photo-1500382017468-9049fed747ef?q=80&w=1536&h=1024&auto=format&fit=crop",
		imageB:
			"https://images.unsplash.com/photo-1441974231531-c6227db76b6e?q=80&w=1536&h=1024&auto=format&fit=crop",
		refImage: "/assets/image_prompts/0e0917d1e4220068f30382a257f5da72.jpg",
		modelAId: "real-ref-3",
		modelBId: "real-ref-3-alt",
		isLoading: false,
		error: null,
	},
	{
		prompt:
			"seia (blue archive), 1girl, animal ears, fox ears, halo, highleg one-piece swimsuit, loli, night sky, pool, sitting, white swimsuit",
		imageA:
			"https://images.unsplash.com/photo-1518770660439-4636190af475?q=80&w=1024&h=1024&auto=format&fit=crop",
		imageB:
			"https://images.unsplash.com/photo-1512446816042-444d641267d4?q=80&w=1024&h=1024&auto=format&fit=crop",
		refImage: "/assets/image_prompts/03f73abeb4f35df3db91aae08fa05160.jpg",
		modelAId: "real-ref-4",
		modelBId: "real-ref-4-alt",
		isLoading: false,
		error: null,
	},
	{
		prompt:
			"hatsune miku, 1girl, classroom, flowers, holding bouquet, plaid skirt, sweater vest, very long hair, vocaloid, white shirt",
		imageA:
			"https://images.unsplash.com/photo-1524504388940-b1c1722653e1?q=80&w=1024&h=1536&auto=format&fit=crop",
		imageB:
			"https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?q=80&w=1024&h=1536&auto=format&fit=crop",
		refImage: "/assets/image_prompts/3aec7c8bf238231d1eee458f80308232.jpg",
		modelAId: "real-ref-5",
		modelBId: "real-ref-5-alt",
		isLoading: false,
		error: null,
	},
	{
		prompt:
			"phoebe (wuthering waves), 1girl, blonde hair, blue sash, cowboy shot, hair ornament, holding book, purple eyes, white hat, white shirt",
		imageA:
			"https://images.unsplash.com/photo-1544005313-94ddf0286df2?q=80&w=1024&h=1024&auto=format&fit=crop",
		imageB:
			"https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?q=80&w=1024&h=1024&auto=format&fit=crop",
		refImage: "/assets/image_prompts/3f60ff8973450b57550a4a67983a3318.png",
		modelAId: "real-ref-6",
		modelBId: "real-ref-6-alt",
		isLoading: false,
		error: null,
	},
];

export default function Arena() {
	const { models } = useData();
	const [activeTab, setActiveTab] = useState<ArenaTab>(() => {
		return (localStorage.getItem("arena_activeTab") as ArenaTab) || "battle";
	});

	const [battle, setBattle] = useState<BattleState>({
		prompt: "",
		imageA: "",
		imageB: "",
		modelAId: "",
		modelBId: "",
		isLoading: true,
		error: null,
	});

	const [nextBattle, setNextBattle] = useState<BattleState | null>(null);
	const [isDemoMode, setIsDemoMode] = useState(false);
	const [demoIndex, setDemoIndex] = useState(0);

	const prepareNextBattle = useCallback(
		async (currentDemoIndex: number) => {
			try {
				if (models.length < 2) {
					const nextDemoIndex =
						(currentDemoIndex + 1) % MOCK_BATTLE_SETS.length;
					setNextBattle(MOCK_BATTLE_SETS[nextDemoIndex]);
					return;
				}

				// 1. Pick two random models
				const shuffledModels = [...models].sort(() => 0.5 - Math.random());
				const modelA = shuffledModels[0];
				const modelB = shuffledModels[1];

				// 2. Fetch outputs for both
				const [outputsA, outputsB] = await Promise.all([
					fetchModelOutputs(modelA.id),
					fetchModelOutputs(modelB.id),
				]);

				// 3. Find common prompt + seed combinations
				const common = outputsA.filter((oa) =>
					outputsB.some((ob) => ob.prompt === oa.prompt && ob.seed === oa.seed),
				);

				if (common.length === 0) {
					const nextDemoIndex =
						(currentDemoIndex + 1) % MOCK_BATTLE_SETS.length;
					setNextBattle({
						...MOCK_BATTLE_SETS[nextDemoIndex],
						error: "No common images found. Preparing demo battle.",
					});
					return;
				}

				// 4. Pick a random common result
				const selection = common[Math.floor(Math.random() * common.length)];
				const outputB = outputsB.find(
					(ob) => ob.prompt === selection.prompt && ob.seed === selection.seed,
				) as ModelOutput;

				// Randomize order (A/B)
				const swap = Math.random() > 0.5;
				const imageA = getImageUrl(
					swap ? outputB.url : selection.url,
					swap ? outputB.mtime : selection.mtime,
				);
				const imageB = getImageUrl(
					swap ? selection.url : outputB.url,
					swap ? selection.mtime : outputB.mtime,
				);

				// Pre-fetch images
				const imgA = new Image();
				imgA.src = imageA;
				const imgB = new Image();
				imgB.src = imageB;
				if (selection.image_ref) {
					const imgRef = new Image();
					imgRef.src = getImageUrl(selection.image_ref);
				}

				setNextBattle({
					prompt: selection.prompt,
					imageA,
					imageB,
					refImage: selection.image_ref
						? getImageUrl(selection.image_ref)
						: undefined,
					modelAId: swap ? modelB.id : modelA.id,
					modelBId: swap ? modelA.id : modelB.id,
					isLoading: false,
					error: null,
					demoIndex: currentDemoIndex, // Store index to avoid stale swaps
				});
			} catch (err) {
				console.error("Failed to prepare next battle", err);
				const nextDemoIndex = (currentDemoIndex + 1) % MOCK_BATTLE_SETS.length;
				setNextBattle(MOCK_BATTLE_SETS[nextDemoIndex]);
			}
		},
		[models],
	);

	const startNewBattle = useCallback(async () => {
		if (nextBattle) {
			setBattle(nextBattle);
			setNextBattle(null);
			setIsDemoMode(models.length < 2 || !!nextBattle.error);
			// Start preparing the ONE AFTER focus
			prepareNextBattle(demoIndex);
			return;
		}

		setBattle((prev) => ({ ...prev, isLoading: true, error: null }));

		// If no nextBattle (first load), do it synchronously
		if (models.length < 2) {
			setIsDemoMode(true);
			setBattle(MOCK_BATTLE_SETS[demoIndex]);
			prepareNextBattle(demoIndex);
			return;
		}

		setIsDemoMode(false);
		try {
			// Initial load logic (essentially what prepareNextBattle does but sets 'battle' directly)
			const shuffledModels = [...models].sort(() => 0.5 - Math.random());
			const modelA = shuffledModels[0];
			const modelB = shuffledModels[1];
			const [outputsA, outputsB] = await Promise.all([
				fetchModelOutputs(modelA.id),
				fetchModelOutputs(modelB.id),
			]);
			const common = outputsA.filter((oa) =>
				outputsB.some((ob) => ob.prompt === oa.prompt && ob.seed === oa.seed),
			);

			if (common.length === 0) {
				setBattle(MOCK_BATTLE_SETS[demoIndex]);
				setIsDemoMode(true);
			} else {
				const selection = common[Math.floor(Math.random() * common.length)];
				const outputB = outputsB.find(
					(ob) => ob.prompt === selection.prompt && ob.seed === selection.seed,
				) as ModelOutput;
				const swap = Math.random() > 0.5;
				setBattle({
					prompt: selection.prompt,
					imageA: getImageUrl(swap ? outputB.url : selection.url),
					imageB: getImageUrl(swap ? selection.url : outputB.url),
					refImage: selection.image_ref
						? getImageUrl(selection.image_ref)
						: undefined,
					modelAId: swap ? modelB.id : modelA.id,
					modelBId: swap ? modelA.id : modelB.id,
					isLoading: false,
					error: null,
				});
			}
			prepareNextBattle(demoIndex);
		} catch (err) {
			setBattle(MOCK_BATTLE_SETS[demoIndex]);
			setIsDemoMode(true);
			prepareNextBattle(demoIndex);
		}
	}, [models, nextBattle, demoIndex, prepareNextBattle]);

	useEffect(() => {
		localStorage.setItem("arena_activeTab", activeTab);
	}, [activeTab]);

	useEffect(() => {
		if (activeTab === "battle" && !battle.prompt) {
			startNewBattle();
		}
	}, [activeTab, battle.prompt, startNewBattle]);

	const handleVote = (vote: "A" | "B" | "Tie" | "BothBad") => {
		console.log(
			`Vote cast: ${vote} for models ${battle.modelAId} vs ${battle.modelBId}`,
		);

		let nextDemoIdx = demoIndex;
		if (isDemoMode) {
			nextDemoIdx = (demoIndex + 1) % MOCK_BATTLE_SETS.length;
			setDemoIndex(nextDemoIdx);
		}

		startNewBattle();
	};

	return (
		<div className="max-w-[1800px] mx-auto px-6 py-4 h-[calc(100vh-80px)] flex flex-col overflow-hidden">
			{/* Compact Toolbar Header */}
			<div className="flex items-center justify-between mb-6 flex-none bg-white/50 dark:bg-slate-800/50 backdrop-blur-sm p-4 rounded-xl border border-slate-200 dark:border-slate-700">
				<div className="flex items-center gap-6">
					<div className="flex flex-col">
						<h2 className="text-xl font-bold text-slate-800 dark:text-slate-100 flex items-center gap-2">
							<Swords className="w-5 h-5 text-indigo-500" />
							Arena
						</h2>
					</div>

					{/* Tabs moved here */}
					<div className="flex items-center gap-1 p-1 bg-slate-100 dark:bg-slate-900/50 rounded-lg">
						<button
							type="button"
							onClick={() => setActiveTab("battle")}
							className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-sm transition-all ${
								activeTab === "battle"
									? "bg-white dark:bg-slate-700 text-indigo-600 dark:text-indigo-400 shadow-sm font-medium"
									: "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200"
							}`}
						>
							<Swords size={16} />
							Battle
						</button>
						<button
							type="button"
							onClick={() => setActiveTab("leaderboard")}
							className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-sm transition-all ${
								activeTab === "leaderboard"
									? "bg-white dark:bg-slate-700 text-indigo-600 dark:text-indigo-400 shadow-sm font-medium"
									: "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200"
							}`}
						>
							<Swords size={16} className="rotate-180" />
							Ranking
						</button>
					</div>
				</div>

				<div className="flex items-center gap-4">
					<p className="hidden lg:block text-slate-400 text-sm italic">
						Blindly compare models and vote
					</p>
					{isDemoMode && (
						<div className="flex items-center gap-2">
							<button
								type="button"
								onClick={() => {
									const nextIndex = (demoIndex + 1) % MOCK_BATTLE_SETS.length;
									setDemoIndex(nextIndex);
									setBattle(MOCK_BATTLE_SETS[nextIndex]);
								}}
								className="bg-indigo-50 hover:bg-indigo-100 dark:bg-indigo-900/30 dark:hover:bg-indigo-900/50 text-indigo-600 dark:text-indigo-400 px-3 py-1 rounded-full border border-indigo-200/50 dark:border-indigo-800/50 text-xs font-bold transition-colors"
							>
								Switch Sample
							</button>
							<div className="bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300 px-3 py-1 rounded-full border border-amber-200/50 dark:border-amber-800/50 text-xs font-bold animate-pulse">
								Demo Mode
							</div>
						</div>
					)}
				</div>
			</div>

			{/* Views */}
			<div className="flex-1 min-h-0 overflow-hidden">
				<div className="animate-in fade-in slide-in-from-bottom-2 duration-500 h-full w-full">
					{activeTab === "battle" ? (
						<div className="h-full w-full flex flex-col">
							{battle.isLoading ? (
								<div className="flex-1 flex items-center justify-center">
									<div className="flex flex-col items-center gap-4">
										<div className="w-12 h-12 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin"></div>
										<p className="text-slate-500 font-medium">
											Preparing Battle...
										</p>
									</div>
								</div>
							) : (
								<>
									{battle.error && !isDemoMode && (
										<div className="mb-4 bg-red-50 dark:bg-red-900/20 p-4 rounded-xl border border-red-200 dark:border-red-800 text-center">
											<p className="text-red-600 dark:text-red-400 font-medium flex items-center justify-center gap-2">
												{battle.error}
												<button
													type="button"
													onClick={startNewBattle}
													className="underline hover:no-underline"
												>
													Retry
												</button>
											</p>
										</div>
									)}
									<ArenaBattle
										key={`${battle.imageA}-${battle.imageB}-${isDemoMode}`}
										prompt={battle.prompt}
										imageA={battle.imageA}
										imageB={battle.imageB}
										refImage={battle.refImage}
										onVote={handleVote}
									/>
								</>
							)}
						</div>
					) : (
						<ArenaLeaderboard />
					)}
				</div>
			</div>
		</div>
	);
}
