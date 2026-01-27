import type { ModelData } from "../../../types";
import { TransferList } from "../../TransferList";

interface TransferListSectionProps {
	models: ModelData[];
	selectedModelIds: string[];
	setSelectedModelIds: (ids: string[]) => void;
}

export function TransferListSection({
	models,
	selectedModelIds,
	setSelectedModelIds,
}: TransferListSectionProps) {
	return (
		<div className="bg-white/50 dark:bg-slate-900/40 rounded-[22px] p-6 backdrop-blur-sm flex flex-col h-full border border-slate-200/50 dark:border-white/5">
			<div className="flex items-center justify-between mb-6">
				<div>
					<h3 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-slate-900 to-slate-600 dark:from-white dark:to-slate-400">
						Model Library
					</h3>
					<p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
						Select models to compare or drag to reorder.
					</p>
				</div>
				<div className="flex items-center gap-2">
					<span className="text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-widest bg-slate-100 dark:bg-slate-800 px-3 py-1.5 rounded-full border border-slate-200 dark:border-white/5">
						{models.length} Models
					</span>
				</div>
			</div>

			<div className="flex-1 min-h-[400px]">
				<TransferList
					models={models}
					selectedModelIds={selectedModelIds}
					onChange={setSelectedModelIds}
				/>
			</div>
		</div>
	);
}
