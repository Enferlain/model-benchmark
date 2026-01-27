import { ChevronLeft, ChevronRight } from "lucide-react";

interface ControlsProps {
	onMoveRight: () => void;
	onMoveLeft: () => void;
	canMoveRight: boolean;
	canMoveLeft: boolean;
}

export function Controls({
	onMoveRight,
	onMoveLeft,
	canMoveRight,
	canMoveLeft,
}: ControlsProps) {
	return (
		<div className="flex flex-col gap-2 justify-center px-2">
			<button
				onClick={onMoveRight}
				disabled={!canMoveRight}
				className={`
          p-2 rounded-lg border transition-all
          ${
						canMoveRight
							? "bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 shadow-sm"
							: "bg-slate-50 dark:bg-slate-900 border-transparent text-slate-300 dark:text-slate-700 cursor-not-allowed"
					}
        `}
				title="Move Selected to Queue"
			>
				<ChevronRight size={20} />
			</button>

			<button
				onClick={onMoveLeft}
				disabled={!canMoveLeft}
				className={`
          p-2 rounded-lg border transition-all
          ${
						canMoveLeft
							? "bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 shadow-sm"
							: "bg-slate-50 dark:bg-slate-900 border-transparent text-slate-300 dark:text-slate-700 cursor-not-allowed"
					}
        `}
				title="Remove Selected from Queue"
			>
				<ChevronLeft size={20} />
			</button>
		</div>
	);
}
