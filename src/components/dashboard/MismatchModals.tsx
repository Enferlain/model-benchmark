import { Info, Loader2 } from "lucide-react";
import type { CoverageCheckResult, ParamCheckResult } from "../../services/api";

interface MismatchModalsProps {
	showMismatchModal: boolean;
	setShowMismatchModal: (show: boolean) => void;
	paramMismatch: ParamCheckResult | null;
	setParamMismatch: (val: ParamCheckResult | null) => void;
	handleMatchExisting: () => void;
	handleArchiveAndGenerate: () => Promise<void>;
	showCoverageModal: boolean;
	setShowCoverageModal: (show: boolean) => void;
	coverageMismatch: CoverageCheckResult | null;
	setCoverageMismatch: (val: CoverageCheckResult | null) => void;
	handleAnalyzeCommonOnly: () => Promise<void>;
	handleGenerateMissing: () => Promise<void>;
}

export function MismatchModals({
	showMismatchModal,
	setShowMismatchModal,
	paramMismatch,
	setParamMismatch,
	handleMatchExisting,
	handleArchiveAndGenerate,
	showCoverageModal,
	setShowCoverageModal,
	coverageMismatch,
	setCoverageMismatch,
	handleAnalyzeCommonOnly,
	handleGenerateMissing,
}: MismatchModalsProps) {
	return (
		<>
			{/* Parameter Mismatch Modal */}
			{showMismatchModal && paramMismatch && (
				<div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
					<div className="bg-white dark:bg-slate-800 rounded-2xl shadow-2xl p-6 max-w-md w-full mx-4 border border-slate-200 dark:border-slate-700">
						<div className="flex items-center gap-3 mb-4">
							<div className="w-10 h-10 rounded-full bg-amber-100 dark:bg-amber-900/30 flex items-center justify-center">
								<span className="text-amber-600 dark:text-amber-400 text-xl">
									⚠️
								</span>
							</div>
							<div>
								<h3 className="text-lg font-semibold text-slate-800 dark:text-slate-100">
									Parameter Mismatch
								</h3>
								<p className="text-sm text-slate-500 dark:text-slate-400">
									Existing images have different settings
								</p>
							</div>
						</div>

						<div className="bg-slate-50 dark:bg-slate-900/50 rounded-xl p-4 mb-4 space-y-2 text-sm">
							<div className="grid grid-cols-2 gap-4">
								<div>
									<p className="text-xs font-medium text-slate-400 mb-1">
										Existing Settings
									</p>
									<div className="space-y-1 text-slate-600 dark:text-slate-300">
										<p>Steps: {paramMismatch.existing_params?.steps}</p>
										<p>CFG: {paramMismatch.existing_params?.cfg}</p>
										<p>Sampler: {paramMismatch.existing_params?.sampler}</p>
										<p>
											Size: {paramMismatch.existing_params?.width}×
											{paramMismatch.existing_params?.height}
										</p>
									</div>
								</div>
								<div>
									<p className="text-xs font-medium text-slate-400 mb-1">
										Your Settings
									</p>
									<div className="space-y-1 text-slate-600 dark:text-slate-300">
										<p>Steps: {paramMismatch.current_params.steps}</p>
										<p>CFG: {paramMismatch.current_params.cfg}</p>
										<p>Sampler: {paramMismatch.current_params.sampler}</p>
										<p>
											Size: {paramMismatch.current_params.width}×
											{paramMismatch.current_params.height}
										</p>
									</div>
								</div>
							</div>

							{paramMismatch.mismatched_models.length > 0 && (
								<div className="pt-2 border-t border-slate-200 dark:border-slate-700 mt-2">
									<p className="text-xs text-slate-500">
										Affected models:{" "}
										{paramMismatch.mismatched_models
											.map((m) => m.name)
											.join(", ")}
									</p>
								</div>
							)}
						</div>

						<div className="flex gap-3">
							<button
								type="button"
								onClick={handleMatchExisting}
								className="flex-1 px-4 py-2.5 bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-200 rounded-xl hover:bg-slate-200 dark:hover:bg-slate-600 transition-colors text-sm font-medium"
							>
								Match Existing
							</button>
							<button
								type="button"
								onClick={handleArchiveAndGenerate}
								className="flex-1 px-4 py-2.5 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-colors text-sm font-medium"
							>
								Archive & Regenerate
							</button>
						</div>

						<button
							type="button"
							onClick={() => {
								setShowMismatchModal(false);
								setParamMismatch(null);
							}}
							className="w-full mt-3 px-4 py-2 text-slate-500 hover:text-slate-700 dark:hover:text-slate-300 text-sm transition-colors"
						>
							Cancel
						</button>
					</div>
				</div>
			)}

			{/* Coverage Mismatch Modal */}
			{showCoverageModal && coverageMismatch && (
				<div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
					<div className="bg-white dark:bg-slate-800 rounded-2xl shadow-2xl p-6 max-w-md w-full mx-4 border border-slate-200 dark:border-slate-700">
						<div className="flex items-center gap-3 mb-4">
							<div className="w-10 h-10 rounded-full bg-orange-100 dark:bg-orange-900/30 flex items-center justify-center">
								<span className="text-orange-600 dark:text-orange-400 text-xl">
									📊
								</span>
							</div>
							<div>
								<h3 className="text-lg font-semibold text-slate-800 dark:text-slate-100">
									Prompt Coverage Mismatch
								</h3>
								<p className="text-sm text-slate-500 dark:text-slate-400">
									Models have different prompt coverage
								</p>
							</div>
						</div>

						<div className="bg-slate-50 dark:bg-slate-900/50 rounded-xl p-4 mb-4 text-sm">
							<p className="text-slate-600 dark:text-slate-300 mb-2">
								<span className="font-medium">
									{coverageMismatch.common_count}
								</span>{" "}
								prompts are common to all models.
								{coverageMismatch.image_count_mismatch && (
									<span className="text-orange-500 ml-2">
										(image counts vary)
									</span>
								)}
							</p>
							<div className="space-y-1 text-slate-500 dark:text-slate-400 text-xs">
								{coverageMismatch.model_coverage.map((m) => (
									<div key={m.name} className="flex justify-between">
										<span className="truncate max-w-[150px]">{m.name}</span>
										<span>
											{m.image_count} images / {m.count} prompts
											{m.missing_count > 0 && (
												<span className="text-orange-500 ml-1">
													({m.missing_count} missing)
												</span>
											)}
										</span>
									</div>
								))}
							</div>
						</div>

						<div className="flex gap-3">
							<button
								type="button"
								onClick={handleAnalyzeCommonOnly}
								className="flex-1 px-4 py-2.5 bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-200 rounded-xl hover:bg-slate-200 dark:hover:bg-slate-600 transition-colors text-sm font-medium"
							>
								Analyze Common Only
							</button>
							<button
								type="button"
								onClick={handleGenerateMissing}
								className="flex-1 px-4 py-2.5 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-colors text-sm font-medium"
							>
								Generate Missing
							</button>
						</div>

						<button
							type="button"
							onClick={() => {
								setShowCoverageModal(false);
								setCoverageMismatch(null);
							}}
							className="w-full mt-3 px-4 py-2 text-slate-500 hover:text-slate-700 dark:hover:text-slate-300 text-sm transition-colors"
						>
							Cancel
						</button>
					</div>
				</div>
			)}
		</>
	);
}
