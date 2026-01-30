import type React from "react";

interface DeleteConfirmModalProps {
	isOpen: boolean;
	onClose: () => void;
	onConfirm: () => void;
	modelName: string;
}

export const DeleteConfirmModal: React.FC<DeleteConfirmModalProps> = ({
	isOpen,
	onClose,
	onConfirm,
	modelName,
}) => {
	if (!isOpen) return null;

	return (
		<div className="fixed inset-0 z-50 flex items-center justify-center p-4">
			<button
				type="button"
				className="absolute inset-0 bg-black/50 backdrop-blur-sm w-full h-full border-none p-0 m-0"
				onClick={onClose}
				aria-label="Close modal"
			/>
			<div className="relative bg-white dark:bg-slate-800 rounded-2xl shadow-2xl max-w-md w-full p-6 border border-slate-200 dark:border-slate-700 animate-in fade-in zoom-in-95 duration-200">
				<h3 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-2">
					Remove Model?
				</h3>
				<p className="text-slate-600 dark:text-slate-300 mb-6">
					Are you sure you want to remove{" "}
					<span className="font-semibold">{modelName}</span> from the database?
					<br />
					<span className="text-xs text-slate-500 block mt-2">
						(The file will NOT be deleted from disk)
					</span>
				</p>

				<div className="flex justify-end gap-3">
					<button
						type="button"
						onClick={onClose}
						className="px-4 py-2 text-sm font-medium text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-700 rounded-lg transition-colors"
					>
						Cancel
					</button>
					<button
						type="button"
						onClick={() => onConfirm()}
						className="px-4 py-2 text-sm font-medium text-white bg-red-600 hover:bg-red-700 rounded-lg shadow-lg shadow-red-500/30 transition-colors"
					>
						Remove
					</button>
				</div>
			</div>
		</div>
	);
};
