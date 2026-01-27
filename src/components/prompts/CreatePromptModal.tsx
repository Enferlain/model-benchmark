import { Image as ImageIcon, X } from "lucide-react";
import type React from "react";

interface CreatePromptModalProps {
	isOpen: boolean;
	onClose: () => void;
	onCreate: (e: React.FormEvent) => void;
	newPromptText: string;
	setNewPromptText: (text: string) => void;
	newPromptImage: File | null;
	setNewPromptImage: (file: File | null) => void;
}

export const CreatePromptModal: React.FC<CreatePromptModalProps> = ({
	isOpen,
	onClose,
	onCreate,
	newPromptText,
	setNewPromptText,
	newPromptImage,
	setNewPromptImage,
}) => {
	if (!isOpen) return null;

	return (
		<div
			className="fixed inset-0 z-50 bg-white/90 dark:bg-slate-900/95 backdrop-blur-md flex items-center justify-center p-8"
			onClick={onClose}
			onKeyDown={(e) => e.key === "Escape" && onClose()}
		>
			<div
				role="dialog"
				aria-modal="true"
				aria-labelledby="create-prompt-title"
				className="w-full max-w-2xl flex flex-col h-full max-h-[600px] bg-white dark:bg-slate-800 rounded-2xl shadow-2xl border border-slate-200 dark:border-white/10 animation-fade-in-up"
				onClick={(e) => e.stopPropagation()}
			>
				<div className="p-6 border-b border-slate-200 dark:border-white/5 flex justify-between items-center">
					<h2
						id="create-prompt-title"
						className="text-xl font-bold text-slate-800 dark:text-slate-100"
					>
						Create New Prompt
					</h2>
					<button
						onClick={onClose}
						className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
					>
						<X size={24} />
					</button>
				</div>

				<form
					onSubmit={onCreate}
					className="flex-1 p-6 flex flex-col gap-6 overflow-y-auto"
				>
					<div>
						<label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
							Prompt Text
						</label>
						<textarea
							required
							value={newPromptText}
							onChange={(e) => setNewPromptText(e.target.value)}
							className="w-full h-40 bg-slate-50 dark:bg-slate-900/50 border border-slate-200 dark:border-white/10 rounded-xl p-4 font-mono text-sm focus:ring-2 focus:ring-blue-500 outline-none"
							placeholder="Enter your new prompt here..."
						/>
					</div>

					<div>
						<label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
							Image to Prompt (Drag to Interrogate)
						</label>
						<div className="border-2 border-dashed border-slate-200 dark:border-white/10 rounded-xl p-8 text-center cursor-pointer hover:bg-slate-50 dark:hover:bg-white/5 transition-colors relative">
							<input
								type="file"
								accept="image/*"
								onChange={(e) => setNewPromptImage(e.target.files?.[0] || null)}
								className="absolute inset-0 opacity-0 cursor-pointer"
							/>
							{newPromptImage ? (
								<div className="flex flex-col items-center text-blue-500">
									<ImageIcon size={32} className="mb-2" />
									<span className="font-medium">{newPromptImage.name}</span>
								</div>
							) : (
								<div className="flex flex-col items-center text-slate-400">
									<ImageIcon size={32} className="mb-2" />
									<span>Click or drag to upload an image</span>
								</div>
							)}
						</div>
					</div>
				</form>

				<div className="p-6 border-t border-slate-200 dark:border-white/5 flex justify-end gap-3">
					<button
						type="button"
						onClick={onClose}
						className="px-4 py-2 text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-white/5 rounded-lg text-sm font-medium"
					>
						Cancel
					</button>
					<button
						type="submit"
						className="px-6 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-sm font-medium shadow-lg shadow-blue-500/20"
					>
						Create Prompt
					</button>
				</div>
			</div>
		</div>
	);
};
