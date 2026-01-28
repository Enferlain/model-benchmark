import type { DragEndEvent } from "@dnd-kit/core";
import { arrayMove } from "@dnd-kit/sortable";
import { Image as ImageIcon } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { CreatePromptModal } from "../components/prompts/CreatePromptModal";
import { PromptDetailEditor } from "../components/prompts/PromptDetailEditor";
import { PromptList } from "../components/prompts/PromptList";
import { useData } from "../context/DataContext";
import {
	createPrompt,
	deletePrompt,
	setAllPromptsEnabled,
	shufflePrompts,
	updatePromptText,
} from "../services/api";

export default function PromptEditor() {
	const {
		allPrompts: prompts,
		isLoadingPrompts: isLoading,
		refreshPrompts: loadPrompts,
	} = useData();
	const [selectedId, setSelectedId] = useState<string | null>(() => {
		return localStorage.getItem("promptEditor_selectedId");
	});
	const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
	const [searchQuery, setSearchQuery] = useState("");

	const selectedPrompt = useMemo(() => {
		return prompts.find((p) => p.id === selectedId) || null;
	}, [prompts, selectedId]);

	useEffect(() => {
		if (selectedId) {
			localStorage.setItem("promptEditor_selectedId", selectedId);
		} else {
			localStorage.removeItem("promptEditor_selectedId");
		}
	}, [selectedId]);

	const handleDragEnd = useCallback(
		async (event: DragEndEvent) => {
			const { active, over } = event;

			if (over && active.id !== over.id) {
				const oldIndex = prompts.findIndex((p) => p.id === (active as any).id);
				const newIndex = prompts.findIndex((p) => p.id === (over as any).id);

				const newPrompts = arrayMove(prompts, oldIndex, newIndex);
				// Update backend
				try {
					await updatePromptText("order", {
						order: newPrompts.map((p: any) => p.id),
					});
					await loadPrompts();
				} catch (error) {
					console.error("Error updating prompt order:", error);
				}
			}
		},
		[prompts, loadPrompts],
	);

	const handleToggleEnable = useCallback(
		async (id: string, enabled: boolean) => {
			try {
				await updatePromptText(id, { enabled });
				await loadPrompts();
			} catch (error) {
				console.error("Error toggling prompt:", error);
			}
		},
		[loadPrompts],
	);

	const handleUpdateText = useCallback(
		async (id: string, text: string) => {
			try {
				await updatePromptText(id, text);
				await loadPrompts();
			} catch (error) {
				console.error("Error updating prompt:", error);
			}
		},
		[loadPrompts],
	);

	const handleShuffle = useCallback(async () => {
		try {
			await shufflePrompts();
			await loadPrompts();
		} catch (error) {
			console.error("Error shuffling prompts:", error);
		}
	}, [loadPrompts]);

	const handleEnableAll = useCallback(
		async (enabled: boolean) => {
			try {
				await setAllPromptsEnabled(enabled);
				await loadPrompts();
			} catch (error) {
				console.error("Error enabling/disabling all prompts:", error);
			}
		},
		[loadPrompts],
	);

	const handleCreatePrompt = useCallback(
		async (text: string, image: File | null) => {
			const formData = new FormData();
			formData.append("text", text);
			if (image) formData.append("image", image);

			try {
				const result = await createPrompt(formData);
				await loadPrompts();
				setSelectedId(result.id);
				setIsCreateModalOpen(false);
			} catch (error) {
				console.error("Error creating prompt:", error);
			}
		},
		[loadPrompts],
	);

	const handleDeletePrompt = useCallback(
		async (id: string) => {
			try {
				await deletePrompt(id);
				if (selectedId === id) setSelectedId(null);
				await loadPrompts();
			} catch (error) {
				console.error("Error deleting prompt:", error);
			}
		},
		[selectedId, loadPrompts],
	);

	const handleSetDefaultImage = useCallback(
		async (id: string, imagePath: string) => {
			try {
				await updatePromptText(id, { default_image: imagePath });
				await loadPrompts();
			} catch (error) {
				console.error("Error setting default image:", error);
			}
		},
		[loadPrompts],
	);

	const filteredPrompts = useMemo(() => {
		if (!searchQuery) return prompts;
		const query = searchQuery.toLowerCase();
		return prompts.filter((p) => p.text.toLowerCase().includes(query));
	}, [prompts, searchQuery]);

	return (
		<div className="h-[calc(100vh-64px)] overflow-hidden flex flex-col">
			<div className="flex-1 flex overflow-hidden">
				<div className="w-80 border-r border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900/50 flex flex-col overflow-hidden">
					<PromptList
						prompts={filteredPrompts}
						selectedId={selectedId}
						onSelect={setSelectedId}
						onToggleEnable={handleToggleEnable}
						onDragEnd={handleDragEnd}
						onShuffle={handleShuffle}
						onEnableAll={handleEnableAll}
						onCreateClick={() => setIsCreateModalOpen(true)}
						searchQuery={searchQuery}
						onSearchChange={setSearchQuery}
						isLoading={isLoading}
					/>
				</div>

				<div className="flex-1 bg-white dark:bg-slate-800 overflow-y-auto">
					{selectedPrompt ? (
						<PromptDetailEditor
							prompt={selectedPrompt}
							onUpdateText={handleUpdateText}
							onDelete={handleDeletePrompt}
							onSetDefaultImage={handleSetDefaultImage}
						/>
					) : (
						<div className="h-full flex flex-col items-center justify-center text-slate-400 p-8 text-center">
							<div className="w-16 h-16 rounded-full bg-slate-100 dark:bg-slate-700/50 flex items-center justify-center mb-4">
								<ImageIcon size={32} className="text-slate-300" />
							</div>
							<h3 className="text-lg font-medium text-slate-600 dark:text-slate-300">
								No Prompt Selected
							</h3>
							<p className="max-w-xs mt-2">
								Select a prompt from the list to edit its details or view its
								associated benchmark images.
							</p>
						</div>
					)}
				</div>
			</div>

			<CreatePromptModal
				isOpen={isCreateModalOpen}
				onClose={() => setIsCreateModalOpen(false)}
				onConfirm={handleCreatePrompt}
			/>
		</div>
	);
}
