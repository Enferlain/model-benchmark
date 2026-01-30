import type { DragEndEvent } from "@dnd-kit/core";
import { arrayMove } from "@dnd-kit/sortable";
import { Image as ImageIcon } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { CreatePromptModal } from "../components/prompts/CreatePromptModal";
import { PromptDetailEditor } from "../components/prompts/PromptDetailEditor";
import { PromptList } from "../components/prompts/PromptList";
import {
	createPrompt,
	deletePrompt,
	setAllPromptsEnabled,
	shufflePrompts,
	updatePromptText,
} from "../services/api";
import { useData } from "../context/DataContext";
import type { PromptData } from "../types";


export default function PromptEditor() {
	const {
		allPrompts: prompts,
		isLoadingPrompts: isLoading,
		refreshPrompts: loadPrompts,
	} = useData();

	const [selectedId, setSelectedId] = useState<string | null>(() => {
		return localStorage.getItem("promptEditor_selectedId");
	});
	const [editText, setEditText] = useState("");
	const [editAlias, setEditAlias] = useState("");
	const [isSaving, setIsSaving] = useState(false);
	const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
	const [searchQuery, setSearchQuery] = useState("");
	
	// Create Modal State
	const [newPromptText, setNewPromptText] = useState("");
	const [newPromptImage, setNewPromptImage] = useState<File | null>(null);

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

	// Sync local edit state with selected prompt
	useEffect(() => {
		if (selectedPrompt) {
			setEditText(selectedPrompt.text || "");
			setEditAlias(selectedPrompt.alias || "");
		} else {
			setEditText("");
			setEditAlias("");
		}
	}, [selectedPrompt]);

	const handleDragEnd = useCallback(
		async (event: DragEndEvent) => {
			const { active, over } = event;

			if (over && active.id !== over.id) {
				const oldIndex = prompts.findIndex((p) => p.id === active.id);
				const newIndex = prompts.findIndex((p) => p.id === over.id);

				const newPrompts = arrayMove(prompts, oldIndex, newIndex);
				// Update backend
				try {
					await updatePromptText("order", {
						order: newPrompts.map((p) => p.id),
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
		async (e: React.MouseEvent, prompt: PromptData) => {
			e.stopPropagation();
			const newEnabled = !prompt.enabled;
			try {
				await updatePromptText(prompt.id, { enabled: newEnabled });
				await loadPrompts();
			} catch (error) {
				console.error("Error toggling prompt:", error);
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
		async (e: React.FormEvent) => {
			e.preventDefault();
			const formData = new FormData();
			formData.append("text", newPromptText);
			if (newPromptImage) formData.append("image", newPromptImage);

			try {
				const result = await createPrompt(formData);
				await loadPrompts();
				setSelectedId(result.id);
				setIsCreateModalOpen(false);
				// Reset modal fields
				setNewPromptText("");
				setNewPromptImage(null);
			} catch (error) {
				console.error("Error creating prompt:", error);
			}
		},
		[loadPrompts, newPromptText, newPromptImage],
	);

	const handleSave = useCallback(async () => {
		if (!selectedId) return;
		setIsSaving(true);
		try {
			await updatePromptText(selectedId, {
				text: editText,
				alias: editAlias,
			});
			await loadPrompts();
		} catch (error) {
			console.error("Error saving prompt:", error);
		} finally {
			setIsSaving(false);
		}
	}, [selectedId, editText, editAlias, loadPrompts]);

	const handleDeletePrompt = useCallback(
		async (e: React.MouseEvent, id: string) => {
			e.stopPropagation();
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


	const filteredPrompts = useMemo(() => {
		if (!searchQuery) return prompts;
		const query = searchQuery.toLowerCase();
		return prompts.filter((p) => p.text.toLowerCase().includes(query));
	}, [prompts, searchQuery]);

	return (
		<div className="max-w-[1800px] mx-auto h-[calc(100vh-100px)] pt-6 px-6 flex gap-6">
			<PromptList
				prompts={filteredPrompts}
				selectedId={selectedId}
				onSelect={setSelectedId}
				onToggle={handleToggleEnable}
				onDragEnd={handleDragEnd}
				onShuffle={handleShuffle}
				onEnableAll={handleEnableAll}
				onCreate={() => setIsCreateModalOpen(true)}
				searchQuery={searchQuery}
				onSearchChange={setSearchQuery}
				isLoading={isLoading}
				onDelete={handleDeletePrompt}
			/>

			{selectedPrompt ? (
				<PromptDetailEditor
					prompt={selectedPrompt}
					editText={editText}
					editAlias={editAlias}
					isSaving={isSaving}
					isDirty={
						editText !== (selectedPrompt.text || "") ||
						editAlias !== (selectedPrompt.alias || "")
					}
					onTextChange={setEditText}
					onAliasChange={setEditAlias}
					onSave={handleSave}
				/>
			) : (
				<div className="flex-1 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 shadow-sm flex flex-col items-center justify-center text-slate-400 p-8 text-center">
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

			<CreatePromptModal
				isOpen={isCreateModalOpen}
				onClose={() => setIsCreateModalOpen(false)}
				onCreate={handleCreatePrompt}
				newPromptText={newPromptText}
				setNewPromptText={setNewPromptText}
				newPromptImage={newPromptImage}
				setNewPromptImage={setNewPromptImage}
			/>
		</div>
	);
}
