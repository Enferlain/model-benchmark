import {
	closestCenter,
	DndContext,
	type DragEndEvent,
	DragOverlay,
	KeyboardSensor,
	PointerSensor,
	useSensor,
	useSensors,
} from "@dnd-kit/core";
import {
	SortableContext,
	sortableKeyboardCoordinates,
	useSortable,
	verticalListSortingStrategy,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import {
	CheckSquare,
	FileText,
	Loader2,
	Plus,
	Search,
	Shuffle,
	Square,
	Trash2,
} from "lucide-react";
import React, { useState } from "react";
import { createPortal } from "react-dom";
import { Virtuoso } from "react-virtuoso";
import type { PromptData } from "../../types";

// Helper to convert asset path to thumbnail URL
const getThumbnailUrl = (imagePath: string, size = 256) =>
	`/api/thumbnails/${imagePath.replace(/^\/assets\//, "")}?w=${size}&h=${size}`;

interface PromptListProps {
	prompts: PromptData[];
	isLoading: boolean;
	searchQuery: string;
	selectedId: string | null;
	onSearchChange: (q: string) => void;
	onSelect: (id: string | null) => void;
	onCreate: () => void;
	onShuffle: () => void;
	onEnableAll: (enabled: boolean) => void;
	onToggle: (e: React.MouseEvent, prompt: PromptData) => void;
	onDelete: (e: React.MouseEvent, id: string) => void;
	onDragEnd: (event: DragEndEvent) => void;
}

export const PromptList: React.FC<PromptListProps> = ({
	prompts,
	isLoading,
	searchQuery,
	selectedId,
	onSearchChange,
	onSelect,
	onCreate,
	onShuffle,
	onEnableAll,
	onToggle,
	onDelete,
	onDragEnd,
}) => {
	// Track active drag item for overlay
	const [activeId, setActiveId] = useState<string | null>(null);

	// DnD Sensors
	const sensors = useSensors(
		useSensor(PointerSensor, { activationConstraint: { distance: 8 } }),
		useSensor(KeyboardSensor, {
			coordinateGetter: sortableKeyboardCoordinates,
		}),
	);

	const filteredPrompts = React.useMemo(() => {
		if (!Array.isArray(prompts)) return [];
		const q = searchQuery.toLowerCase();
		return prompts.filter((p) => {
			if (!p) return false;
			const text = p.text || "";
			const id = p.id || "";
			const alias = p.alias || "";
			return (
				text.toLowerCase().includes(q) ||
				id.toLowerCase().includes(q) ||
				alias.toLowerCase().includes(q)
			);
		});
	}, [prompts, searchQuery]);

	const allEnabled =
		filteredPrompts.length > 0 && filteredPrompts.every((p) => p.enabled);

	const activePrompt = React.useMemo(
		() => (activeId ? filteredPrompts.find((p) => p.id === activeId) : null),
		[activeId, filteredPrompts],
	);

	return (
		<div className="w-1/3 min-w-[320px] max-w-[400px] flex flex-col bg-white dark:bg-slate-800/50 rounded-2xl shadow-lg border border-slate-200 dark:border-white/5 overflow-hidden h-full">
			<div className="p-4 border-b border-slate-200 dark:border-white/5 flex gap-2">
				<div className="relative flex-1">
					<Search
						className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"
						size={18}
					/>
					<input
						type="text"
						placeholder="Search prompts..."
						value={searchQuery}
						onChange={(e) => onSearchChange(e.target.value)}
						className="w-full pl-10 pr-4 py-2 bg-slate-100 dark:bg-slate-900/50 border-none rounded-xl text-sm focus:ring-2 focus:ring-blue-500 outline-none transition-all"
					/>
				</div>
				<button
					type="button"
					onClick={onCreate}
					className="p-2 bg-blue-600 hover:bg-blue-500 text-white rounded-xl transition-colors shadow-lg shadow-blue-500/20"
					title="Add New Prompt"
				>
					<Plus size={20} />
				</button>
			</div>

			{/* Bulk Actions Bar */}
			<div className="px-4 py-2 border-b border-slate-200 dark:border-white/5 flex gap-2 overflow-x-auto">
				<button
					type="button"
					onClick={onShuffle}
					className="p-1.5 px-3 bg-slate-100 dark:bg-slate-700/50 hover:bg-slate-200 dark:hover:bg-slate-600 rounded-lg text-xs font-medium text-slate-600 dark:text-slate-300 flex items-center gap-1.5 transition-colors whitespace-nowrap"
				>
					<Shuffle size={12} /> Shuffle
				</button>
				<button
					type="button"
					onClick={() => onEnableAll(!allEnabled)}
					className="p-1.5 px-3 bg-slate-100 dark:bg-slate-700/50 hover:bg-slate-200 dark:hover:bg-slate-600 rounded-lg text-xs font-medium text-slate-600 dark:text-slate-300 flex items-center gap-1.5 transition-colors whitespace-nowrap"
				>
					{allEnabled ? (
						<>
							<Square size={12} /> Disable All
						</>
					) : (
						<>
							<CheckSquare size={12} /> Enable All
						</>
					)}
				</button>
			</div>

			<div className="flex-1 relative">
				{isLoading ? (
					<div className="flex justify-center p-8 text-slate-400">
						<Loader2 className="animate-spin" />
					</div>
				) : filteredPrompts.length === 0 ? (
					<div className="text-center p-8 text-slate-500 text-sm">
						No prompts found.
					</div>
				) : (
					<DndContext
						sensors={sensors}
						collisionDetection={closestCenter}
						onDragStart={(event) => setActiveId(event.active.id as string)}
						onDragEnd={(event) => {
							setActiveId(null);
							onDragEnd(event);
						}}
						onDragCancel={() => setActiveId(null)}
					>
						<SortableContext
							items={filteredPrompts.map((p) => p.id)}
							strategy={verticalListSortingStrategy}
						>
							<Virtuoso
								style={{ height: "100%" }}
								data={filteredPrompts}
								itemContent={(index, prompt) => (
									<div className="p-1 px-2">
										<SortableItem
											key={prompt.id}
											prompt={prompt}
											idx={index}
											isSelected={selectedId === prompt.id}
											onSelect={onSelect}
											onToggle={onToggle}
											onDelete={onDelete}
										/>
									</div>
								)}
							/>
						</SortableContext>
						{createPortal(
							<DragOverlay zIndex={100} adjustScale={true}>
								{activePrompt ? (
									<div className="opacity-90 scale-105 shadow-2xl rounded-xl overflow-hidden pointer-events-none">
										<PromptItemDisplay
											prompt={activePrompt}
											idx={filteredPrompts.findIndex((p) => p.id === activeId)}
											isSelected={false}
										/>
									</div>
								) : null}
							</DragOverlay>,
							document.body,
						)}
					</DndContext>
				)}
			</div>
		</div>
	);
};

interface SortableItemProps {
	prompt: PromptData;
	idx: number;
	isSelected: boolean;
	onSelect: (id: string | null) => void;
	onToggle: (e: React.MouseEvent, prompt: PromptData) => void;
	onDelete: (e: React.MouseEvent, id: string) => void;
}

// Sortable Item Component
const SortableItem = React.memo(function SortableItem({
	prompt,
	idx,
	isSelected,
	onSelect,
	onToggle,
	onDelete,
}: SortableItemProps) {
	const {
		attributes,
		listeners,
		setNodeRef,
		transform,
		transition,
		isDragging,
	} = useSortable({ id: prompt.id });
	const style = {
		transform: CSS.Transform.toString(transform),
		transition,
		opacity: isDragging ? 0.3 : 1,
		zIndex: isDragging ? 50 : "auto",
	};

	return (
		<button
			type="button"
			ref={setNodeRef}
			style={style}
			{...attributes}
			{...listeners}
			onClick={() => onSelect(prompt.id)}
			onKeyDown={(e) => {
				if (e.key === "Enter" || e.key === " ") {
					onSelect(prompt.id);
				}
			}}
			className={`w-full text-left cursor-grab active:cursor-grabbing ${!prompt.enabled ? "opacity-50 grayscale" : ""}`}
		>
			<PromptItemDisplay
				prompt={prompt}
				idx={idx}
				isSelected={isSelected}
				onToggle={onToggle}
				onDelete={onDelete}
			/>
		</button>
	);
});

interface PromptItemDisplayProps {
	prompt: PromptData;
	idx: number;
	isSelected: boolean;
	onToggle?: (e: React.MouseEvent, prompt: PromptData) => void;
	onDelete?: (e: React.MouseEvent, id: string) => void;
}

const PromptItemDisplay = React.memo<PromptItemDisplayProps>(
	({ prompt, idx, isSelected, onToggle, onDelete }) => {
		return (
			<div
				className={`group p-3 rounded-xl transition-colors border border-transparent relative select-none ${
					isSelected
						? "bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800/50 shadow-sm"
						: "bg-white dark:bg-slate-800 hover:bg-slate-50 dark:hover:bg-white/5 hover:border-slate-200 dark:hover:border-white/10"
				}`}
			>
				<div className="absolute -top-1 -left-1 z-10 w-5 h-5 rounded-full bg-slate-800 dark:bg-slate-700 text-white text-[9px] font-bold flex items-center justify-center border-2 border-white dark:border-slate-800 shadow-sm">
					{idx + 1}
				</div>

				<div className="flex gap-3">
					<div className="w-16 h-16 shrink-0 bg-slate-100 dark:bg-slate-900 rounded-lg overflow-hidden border border-slate-200 dark:border-white/5 flex items-center justify-center text-slate-300">
						{prompt.image ? (
							<img
								src={getThumbnailUrl(prompt.image)}
								alt="ref"
								className="w-full h-full object-cover"
								loading="lazy"
								decoding="async"
							/>
						) : (
							<FileText size={24} />
						)}
					</div>
					<div className="flex-1 min-w-0 flex flex-col justify-center">
						<div className="flex justify-between items-center mb-0.5">
							<span className="text-xs font-bold text-slate-700 dark:text-slate-200 truncate">
								{prompt.alias ? prompt.alias : prompt.id}
							</span>
							<div
								className="flex items-center gap-1"
								onMouseDown={(e) => e.stopPropagation()}
								role="none"
							>
								{onToggle && (
									<button
										type="button"
										onClick={(e) => onToggle(e, prompt)}
										className={`w-8 h-4 rounded-full p-0.5 cursor-pointer transition-colors ${prompt.enabled ? "bg-green-500" : "bg-slate-300 dark:bg-slate-600"}`}
										title={prompt.enabled ? "Enabled" : "Disabled"}
										role="switch"
										aria-checked={prompt.enabled}
									>
										<div
											className={`w-3 h-3 bg-white rounded-full shadow-sm transition-transform ${prompt.enabled ? "translate-x-4" : "translate-x-0"}`}
										/>
									</button>
								)}
								{onDelete && (
									<button
										type="button"
										onClick={(e) => onDelete(e, prompt.id)}
										className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-100 hover:text-red-500 dark:hover:bg-red-900/30 rounded transition-all ml-1"
									>
										<Trash2 size={14} />
									</button>
								)}
							</div>
						</div>
						<div className="text-[10px] font-mono text-slate-400 mb-1 truncate">
							{prompt.filename}
						</div>
						<p
							className={`text-xs line-clamp-1 leading-relaxed ${prompt.text ? "text-slate-600 dark:text-slate-400" : "text-slate-400/50 italic"}`}
						>
							{prompt.text || "(No text content)"}
						</p>
					</div>
				</div>
			</div>
		);
	},
);
