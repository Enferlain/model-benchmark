import { useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { GripVertical } from "lucide-react";
import type React from "react";
import { forwardRef } from "react";
import type { ModelData } from "../../types";
import { stringToColor } from "../../utils/colorUtils";

export interface ListItemProps {
	model: ModelData;
	isSelected?: boolean;
	onClick?: () => void;
	isDragging?: boolean;
	dragHandleProps?: any;
	style?: React.CSSProperties;
	key?: React.Key;
}

// 1. Pure Visual Component (No dnd hooks)
// We use forwardRef to allow dnd-kit to attach to the DOM element
export const ListItem = forwardRef<HTMLDivElement, ListItemProps>(
	({ model, isSelected, onClick, isDragging, dragHandleProps, style }, ref) => {
		return (
			<div
				ref={ref}
				style={style}
				className={`
        group relative flex items-center gap-3 p-3 rounded-xl border transition-colors duration-200 select-none overflow-hidden max-w-full
        ${
					isSelected
						? "bg-blue-50/80 border-blue-200 dark:bg-blue-500/20 dark:border-blue-500/30"
						: "bg-white border-slate-200 hover:border-slate-300 dark:bg-slate-800 dark:border-slate-700 dark:hover:border-slate-600"
				}
        ${
					isDragging
						? "shadow-2xl scale-105 z-50 cursor-grabbing bg-white/95 dark:bg-slate-800/95 ring-2 ring-blue-500/50"
						: "hover:shadow-sm cursor-grab"
				}
      `}
				onClick={onClick}
			>
				{/* Drag Handle */}
				<div
					{...dragHandleProps}
					className="text-slate-400 hover:text-slate-600 dark:text-slate-500 dark:hover:text-slate-300 cursor-grab active:cursor-grabbing p-1"
				>
					<GripVertical size={16} />
				</div>

				{/* Color Dot */}
				<div
					className="w-2.5 h-2.5 rounded-full shrink-0"
					style={{ backgroundColor: stringToColor(model.hash || model.id) }}
				/>

				{/* Content */}
				<div className="flex-1 min-w-0 flex flex-col">
					<div className="flex items-center gap-2">
						<span className="font-medium text-sm text-slate-700 dark:text-slate-200 truncate">
							{model.name}
						</span>
						{model.is_missing && (
							<span className="shrink-0 px-1 py-0.5 bg-red-100 text-red-600 text-[9px] rounded font-bold uppercase tracking-wide">
								Missing
							</span>
						)}
					</div>

					<div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
						<span className="truncate">{model.model_type}</span>
						<span>•</span>
						<span className="truncate">{model.source}</span>
					</div>
				</div>
			</div>
		);
	},
);

ListItem.displayName = "ListItem";

// 2. Sortable Logic Wrapper
// This is what goes in the list. It manages the dnd state and passes it to the Visual Component.
export function SortableListItem(props: ListItemProps) {
	const {
		attributes,
		listeners,
		setNodeRef,
		transform,
		transition,
		isDragging,
	} = useSortable({ id: props.model.id });

	const style = {
		transform: CSS.Transform.toString(transform),
		transition,
		opacity: isDragging ? 0.05 : 1, // Almost invisible placeholder while dragging
	};

	return (
		<ListItem
			ref={setNodeRef}
			style={style}
			{...props}
			isDragging={isDragging}
			dragHandleProps={{ ...attributes, ...listeners }}
		/>
	);
}
