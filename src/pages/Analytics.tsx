import { BarChart3 } from "lucide-react";
import { useEffect, useState } from "react";
import { ModelTable } from "../components/ModelTable";
import { useData } from "../context/DataContext";
import { deleteModel } from "../services/api";

export default function Analytics() {
	const { models, refreshModels: fetchModels } = useData();
	const [selectedId, setSelectedId] = useState<string | null>(() => {
		return localStorage.getItem("analytics_selectedId");
	});

	useEffect(() => {
		if (selectedId) {
			localStorage.setItem("analytics_selectedId", selectedId);
		} else {
			localStorage.removeItem("analytics_selectedId");
		}
	}, [selectedId]);

	const handleDeleteModel = async (id: string, deleteFile: boolean) => {
		try {
			await deleteModel(id, deleteFile);
			if (selectedId === id) setSelectedId(null);
			fetchModels();
		} catch (error) {
			console.error("Error deleting model:", error);
			// Ideally show a toast here
		}
	};

	return (
		<div className="max-w-[1800px] mx-auto px-6 py-8">
			<div className="mb-8">
				<h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-slate-900 to-slate-600 dark:from-white dark:to-slate-400 flex items-center gap-3">
					<BarChart3 className="text-blue-500" />
					Benchmark Analytics
				</h1>
				<p className="text-slate-500 dark:text-slate-400 mt-2">
					Detailed performance metrics and management for {models.length}{" "}
					models.
				</p>
			</div>

			<div className="space-y-6">
				<ModelTable
					models={models}
					onDelete={handleDeleteModel}
					selectedId={selectedId}
					onSelect={setSelectedId}
				/>
			</div>
		</div>
	);
}
