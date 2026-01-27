import { Info } from "lucide-react";
import type { MetricOption } from "../../types";

interface MetricInfoCardProps {
	xMetric: MetricOption;
	yMetric: MetricOption;
}

export function MetricInfoCard({ xMetric, yMetric }: MetricInfoCardProps) {
	return (
		<div className="p-5 rounded-3xl border border-blue-100/50 dark:border-blue-500/10 bg-blue-50/50 dark:bg-blue-500/5 backdrop-blur-md">
			<div className="flex gap-3">
				<Info
					className="text-blue-500 dark:text-blue-400 shrink-0 mt-0.5"
					size={18}
				/>
				<div className="space-y-3">
					<p className="text-sm text-blue-900 dark:text-blue-100 font-medium">
						Metric Info
					</p>

					<div className="space-y-2">
						<div className="text-xs text-blue-800/80 dark:text-blue-200/80">
							<span className="font-semibold">{xMetric.label}:</span>{" "}
							{xMetric.description}
							{xMetric.direction && (
								<span className="ml-1 opacity-75">
									(
									{xMetric.direction === "higher"
										? "Higher is better ⬆️"
										: "Lower is better ⬇️"}
									)
								</span>
							)}
						</div>

						<div className="text-xs text-blue-800/80 dark:text-blue-200/80">
							<span className="font-semibold">{yMetric.label}:</span>{" "}
							{yMetric.description}
							{yMetric.direction && (
								<span className="ml-1 opacity-75">
									(
									{yMetric.direction === "higher"
										? "Higher is better ⬆️"
										: "Lower is better ⬇️"}
									)
								</span>
							)}
						</div>
					</div>

					{/* Extended Info button or subtle indicator if needed could go here, 
					    but the user specifically complained about the "style", 
					    and the above matches the most recent "good" version. */}
				</div>
			</div>
		</div>
	);
}
