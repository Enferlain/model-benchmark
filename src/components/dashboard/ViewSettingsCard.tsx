import { METRIC_OPTIONS } from "../../constants";
import type { MetricKey } from "../../types";

interface ViewSettingsCardProps {
	xMetricKey: MetricKey;
	setXMetricKey: (val: MetricKey) => void;
	yMetricKey: MetricKey;
	setYMetricKey: (val: MetricKey) => void;
}

export function ViewSettingsCard({
	xMetricKey,
	setXMetricKey,
	yMetricKey,
	setYMetricKey,
}: ViewSettingsCardProps) {
	return (
		<div className="p-6 rounded-3xl shadow-xl shadow-slate-200/50 dark:shadow-black/20 border border-white/60 dark:border-white/5 bg-white/90 dark:bg-slate-800/80 backdrop-blur-md transition-shadow hover:shadow-2xl">
			<h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-4">
				View Settings
			</h2>
			<div className="space-y-4">
				<div>
					<label
						htmlFor="x-metric-select"
						className="block text-[10px] font-bold text-slate-500 dark:text-slate-400 mb-2 ml-1 opacity-80"
					>
						X-AXIS METRIC
					</label>
					<div className="relative">
						<select
							id="x-metric-select"
							value={xMetricKey}
							onChange={(e) => setXMetricKey(e.target.value as MetricKey)}
							className="w-full px-4 py-3 bg-white/50 dark:bg-black/20 border border-slate-200/60 dark:border-white/5 rounded-xl text-sm appearance-none focus:outline-none focus:ring-2 focus:ring-blue-500/30 dark:focus:ring-blue-400/20 text-slate-800 dark:text-slate-200 cursor-pointer backdrop-blur-sm"
						>
							{METRIC_OPTIONS.map((opt) => (
								<option key={opt.value} value={opt.value}>
									{opt.label}
								</option>
							))}
						</select>
						<div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none opacity-40">
							<svg
								width="10"
								height="6"
								viewBox="0 0 10 6"
								fill="none"
								xmlns="http://www.w3.org/2000/svg"
								aria-hidden="true"
							>
								<title>Dropdown Arrow</title>
								<path
									d="M1 1L5 5L9 1"
									stroke="currentColor"
									strokeWidth="1.5"
									strokeLinecap="round"
									strokeLinejoin="round"
								/>
							</svg>
						</div>
					</div>
				</div>

				<div>
					<label
						htmlFor="y-metric-select"
						className="block text-[10px] font-bold text-slate-500 dark:text-slate-400 mb-2 ml-1 opacity-80"
					>
						Y-AXIS METRIC
					</label>
					<div className="relative">
						<select
							id="y-metric-select"
							value={yMetricKey}
							onChange={(e) => setYMetricKey(e.target.value as MetricKey)}
							className="w-full px-4 py-3 bg-white/50 dark:bg-black/20 border border-slate-200/60 dark:border-white/5 rounded-xl text-sm appearance-none focus:outline-none focus:ring-2 focus:ring-blue-500/30 dark:focus:ring-blue-400/20 text-slate-800 dark:text-slate-200 cursor-pointer backdrop-blur-sm"
						>
							{METRIC_OPTIONS.map((opt) => (
								<option key={opt.value} value={opt.value}>
									{opt.label}
								</option>
							))}
						</select>
						<div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none opacity-40">
							<svg
								width="10"
								height="6"
								viewBox="0 0 10 6"
								fill="none"
								xmlns="http://www.w3.org/2000/svg"
								aria-hidden="true"
							>
								<title>Dropdown Arrow</title>
								<path
									d="M1 1L5 5L9 1"
									stroke="currentColor"
									strokeWidth="1.5"
									strokeLinecap="round"
									strokeLinejoin="round"
								/>
							</svg>
						</div>
					</div>
				</div>
			</div>
		</div>
	);
}
