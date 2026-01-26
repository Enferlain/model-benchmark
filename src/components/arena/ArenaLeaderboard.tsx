import React from 'react';
import { Trophy, Medal, MoreHorizontal } from 'lucide-react';

export function ArenaLeaderboard() {
  const dummyData = [
    { rank: 1, name: "Flux 1.0 Pro", elo: 1250, battles: 450, winRate: "68%" },
    { rank: 2, name: "Midjourney v6", elo: 1245, battles: 890, winRate: "65%" },
    { rank: 3, name: "SDXL 1.0 Base", elo: 1100, battles: 1200, winRate: "45%" },
    { rank: 4, name: "Animagine XL 3.1", elo: 1080, battles: 340, winRate: "52%" },
    { rank: 5, name: "Playground v2", elo: 1050, battles: 210, winRate: "48%" },
  ];

  return (
    <div className="max-w-5xl mx-auto py-8">
      <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-sm border border-slate-200 dark:border-slate-700 overflow-hidden">
        <div className="p-6 border-b border-slate-200 dark:border-slate-700 bg-slate-50/50 dark:bg-slate-800/50">
           <h3 className="text-xl font-bold flex items-center gap-2 text-slate-800 dark:text-slate-100">
             <Trophy className="text-yellow-500" size={24} />
             Leaderboard
           </h3>
           <p className="text-slate-500 text-sm mt-1">
             Global rankings based on blind arena battles.
           </p>
        </div>

        <table className="w-full text-left border-collapse">
            <thead>
                <tr className="text-xs font-bold text-slate-400 uppercase tracking-wider border-b border-slate-100 dark:border-slate-700">
                    <th className="p-4 pl-6 w-20">Rank</th>
                    <th className="p-4">Model</th>
                    <th className="p-4 text-right">ELO Score</th>
                    <th className="p-4 text-right">Win Rate</th>
                    <th className="p-4 text-right pr-6">Battles</th>
                </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-700">
                {dummyData.map((model) => (
                    <tr key={model.rank} className="group hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors">
                        <td className="p-4 pl-6">
                            {model.rank === 1 && <Medal className="text-yellow-400" size={24} />}
                            {model.rank === 2 && <Medal className="text-slate-300" size={24} />}
                            {model.rank === 3 && <Medal className="text-amber-600" size={24} />}
                            {model.rank > 3 && <span className="text-slate-500 font-mono text-lg font-bold">#{model.rank}</span>}
                        </td>
                        <td className="p-4">
                            <span className="font-bold text-slate-700 dark:text-slate-200 group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors">
                                {model.name}
                            </span>
                        </td>
                        <td className="p-4 text-right font-mono text-slate-600 dark:text-slate-400 font-medium">
                            {model.elo}
                        </td>
                         <td className="p-4 text-right font-mono text-green-600 dark:text-green-400">
                            {model.winRate}
                        </td>
                        <td className="p-4 text-right pr-6 text-slate-400">
                            {model.battles}
                        </td>
                    </tr>
                ))}
            </tbody>
        </table>
      </div>
    </div>
  );
}
