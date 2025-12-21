import React, { useState, useMemo, useRef } from 'react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  Label,
  ReferenceLine
} from 'recharts';
import { Maximize2, Minimize2 } from 'lucide-react';
import { ModelData, MetricKey, MetricOption } from '../types';

interface ScatterPlotProps {
  data: ModelData[];
  xMetric: MetricOption;
  yMetric: MetricOption;
  isDarkMode?: boolean;
}

const CustomTooltip = ({ active, payload, isDarkMode }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload as ModelData;
    
    if (!data || !data.name) return null;

    return (
      <div className={`p-4 border shadow-2xl rounded-2xl text-sm z-50 backdrop-blur-xl ${isDarkMode ? 'bg-slate-800/80 border-white/10 text-slate-200' : 'bg-white/80 border-white/50 text-slate-800'}`}>
        <p className={`font-bold mb-1 ${isDarkMode ? 'text-slate-100' : 'text-slate-800'}`}>{data.name}</p>
        <p className={`text-xs mb-3 ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>{data.source}</p>
        <div className="space-y-1.5 font-mono text-xs">
          <p className={`${isDarkMode ? 'text-slate-400' : 'text-slate-600'}`}>
            <span className="font-semibold text-blue-500 dark:text-blue-400">X:</span> {payload[0].value.toFixed(2)}
          </p>
          <p className={`${isDarkMode ? 'text-slate-400' : 'text-slate-600'}`}>
            <span className="font-semibold text-pink-500 dark:text-pink-400">Y:</span> {payload[1].value.toFixed(2)}
          </p>
        </div>
      </div>
    );
  }
  return null;
};

// Distance calculation helper
const calculateDistance = (p1: ModelData, p2: ModelData, xKey: string, yKey: string) => {
    const dx = (p1[xKey as keyof ModelData] as number) - (p2[xKey as keyof ModelData] as number);
    const dy = (p1[yKey as keyof ModelData] as number) - (p2[yKey as keyof ModelData] as number);
    return Math.sqrt(dx * dx + dy * dy).toFixed(4);
}

export const ScatterPlot: React.FC<ScatterPlotProps> = ({ data, xMetric, yMetric, isDarkMode = false }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const clickProcessedRef = useRef(false);

  const selectedNode = useMemo(() => 
    data.find(n => n.id === selectedNodeId), 
  [data, selectedNodeId]);

  // Define colors based on mode
  const gridColor = isDarkMode ? '#475569' : '#cbd5e1'; 
  const axisColor = isDarkMode ? '#94a3b8' : '#64748b'; 
  const axisLineColor = isDarkMode ? '#334155' : '#cbd5e1';
  const textColor = isDarkMode ? '#e2e8f0' : '#475569';
  const linkColor = isDarkMode ? '#94a3b8' : '#64748b';
  
  // Helper to cast for indexing
  const getX = (item: ModelData) => item[xMetric.value as MetricKey] as number;
  const getY = (item: ModelData) => item[yMetric.value as MetricKey] as number;

  // Handler for clicking a specific point
  const handlePointClick = (node: any) => {
    clickProcessedRef.current = true;
    const id = node.payload?.id || node.id;
    if (id) {
      setSelectedNodeId(prev => prev === id ? null : id);
    }
    setTimeout(() => {
      clickProcessedRef.current = false;
    }, 100);
  };

  // Handler for clicking the chart background
  const handleChartClick = (e: any) => {
    if (clickProcessedRef.current) return;
    if (!e || !e.activePayload || e.activePayload.length === 0) {
      setSelectedNodeId(null);
    }
  };

  return (
    <div 
      className={`rounded-3xl shadow-xl shadow-slate-200/50 dark:shadow-black/20 border flex flex-col transition-all duration-300 overflow-hidden backdrop-blur-xl ${
        isDarkMode ? 'bg-slate-800/40 border-white/5' : 'bg-white/60 border-white/60'
      } ${
        isExpanded 
          ? 'fixed inset-0 z-50 p-6 m-0 h-screen w-screen rounded-none bg-slate-100/90 dark:bg-slate-900/90' 
          : 'w-full h-[650px] p-4 relative'
      }`}
    >
      <style>{`
        .recharts-wrapper, .recharts-surface { outline: none !important; }
        :focus { outline: none !important; }
      `}</style>

      {/* Header / Controls */}
      <div className="flex justify-between items-center mb-2 h-8 shrink-0 px-2">
        <div className={`text-sm font-medium ${isDarkMode ? 'text-slate-200' : 'text-slate-700'}`}>
             {isExpanded ? `${yMetric.label} vs ${xMetric.label}` : ''}
        </div>
        <button 
           onClick={() => setIsExpanded(!isExpanded)}
           className={`p-2 rounded-full transition-colors absolute right-6 top-6 z-10 ${
             isDarkMode 
               ? 'text-slate-400 hover:text-slate-200 bg-white/5 hover:bg-white/10' 
               : 'text-slate-400 hover:text-slate-700 bg-black/5 hover:bg-black/10'
           }`}
           title={isExpanded ? "Exit Full Screen" : "Full Screen"}
        >
           {isExpanded ? <Minimize2 size={18} /> : <Maximize2 size={18} />}
        </button>
      </div>

      {/* Chart Area */}
      <div className="flex-1 min-h-0 w-full select-none">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart
            margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
            onClick={handleChartClick}
          >
            <CartesianGrid stroke={gridColor} strokeOpacity={isDarkMode ? 0.2 : 0.4} strokeDasharray="3 3" />
            
            <XAxis 
              type="number" 
              dataKey={xMetric.value} 
              name={xMetric.label} 
              stroke={axisColor}
              fontSize={12}
              tickLine={false}
              axisLine={{ stroke: axisLineColor, strokeOpacity: 0.5 }}
              tick={{ fill: textColor }}
              domain={['auto', 'auto']}
            >
               <Label value={xMetric.label} offset={-10} position="insideBottom" style={{ fill: textColor, fontSize: '12px', fontWeight: 500, opacity: 0.8 }} />
            </XAxis>
            <YAxis 
              type="number" 
              dataKey={yMetric.value} 
              name={yMetric.label} 
              stroke={axisColor}
              fontSize={12}
              tickLine={false}
              axisLine={{ stroke: axisLineColor, strokeOpacity: 0.5 }}
              tick={{ fill: textColor }}
              domain={['auto', 'auto']}
            >
              <Label value={yMetric.label} angle={-90} position="insideLeft" style={{ fill: textColor, fontSize: '12px', fontWeight: 500, opacity: 0.8 }} />
            </YAxis>
            
            <Tooltip 
              content={<CustomTooltip isDarkMode={isDarkMode} />} 
              cursor={{ strokeDasharray: '3 3', stroke: isDarkMode ? '#94a3b8' : '#64748b', strokeOpacity: 0.5 }} 
              isAnimationActive={false}
            />
            
            {selectedNode && data.map((model) => {
               if (model.id === selectedNode.id) return null;
               
               const dist = calculateDistance(selectedNode, model, xMetric.value, yMetric.value);
               
               return (
                 <ReferenceLine
                   key={`link-${selectedNode.id}-${model.id}`}
                   segment={[
                     { x: getX(selectedNode), y: getY(selectedNode) },
                     { x: getX(model), y: getY(model) }
                   ]}
                   stroke={linkColor}
                   strokeOpacity={0.3}
                   strokeDasharray="3 3"
                   label={{ 
                       value: `${dist}`, 
                       position: 'center', 
                       fill: textColor, 
                       fontSize: 10,
                   }}
                 />
               );
            })}

            <Scatter 
                name="Models" 
                data={data}
                cursor="pointer"
                animationDuration={500}
                onClick={handlePointClick}
            >
              {data.map((entry, index) => (
                <Cell 
                  key={`cell-${index}`} 
                  fill={
                    entry.source === 'Civitai' 
                      ? (isDarkMode ? '#60A5FA' : '#3b82f6') 
                      : (isDarkMode ? '#F472B6' : '#ec4899')
                  } 
                  fillOpacity={selectedNodeId && selectedNodeId !== entry.id ? 0.4 : 0.9}
                  stroke={isDarkMode ? '#1e293b' : '#fff'}
                  strokeWidth={2}
                />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Legend */}
      <div className={`flex justify-center mt-4 gap-6 text-xs shrink-0 h-6 ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
        <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${isDarkMode ? 'bg-blue-400' : 'bg-blue-500'}`}></div>
            <span>Civitai</span>
        </div>
        <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${isDarkMode ? 'bg-pink-400' : 'bg-pink-500'}`}></div>
            <span>HuggingFace</span>
        </div>
      </div>
    </div>
  );
};