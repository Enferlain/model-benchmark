import React from 'react';
import { Save, Image as ImageIcon, FileText, Loader2, Type } from 'lucide-react';
import { PromptData } from '../../types';

interface PromptDetailEditorProps {
    prompt: PromptData | undefined;
    editText: string;
    editAlias: string;
    isSaving: boolean;
    isDirty: boolean;
    onTextChange: (text: string) => void;
    onAliasChange: (text: string) => void;
    onSave: () => void;
}

export const PromptDetailEditor: React.FC<PromptDetailEditorProps> = ({
    prompt,
    editText,
    editAlias,
    isSaving,
    isDirty,
    onTextChange,
    onAliasChange,
    onSave
}) => {
    return (
        <div className="flex-1 bg-white dark:bg-slate-800/50 rounded-2xl shadow-lg border border-slate-200 dark:border-white/5 overflow-hidden flex flex-col backdrop-blur-sm relative">
        {prompt ? (
          <>
            <div className="p-4 border-b border-slate-200 dark:border-white/5 flex justify-between items-center bg-slate-50/50 dark:bg-slate-900/20">
              <div className="flex items-center gap-3">
                 {prompt.image ? <ImageIcon size={18} className="text-blue-500"/> : <FileText size={18} className="text-slate-500"/>}
                 <span className="font-mono text-sm text-slate-500">{prompt.filename}</span>
              </div>
              <button
                onClick={onSave}
                disabled={!isDirty || isSaving}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${
                  isDirty
                    ? 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-500/20'
                    : 'bg-slate-100 dark:bg-white/5 text-slate-400 cursor-not-allowed'
                }`}
              >
                {isSaving ? <Loader2 size={16} className="animate-spin" /> : <Save size={16} />}
                Save Changes
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-6">
               {prompt.image && (
                 <div className="shrink-0">
                   <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">Reference Image</h3>
                   <div className="rounded-xl overflow-hidden border border-slate-200 dark:border-white/10 shadow-lg inline-block md:max-w-md lg:max-w-lg bg-black/5 dark:bg-black/20">
                     <img src={prompt.image} alt="Reference" className="max-h-[400px] w-auto object-contain" />
                   </div>
                 </div>
               )}

               {/* Alias Input */}
               <div>
                  <label htmlFor="prompt-alias" className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-2">
                     <Type size={14} /> Alias / Nickname
                  </label>
                  <input
                    id="prompt-alias"
                    type="text"
                    value={editAlias}
                    onChange={e => onAliasChange(e.target.value)}
                    className="w-full bg-slate-50 dark:bg-slate-900/50 border border-slate-200 dark:border-white/10 rounded-xl px-4 py-3 font-medium text-sm text-slate-800 dark:text-slate-200 focus:ring-2 focus:ring-blue-500 outline-none placeholder:text-slate-400"
                    placeholder="E.g. 'Red Car' (Used for display and filenames)"
                  />
                  <p className="text-[10px] text-slate-400 mt-2 ml-1">
                      This alias will be used in the UI and can be used for output filenames instead of the raw ID.
                  </p>
               </div>

               <div className="flex-1 flex flex-col min-h-[200px]">
                 <label htmlFor="prompt-text" className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">Prompt Text</label>
                 <textarea
                   id="prompt-text"
                   value={editText}
                   onChange={e => onTextChange(e.target.value)}
                   className="flex-1 w-full bg-slate-50 dark:bg-slate-900/50 border border-slate-200 dark:border-white/10 rounded-xl p-4 font-mono text-sm leading-relaxed text-slate-800 dark:text-slate-200 focus:ring-2 focus:ring-blue-500 outline-none resize-none"
                   placeholder="Enter prompt text here..."
                 />
               </div>
            </div>
          </>
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-slate-400">
             <div className="w-16 h-16 rounded-2xl bg-slate-100 dark:bg-white/5 flex items-center justify-center mb-4">
               <FileText size={32} className="opacity-50"/>
             </div>
             <p>Select a prompt to edit</p>
          </div>
        )}
        </div>
    );
}
