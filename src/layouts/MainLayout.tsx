import React from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import { Sparkles, Sun, Moon, LayoutDashboard, Images, Edit3, GitCompare, Trophy, Loader2 } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';

interface MainLayoutProps {
  isLoading?: boolean;
  modelCount?: number;
}

export const MainLayout: React.FC<MainLayoutProps> = ({ isLoading = false, modelCount = 0 }) => {
  const { isDarkMode, toggleTheme } = useTheme();

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-100 via-blue-50 to-white dark:from-slate-900 dark:via-[#1e293b] dark:to-slate-900 font-sans text-slate-900 dark:text-slate-100 pb-10 transition-all duration-500">
      {/* Glass Header */}
      <header className="sticky top-0 z-30 transition-all duration-300 border-b border-white/20 dark:border-white/5 bg-white/70 dark:bg-slate-900/60 backdrop-blur-xl supports-[backdrop-filter]:bg-white/60">
        <div className="max-w-[1800px] mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-blue-500/10 dark:bg-blue-400/10 text-blue-600 dark:text-blue-400 backdrop-blur-sm">
              <Sparkles size={20} />
            </div>
            <div>
              <h1 className="text-xl font-semibold tracking-tight text-slate-800 dark:text-slate-100">
                Benchmark Explorer
              </h1>
            </div>

            {/* Navigation */}
            <nav className="ml-8 hidden md:flex items-center space-x-1">
              <NavLink to="/" className={({ isActive }) => `px-3 py-2 rounded-lg text-sm font-medium transition-colors ${isActive ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300' : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800'}`}>
                <span className="flex items-center gap-2"><LayoutDashboard size={16}/> Dashboard</span>
              </NavLink>
              <NavLink to="/gallery" className={({ isActive }) => `px-3 py-2 rounded-lg text-sm font-medium transition-colors ${isActive ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300' : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800'}`}>
                <span className="flex items-center gap-2"><Images size={16}/> Gallery</span>
              </NavLink>
              <NavLink to="/prompts" className={({ isActive }) => `px-3 py-2 rounded-lg text-sm font-medium transition-colors ${isActive ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300' : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800'}`}>
                <span className="flex items-center gap-2"><Edit3 size={16}/> Prompts</span>
              </NavLink>
              <NavLink to="/compare" className={({ isActive }) => `px-3 py-2 rounded-lg text-sm font-medium transition-colors ${isActive ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300' : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800'}`}>
                 <span className="flex items-center gap-2"><GitCompare size={16}/> Compare</span>
              </NavLink>
              <NavLink to="/arena" className={({ isActive }) => `px-3 py-2 rounded-lg text-sm font-medium transition-colors ${isActive ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300' : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800'}`}>
                 <span className="flex items-center gap-2"><Trophy size={16}/> Arena</span>
              </NavLink>
            </nav>
          </div>

          <div className="flex items-center gap-4">
            <div className="hidden sm:block">
              <span className="text-xs font-medium text-slate-600 dark:text-slate-300 bg-white/50 dark:bg-white/5 px-3 py-1.5 rounded-full border border-slate-200/50 dark:border-white/10 backdrop-blur-sm shadow-sm flex items-center gap-2">
                {isLoading ? (
                  <Loader2 size={12} className="animate-spin" />
                ) : (
                  modelCount
                )}{" "}
                Models tracked
              </span>
            </div>

            <button
              onClick={toggleTheme}
              className="p-2.5 text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-blue-200 bg-white/50 dark:bg-white/5 hover:bg-white dark:hover:bg-white/10 border border-transparent hover:border-slate-200 dark:hover:border-white/10 rounded-full transition-all duration-300 backdrop-blur-sm shadow-sm hover:shadow-md"
              title={
                isDarkMode ? "Switch to Light Mode" : "Switch to Dark Mode"
              }
            >
              {isDarkMode ? <Sun size={18} /> : <Moon size={18} />}
            </button>
          </div>
        </div>
      </header>

      <main>
        <Outlet />
      </main>
    </div>
  );
};
