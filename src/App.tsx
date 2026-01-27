import { BrowserRouter, Route, Routes } from "react-router-dom";
import { DataProvider } from "./context/DataContext";
import { ThemeProvider } from "./context/ThemeContext";
import { MainLayout } from "./layouts/MainLayout";
import Analytics from "./pages/Analytics";
import Arena from "./pages/Arena";
import Compare from "./pages/Compare";
import Dashboard from "./pages/Dashboard";
import Database from "./pages/Database";
import Gallery from "./pages/Gallery";
import PromptEditor from "./pages/PromptEditor";

export default function App() {
	return (
		<ThemeProvider>
			<DataProvider>
				<BrowserRouter>
					<Routes>
						<Route path="/" element={<MainLayout />}>
							<Route index element={<Dashboard />} />
							<Route path="gallery" element={<Gallery />} />
							<Route path="prompts" element={<PromptEditor />} />
							<Route path="compare" element={<Compare />} />
							<Route path="analytics" element={<Analytics />} />
							<Route path="arena" element={<Arena />} />
							<Route path="database" element={<Database />} />
						</Route>
					</Routes>
				</BrowserRouter>
			</DataProvider>
		</ThemeProvider>
	);
}
