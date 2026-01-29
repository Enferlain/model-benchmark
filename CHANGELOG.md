# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-01-29

### Added

- **Database Schema Evolution (Alembic)**: Integrated Alembic for robust, versioned database migrations.
- **Arena Infrastructure**: Added new `arenavote` and `prompt` tables to support detailed battle tracking and stable prompt entities.
- **Bradley-Terry Support**: Introduced `bt_score` to the `Model` table, laying the groundwork for the new multi-dimensional ranking system.
- **Image Indexing System**: Automated extraction of 8-character UUIDs from PNG metadata into the `ImageOutput` table for high-performance gallery lookups.
- **BT Calculation Engine**: Implemented a statistically grounded Bradley-Terry ranking service (`bt_service.py`) using the MM algorithm to derive model strengths from arena votes.
- **Arena Voting API**: New `/arena/vote` and `/arena/leaderboard` endpoints with support for ties and stable prompt entity mapping.

### Changed

- **Industry-Standard Hashing (SHA256)**: Transitioned model hashing from BLAKE3 to SHA256 to align with HuggingFace, ComfyUI, and other common AI tooling.
- **API Data Model**: Updated the backend API to expose `bt_score` and `hash_type`.
- **Database Architecture**: Discontinued automatic SQLModel table creation in favor of explicit Alembic migrations for better stability and traceability.

## [Unreleased] - 2026-01-28

### Fixed

- **Arena Alignment Precision**:
  - Implemented dynamic pixel-width calculation for alignment units based on image aspect ratio and container height.
  - Achieved perfect **0px horizontal offset** between Vote buttons and image centers across all aspect ratios.
  - Standardized the center gap to a consistent **~17.5px** (within 1.5px tolerance) for Square, Landscape, and Portrait modes.
  - Fixed "floating" images in Square and Portrait modes by constraining units to exact image widths and pulling them flush against the divider.
- **Arena Animation Synchronization**:
  - Synchronized the reference image container's border and background transitions with its 500ms scaling animation.
  - Replaced immediate layout toggles with smooth transparency transitions (`border-transparent`, `bg-transparent`) to eliminate visual "jumps" during expansion and contraction.
- **Arena Layout Consistency**:
  - Fixed an issue where portrait (narrow) images would drift apart when the reference was expanded.
  - Switched from distributive `flex-1` to calculated `flex-none` widths, ensuring images always "hug" the center divider regardless of aspect ratio.
- **Project-wide Quality & Accessibility**:
  - Resolved 100+ Biome linting errors across the codebase, primarily focusing on missing `type="button"` attributes and accessibility violations.
  - Converted interactive static elements (`div`, `img`) to semantic `<button>` elements for better screen reader support and keyboard navigation.
  - Fixed broken accessibility associations (labels without inputs) in `TransferList` and `ScanSettingsPanel`.
  - Achieved clean `tsc` type-checking and high-performance frontend verification.

## [Unreleased] - 2026-01-27

### Fixed

- **Drag and Drop Precision**: Significantly improved model transfer UX in the Dashboard.
  - Switched to `pointerWithin` collision detection to prevent unintentional "auto-dropping" to nearby fields.
  - Tied drop activation to valid targets only; panels no longer show "active" highlights when dragging over self.
  - Expanded droppable hit area to the entire panel for immediate visual feedback.
- **Compare Tab Quality**: Filtered "Common Prompt" list to only include prompts with at least one common seed across all models, preventing empty comparison views.
- **Reference Errors**: Fixed `headerExtra` ReferenceError in `ScatterPlot` by correctly destructuring props.
- **Chart Visibility**: Resolved issue where chart content would disappear during active dataset searches.
- **Full-Screen Positioning**: Fixed portal-based full-screen view positioning to correctly ignore parent container transforms.
- **Type Safety**: Fixed several type errors and resolved implicit `any` usage in `App.tsx`, `Dashboard.tsx`, `Gallery.tsx`, and `PromptEditor.tsx`.
- **Environment Handling**: Fixed missing `import.meta.env` types for Vite environment variables.
- **Backend Logic**: Fixed critical "Undefined name" and unreachable code bugs in `prompt_manager.py`.
- **Arena Layout Overhaul**:
  - Implemented **"Clustered Layout"** to eliminate side-spacing and group images tightly in the center.
  - **Zero-Bar Fitting**: Standardized container logic to shrink-wrap all images (Portrait, Landscape, Square) with `h-fit` and `min-w-0`, enabling 100% pixel coverage without padding bars.
  - **1:1:1 Fairness**: Restored strict triple-flex distribution to ensure Model A, Reference, and Model B are identically sized when expanded.
  - **Symmetrical Animations**: Synchronized expanding/shrinking transitions with a shared 500ms ease-in-out curve and cross-fading opacity for a polished feel.
  - **State Stability**: Added re-mount logic to prevent "stuck" aspect ratios when switching between samples.

### Added

- **Unified Data Caching**: Implemented a global `DataContext` to centralize model, prompt, and benchmark run data.
  - Eliminated redundant API calls when navigating between pages.
  - Provided instant page transitions by serving data from a centralized cache.
- **Global State Persistence**: Enforced `localStorage` persistence for all major UI states (selected models, active tabs, filters, and search queries) across the entire application.
- **Robust Error Handling**: Added explicit error tracking to the global data context to provide better feedback during fetch failures.
- **Repository Tooling**: Integrated **Biome** (frontend) and **Ruff** (backend) for high-performance linting/formatting across the entire monorepo.
- **Backend Infrastructure**: Introduced `pyproject.toml` to standardise backend metadata, dependencies, and tool configurations.
- **Type Checking**: Enabled project-wide type verification using `tsc` for the frontend and **Ty** for the backend.
- **Quality Control Scripts**: Added `npm run check:backend` and `npm run check:all` to standardise development workflow and allow holistic repository verification.
- **Dataset Selector**: Implemented a unified data source switcher for the primary dashboard chart.
  - Support for **All Models**, **Current Queue**, **Past Benchmark Runs**, and **Saved Presets**.
  - Integrated search bar to filter both the data source menu and the chart points simultaneously.
  - Real-time filtering of chart points by model name, ID, and type.
- **Chart Header**: Enhanced chart header with active source indicators and a settings menu for dataset selection.
- **UI Architecture**: Implemented React Portals for the expanded chart view to avoid clipping and layout issues.
- **Visual Polish**:
  - Restored rich metric descriptions and "higher is better" indicators.
  - Reorganized dashboard into a cleaner vertical stack layout.
  - Implemented glassmorphism styling for dashboard cards with optimized blur and borders.

### Changed

- **UI Architecture Migration**: Refactored all application pages (`Dashboard`, `Database`, `PromptEditor`, `Arena`, `Analytics`, `Compare`, `Gallery`) to consume the centralized `DataContext`.
- **Context Consolidation**: Deprecated and merged redundant `GalleryContext` into the unified `DataContext`.

---

## [Unreleased] - 2026-01-26

### Added

- **Transfer List Interface**: Implemented a comprehensive drag-and-drop interface for model selection using `dnd-kit`.
  - **Library Panel**: Searchable list of all available models with "Filter" capabilities.
  - **Queue Panel**: Manage selected models for benchmarking, including a "Preset" system to save/load queue configurations.
  - **Visual Feedback**: Added drag highlights, dashed drop zones, and active state indicators.
- **Analytics Page**: Migrated the detailed metrics table to a dedicated `/analytics` route to declutter the main dashboard.
- **Preset System**: Added local storage-based persistence for Benchmark Queues, allowing users to save sets of models.
- **Filters**: Added advanced filtering to the Library panel for:
  - Model Type (e.g., SDXL, SD1.5)
  - Prediction Type (Epsilon, V-Prediction, Zero-SNR)
  - Source (Civitai, HuggingFace, Local)

### Fixed

- **Drag Interaction**: Resolved "teleporting" bug by separating visual `ListItem` from logic-bearing `SortableListItem` and using Portals for the drag overlay.
- **Scroll Issues**: Fixed horizontal scrollbars appearing during drag operations by enforcing `overflow-x-hidden`.
- **Backend Integration**: Updated generation endpoints to respect the `selected_model_ids` list, allowing for partial benchmark runs.
- **Layout**: Optimized search bar layout to include actionable tool buttons (Filter/Preset) inline.

---

## [unreleased] - 2026-01

### Added

- **Model Persistence**: Models are no longer deleted from the database when files are missing. They are marked as "Offline".
- **Database**: Added `is_missing` column to `Model` table with auto-migration.
- **Metadata**: Added `model_hash` to generated PNG metadata for robust tracking.
- **UI (Database)**: Added Search Bar to filter models.
- **UI (Database)**: Added "OFFLINE" badge for missing models.
- **Model Library Manager**: Added "Import Local Path" feature to Dashboard.
- **Native Dialogs**: Integrated native file/folder pickers for easier model importing.
- **Multi-Select**: Support for importing multiple files at once.
- **Persistence**: Imported local paths are saved to `sources.json` and persist across restarts.
- **Recursive Scan**: Importing a folder now recursively finds all `.safetensors` models.

### Changed

- **Model Manager**: Refactored `scan_models` to perform Soft Deletes instead of Hard Deletes.
- **Generation**: Backend now skips offline models during batch generation.
- **Dashboard**: Offline models are hidden from the generator selector to prevent errors.
- **Sorting**: Offline models are now automatically sorted to the bottom of the Database list.
- **Batch Optimization**: Importing huge model collections now triggers only a single database sync (orders of magnitude faster).
- **Feedback**: Import messages now distinguish between New, Updated, and Unchanged models.

### Added

- **Database Management**: Added "Actions" menu (vertical `...`) to `Database.tsx`.
- **Safe Removal**: Added specific "Remove Model" action that cleans the DB without touching files.
- **Safety**: Removed "Delete File" checkbox to prevent accidental data loss.

### Changed

- Use frontmatter title & description in each language version template
- Replace broken OpenGraph image with an appropriately-sized Keep a Changelog
  image that will render properly (although in English for all languages)
- Fix OpenGraph title & description for all languages so the title and
  description when links are shared are language-appropriate

### Removed
