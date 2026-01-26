# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

## [0.1.0] - 2026-01

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
