# Proposal: Long-Term System Improvements

Based on the audit of the current data architecture and future requirements, I propose the following enhancements to ensure scalability and better analysis.

## 1. Schema Evolution & Migrations

> [!IMPORTANT]
> **Switch to Alembic**
> Currently, the system uses ad-hoc `ALTER TABLE` statements in `init_db`. As the schema grows, we should adopt **Alembic** to manage database migrations safely and predictably.

## 2. Arena: Bradley-Terry (BT) Ranking & Matchups

To provide deeper insights than a simple win/loss count, we will implement a **Bradley-Terry** model. This allows us to estimate the probability that one model is "better" than another based on all observed comparisons, even without direct matchups between every pair.

### Enhanced `arenavote` Schema

We need to track exactly what the user was looking at to identify "niche" model strengths (e.g., Model A is better at anime, Model B at realism).

| Field          | Type     | Purpose                                  |
| :------------- | :------- | :--------------------------------------- |
| `id`           | UUID/Int | Primary Key                              |
| `model_a_hash` | String   | Foreign Key to `model`                   |
| `model_b_hash` | String   | Foreign Key to `model`                   |
| `winner_hash`  | String?  | NULL if Tie or Both Bad                  |
| `vote_type`    | Enum     | `model_a`, `model_b`, `tie`, `both_bad`  |
| `prompt_id`    | UUID?    | Link to the specific prompt used         |
| `seed`         | Int      | Seed used for generation                 |
| `parameters`   | JSON     | Steps, CFG, Denoising context            |
| `created_at`   | DateTime | For tracking model improvement over time |

### How Bradley-Terry Works in the Arena

Unlike Elo (which updates a score after every match like a chess ranking), Bradley-Terry is a **statistical model** that looks at the entire history of matchups to estimate a model's latent "strength."

1.  **Strength Parameter ($\beta_i$):** Every model is assigned a numerical "strength" value.
2.  **Win Probability:** In a matchup between Model A and Model B, the probability of A winning is:
    $$P(A > B) = \frac{\exp(\beta_A)}{\exp(\beta_A) + \exp(\beta_B)}$$
3.  **Maximum Likelihood:** The backend runs a periodic calculation to find the $\beta$ values that most likely explain all the thousands of user votes we've collected.
4.  **The Result:** A leaderboard where the "Score" is derived from these strength values.

### Converting Beta to a Final Score

A raw $\beta$ value (like $1.2$ or $-0.5$) isn't very intuitive for users. We can convert it into several friendly formats:

- **Elo-Equivalent:** We can linearly map $\beta$ to the standard Elo scale (e.g., $1200 + \beta \times 400$). This gives users a familiar number.
- **Win Rate against Baseline:** We can pick a "Reference Model" (e.g., "SDXL Base") and set its $\beta$ to $0$. Every other model's score then becomes its predicted probability of winning against that reference (e.g., "Score: 68%").
- **Logits/Ordinal:** A simple 0-100 scale where 100 is the theoretically "perfect" model that wins every observed matchup.

## 3. Multi-Dimensional Arena Analytics

Because we track the `prompt_id`, `parameters`, and `metadata`, we don't just get one score—we get a map of model performance.

### Segmented Leaderboards

By running the Bradley-Terry calculation on filtered subsets of the data, we can generate:

- **Category Ranks**: "Top Models for _Anime_" vs "Top Models for _Architecture_".
- **Interaction Ranks**: "Best Model for _Ad-hoc Prompts_" (unseen) vs "Best Model for _Reference Fidelity_" (when a reference image is shown).
- **Parameter Sensitivity**: Scores segmented by Step count or CFG range.

### Performance Delta View

We can visualize how a model's performance changes when a Reference Image is introduced.

> [!TIP]
> **Example Insight**: "Model A has a 70% win rate in blind tests, but drops to 40% when a Reference Image is provided (indicating poor prompt adherence but high creative aesthetics)."

## 4. Metric Aggregation & "Latest" Performance

Currently, calculating "Average" or "Latest" metrics requires deep JOINs across multiple tables.

- **`modelstats` [NEW]**: A summary table that stores the "Cumulative Average" and "Latest Run" metrics for each model.
- **Workflow**: Whenever a `BenchmarkRun` is committed, the backend should trigger an update to this summary table.
- **Benefit**: The main Dashboard and Analytics pages will load instantly, even with thousands of benchmark records.

## 4. Prompt Entity Management

Moving from file-based prompts to database-backed entities.

- **`prompt` [NEW]**: A table to store prompts, their versions, and metadata (tags, category).
- **Stability**: This allows us to link a `ModelResult` to a specific `prompt_id` rather than just a filename, preventing data loss if a prompt file is renamed.

## 5. Output Indexing (Gallery Optimization)

- **`image_output` [NEW]**: A table that indexes generated images on disk.
- **Performance**: Instead of the backend performing an `rglob("*.png")` every time you open a model's gallery, it performs a single SQL query.
- **Feature**: Enables global search/filter of ALL generated images across all models (e.g., "Show me all images with seed 218 across every model").

---

## Next Steps

1.  **Alembic Setup**: Initialize the project for standard migrations.
2.  **Arena Schema**: Implement the `arenavote` table with Bradley-Terry compatible fields.
3.  **BT Calculation Logic**: Implement a background task to compute ratings using MLE or a simplified update rule.
4.  **Aggregation Logic**: Hook into the generation completion event to update summary stats.
