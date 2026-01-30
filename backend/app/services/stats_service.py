from datetime import datetime, timezone
from sqlmodel import select, desc
from ..core.database import Session, engine, Model, ModelResult, ModelStats, BenchmarkRun


def update_model_stats(session: Session, model_hash: str):
    """
    Calculates and updates the stats for a specific model.
    """
    # 1. Fetch all results for this model
    statement = select(ModelResult).where(ModelResult.model_hash == model_hash).join(BenchmarkRun).order_by(desc(BenchmarkRun.timestamp))
    results = session.exec(statement).all()

    if not results:
        return

    latest_result = results[0]

    # 2. Calculate averages for all metrics
    all_metrics = [r.metrics for r in results if r.metrics]

    # Get all unique metric keys
    keys = set()
    for m in all_metrics:
        keys.update(m.keys())

    avg_metrics = {}
    for key in keys:
        values = [m[key] for m in all_metrics if key in m]
        if values:
            avg_metrics[key] = round(sum(values) / len(values), 3)

    # 3. Get latest metrics
    latest_metrics = latest_result.metrics or {}

    # 4. Upsert ModelStats
    stats = session.get(ModelStats, model_hash)
    if not stats:
        stats = ModelStats(model_hash=model_hash)

    stats.metrics_latest = latest_metrics
    stats.metrics_avg = avg_metrics
    stats.run_count = len(results)
    stats.last_updated = datetime.now(timezone.utc)

    # Sync with current Model.bt_score (computed by bt_service)
    model = session.get(Model, model_hash)
    if model:
        stats.bt_score = model.bt_score

    session.add(stats)
    session.commit()


def rebuild_all_stats(session: Session):
    """
    Rebuilds stats for all models in the database.
    """
    models = session.exec(select(Model)).all()
    for model in models:
        update_model_stats(session, model.hash)
