from sqlmodel import Session, select
from database import engine, Model, BenchmarkRun, ModelResult


def inspect():
    with Session(engine) as session:
        print("\n--- Models ---")
        models = session.exec(select(Model)).all()
        for m in models:
            print(f"Hash: {m.hash[:8]}... | Name: {m.name} | Path: {m.path}")

        print("\n--- Benchmark Runs ---")
        runs = session.exec(select(BenchmarkRun)).all()
        for r in runs:
            print(f"ID: {r.id} | Time: {r.timestamp} | Params: {r.parameters}")

        print("\n--- Model Results ---")
        results = session.exec(select(ModelResult)).all()
        for res in results:
            print(
                f"RunID: {res.run_id} | Model: {res.model_hash[:8]}... | Metrics: {res.metrics}"
            )


if __name__ == "__main__":
    inspect()
