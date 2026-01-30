from app.core.database import Model, BenchmarkRun, ModelResult, get_session
from datetime import datetime, timezone


def add_fake_data():
    session = get_session()

    # Use the hash found
    model_hash = "c6310906622ae7bae4828e92da16e396414ccb329883cd8b3767bb05e9c4aa80"

    # 1. Create Benchmark Run
    run = BenchmarkRun(
        timestamp=datetime.now(timezone.utc),
        parameters={"steps": 20, "cfg": 7.0, "sampler": "Euler a", "width": 1024, "height": 1024, "seed": 42},
        prompts=["A cyberpunk city at night, neon lights, highly detailed"],
        prompt_set_id="v1_standard",
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    print(f"Created BenchmarkRun with ID: {run.id}")

    # 2. Create Model Result
    result = ModelResult(
        run_id=run.id,
        model_hash=model_hash,
        metrics={"accuracy": 0.852, "diversity": 0.641, "vqa_score": 0.78, "lpips_loss": 0.123},
        image_count=4,
    )
    session.add(result)
    session.commit()
    print(f"Created ModelResult for model: {model_hash}")


if __name__ == "__main__":
    add_fake_data()
