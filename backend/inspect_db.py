from sqlalchemy import inspect
from app.core.database import engine


def inspect_db():
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"Tables found: {tables}")

    for table in tables:
        print(f"\n--- Columns in {table} ---")
        columns = inspector.get_columns(table)
        for col in columns:
            print(f"  {col['name']}: {col['type']}")


if __name__ == "__main__":
    inspect_db()
