import math
from sqlmodel import select, Session
from ..core.database import ArenaVote, Model


def calculate_bt_ratings(votes: list[ArenaVote], model_hashes: list[str], iterations: int = 100) -> dict[str, float]:
    """
    Calculates Bradley-Terry ratings (beta) using the MM algorithm.
    Returns a mapping of model_hash -> beta.
    """
    if not votes or not model_hashes:
        return dict.fromkeys(model_hashes, 1000.0)

    # 1. Initialize data structures
    # W[i] = total wins for model i
    # N[i][j] = total matches between model i and model j
    hashes = list(model_hashes)
    h_to_idx = {h: i for i, h in enumerate(hashes)}
    n = len(hashes)

    W = [0.0] * n
    N = [[0.0] * n for _ in range(n)]

    for v in votes:
        if v.vote_type == "both_bad":
            continue

        i = h_to_idx.get(v.model_a_hash)
        j = h_to_idx.get(v.model_b_hash)

        if i is None or j is None:
            continue

        N[i][j] += 1
        N[j][i] += 1

        if v.vote_type == "model_a":
            W[i] += 1.0
        elif v.vote_type == "model_b":
            W[j] += 1.0
        elif v.vote_type == "tie":
            W[i] += 0.5
            W[j] += 0.5

    # 2. MM Algorithm
    # gamma = e^beta
    gamma = [1.0] * n

    for _ in range(iterations):
        new_gamma = [0.0] * n
        for i in range(n):
            if W[i] == 0:
                new_gamma[i] = 0.01  # Small floor
                continue

            denominator = 0.0
            for j in range(n):
                if i == j:
                    continue
                if N[i][j] > 0:
                    denominator += N[i][j] / (gamma[i] + gamma[j])

            if denominator > 0:
                new_gamma[i] = W[i] / denominator
            else:
                new_gamma[i] = gamma[i]

        # Normalize to prevent overflow/underflow
        sum_gamma = sum(new_gamma)
        if sum_gamma > 0:
            gamma = [g / sum_gamma * n for g in new_gamma]
        else:
            break

    # 3. Convert gamma to beta (log-space) and then to Elo-like scale
    # score = 1000 + 400 * log10(gamma)
    ratings = {}
    for i, h in enumerate(hashes):
        if gamma[i] > 0:
            # log10(gamma) gives a spread where 10x strength = +400 points
            beta = math.log10(gamma[i])
            score = 1000 + (400 * beta)
            ratings[h] = round(score, 1)
        else:
            ratings[h] = 1000.0

    return ratings


def update_all_bt_ratings(session: Session):
    """
    Fetches all votes, calculates new ratings, and updates the models in the DB.
    """
    print("Recalculating Bradley-Terry ratings...")
    votes = session.exec(select(ArenaVote)).all()
    models = session.exec(select(Model)).all()
    model_hashes = [m.hash for m in models]

    ratings_map = calculate_bt_ratings(votes, model_hashes)

    for model in models:
        new_score = ratings_map.get(model.hash, 1000.0)
        if model.bt_score != new_score:
            print(f"Updating {model.name} BT: {model.bt_score} -> {new_score}")
            model.bt_score = new_score
            session.add(model)

    session.commit()
    print("BT Ratings updated.")
