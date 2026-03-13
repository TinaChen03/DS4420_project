import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_ratings(path="ratings.dat"):
    """
    Load MovieLens-style ratings:
    UserID::MovieID::Rating::Timestamp
    """
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )

    # Convert original IDs to contiguous indices
    user_idx, user_ids = pd.factorize(df["user_id"], sort=False)
    item_idx, item_ids = pd.factorize(df["movie_id"], sort=False)

    users = user_idx.astype(np.int32)
    items = item_idx.astype(np.int32)
    ratings = df["rating"].to_numpy(dtype=np.float64)

    return users, items, ratings, user_ids, item_ids


def train_test_split(users, items, ratings, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    n = len(ratings)
    perm = rng.permutation(n)

    test_size = int(n * test_ratio)
    test_idx = perm[:test_size]
    train_idx = perm[test_size:]

    train_data = (users[train_idx], items[train_idx], ratings[train_idx])
    test_data = (users[test_idx], items[test_idx], ratings[test_idx])

    return train_data, test_data


def predict_many(u_idx, i_idx, mu, bu, bi, P, Q, clip=True):
    preds = mu + bu[u_idx] + bi[i_idx] + np.sum(P[u_idx] * Q[i_idx], axis=1)
    if clip:
        preds = np.clip(preds, 1.0, 5.0)
    return preds


def compute_rmse(data, mu, bu, bi, P, Q):
    u_idx, i_idx, r = data
    preds = predict_many(u_idx, i_idx, mu, bu, bi, P, Q, clip=True)
    return np.sqrt(np.mean((r - preds) ** 2))


def train_svd(
    train_data,
    test_data,
    n_users,
    n_items,
    k=20,
    lr=0.005,
    reg=0.02,
    epochs=10,
    seed=42,
):
    rng = np.random.default_rng(seed)

    train_u, train_i, train_r = train_data

    # Parameters
    mu = train_r.mean()
    bu = np.zeros(n_users, dtype=np.float64)
    bi = np.zeros(n_items, dtype=np.float64)

    # Latent factors
    P = rng.normal(0, 0.1, size=(n_users, k))
    Q = rng.normal(0, 0.1, size=(n_items, k))

    train_rmse_history = []
    test_rmse_history = []

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train_r))

        for idx in order:
            u = train_u[idx]
            i = train_i[idx]
            r = train_r[idx]

            pu = P[u].copy()
            qi = Q[i].copy()

            pred = mu + bu[u] + bi[i] + np.dot(pu, qi)
            err = r - pred

            # Update biases
            bu[u] += lr * (err - reg * bu[u])
            bi[i] += lr * (err - reg * bi[i])

            # Update latent vectors
            P[u] += lr * (err * qi - reg * pu)
            Q[i] += lr * (err * pu - reg * qi)

        train_rmse = compute_rmse(train_data, mu, bu, bi, P, Q)
        test_rmse = compute_rmse(test_data, mu, bu, bi, P, Q)

        train_rmse_history.append(train_rmse)
        test_rmse_history.append(test_rmse)

        print(
            f"Epoch {epoch:02d} | "
            f"Train RMSE: {train_rmse:.4f} | "
            f"Test RMSE: {test_rmse:.4f}"
        )

    return mu, bu, bi, P, Q, train_rmse_history, test_rmse_history


def plot_rmse(train_hist, test_hist, save_path="rmse_curve.png"):
    epochs = range(1, len(train_hist) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_hist, marker="o", label="Train RMSE")
    plt.plot(epochs, test_hist, marker="o", label="Test RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("SVD Matrix Factorization RMSE by Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def print_sample_predictions(test_data, mu, bu, bi, P, Q, sample_n=10):
    test_u, test_i, test_r = test_data
    preds = predict_many(test_u[:sample_n], test_i[:sample_n], mu, bu, bi, P, Q)

    print("\nSample predictions:")
    for j in range(min(sample_n, len(test_r))):
        print(
            f"user_idx={test_u[j]:4d}, "
            f"item_idx={test_i[j]:4d}, "
            f"actual={test_r[j]:.1f}, "
            f"pred={preds[j]:.3f}"
        )


def main():
    ratings_path = "ratings.dat"

    users, items, ratings, user_ids, item_ids = load_ratings(ratings_path)

    n_users = len(user_ids)
    n_items = len(item_ids)

    print(f"Loaded {len(ratings)} ratings")
    print(f"Users: {n_users}, Items: {n_items}")

    train_data, test_data = train_test_split(
        users, items, ratings, test_ratio=0.2, seed=42
    )

    mu, bu, bi, P, Q, train_hist, test_hist = train_svd(
        train_data=train_data,
        test_data=test_data,
        n_users=n_users,
        n_items=n_items,
        k=20,
        lr=0.005,
        reg=0.02,
        epochs=10,
        seed=42,
    )

    final_test_rmse = compute_rmse(test_data, mu, bu, bi, P, Q)
    print(f"\nFinal Test RMSE: {final_test_rmse:.4f}")

    plot_rmse(train_hist, test_hist, save_path="rmse_curve.png")
    print_sample_predictions(test_data, mu, bu, bi, P, Q, sample_n=10)


if __name__ == "__main__":
    main()

    final_test_rmse = rmse(test_data, mu, bu, bi, P, Q)
    print(f"\nFinal Test RMSE: {final_test_rmse:.4f}")

    # Show a few predictions
    test_u, test_i, test_r = test_data
    sample_n = 10
    preds = predict_many(test_u[:sample_n], test_i[:sample_n], mu, bu, bi, P, Q, clip=True)

    print("\nSample predictions:")
    for j in range(sample_n):
        print(
            f"user_idx={test_u[j]:4d}, item_idx={test_i[j]:4d}, "
            f"actual={test_r[j]:.1f}, pred={preds[j]:.3f}"
        )


if __name__ == "__main__":
    main()