"""Inspect cholesterol distributions across UCI Heart Disease sites."""

from src.data_loader import load_all


def main() -> None:
    data = load_all()

    print("Cholesterol summary by site")
    print(
        data.groupby("site")["chol"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .round(2)
        .to_string()
    )

    print("\nMissing cholesterol values by site")
    print(
        data.groupby("site")["chol"]
        .apply(lambda values: values.isna().sum())
        .to_string()
    )

    print("\nFive smallest non-null cholesterol values by site")
    for site, group in data.groupby("site"):
        values = (
            group["chol"]
            .dropna()
            .sort_values()
            .head(5)
            .tolist()
        )
        print(f"{site:<12} {values}")

    print("\nFive largest non-null cholesterol values by site")
    for site, group in data.groupby("site"):
        values = (
            group["chol"]
            .dropna()
            .sort_values(ascending=False)
            .head(5)
            .tolist()
        )
        print(f"{site:<12} {values}")


if __name__ == "__main__":
    main()
