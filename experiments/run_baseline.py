def cross_validate(frame, spec, n_splits: int, seed: int) -> dict:
    features = frame[SHARED_FEATURES]
    target = frame["target"].to_numpy()

    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )

    fold_metrics = []

    for train_idx, test_idx in splitter.split(features, target):
        pipeline = build_pipeline(spec)

        pipeline.fit(
            features.iloc[train_idx],
            target[train_idx],
        )

        prob = pipeline.predict_proba(
            features.iloc[test_idx]
        )[:, 1]

        fold_metrics.append(summarize(target[test_idx], prob))

    return {
        key: float(np.mean([m[key] for m in fold_metrics]))
        for key in fold_metrics[0]
    }
