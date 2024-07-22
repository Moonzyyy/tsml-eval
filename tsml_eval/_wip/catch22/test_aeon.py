


X, y = make_example_3d_numpy()

rist = RISTClassifier(n_intervals=3, n_shapelets=3, series_transformers=None)
rist.fit(X, y)
assert isinstance(rist._estimator, ExtraTreesClassifier)
