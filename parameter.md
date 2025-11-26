# best parameter

  1. Logistic Regression (최고 성능: 0.6843)

  {'C': 0.05, 'max_iter': 2000, 'penalty': 'l1', 'solver': 'liblinear'}
  - 5-Fold CV 평균: 0.6843 (±0.0102)

  2. Ridge Classifier (성능: 0.6797)

  {'alpha': 0.001, 'max_iter': 1000, 'solver': 'saga'}
  - 5-Fold CV 평균: 0.6796 (±0.0101)

  3. Perceptron (성능: 0.6455)

  {'alpha': 0.1, 'eta0': 0.5, 'max_iter': 1000, 'penalty': 'l1'}
  - 5-Fold CV 평균: 0.6455 (±0.0121)

  4. SGD Classifier (성능: 0.6899)

  {'alpha': 0.01, 'eta0': 0.1, 'learning_rate': 'optimal',
   'loss': 'squared_hinge', 'max_iter': 3000, 'penalty': 'l1'}
  - 5-Fold CV 평균: 0.6752 (±0.0360) - 분산이 큼

  5. Gaussian Naive Bayes (성능: 0.6835)

  {'var_smoothing': 1e-11}
  - 5-Fold CV 평균: 0.6835 (±0.0091)

  6. Bernoulli Naive Bayes (성능: 0.6847)

  {'alpha': 5.0, 'binarize': 1.0, 'fit_prior': True}
  - 5-Fold CV 평균: 0.6847 (±0.0132)

  두 번째 탐색 (셀 12, 좁은 범위 재탐색)

  약간 다른 범위로 다시 탐색했을 때:
  - Logistic Regression: C=1 (0.6827)
  - SGD Classifier: loss='modified_huber', penalty='elasticnet' (0.6801)