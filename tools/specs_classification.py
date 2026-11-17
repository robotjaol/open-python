"""Classification specs."""

from specs_common import CLS_INTERP, CLS_METRICS, DS_BC, DS_DIGITS, DS_MNIST, DS_TELCO, DS_WINE
from specs_supervised import BASE, CV_F1, SPECS


def cls(num, **kw):
    spec = dict(task="classification", select="F1", scoring="f1", minutes=1, level="Intermediate",
                prereq=["Linear regression notebook", "Train, validation and test splits", "Confusion matrix basics"],
                skills=["Choose a hyperparameter on validation data", "Compare with a baseline", "Run a sensitivity experiment"],
                metrics_md=CLS_METRICS, interpret_md=CLS_INTERP,
                exercises=["Repeat the notebook with `SEED = 7` and report how much the headline metric changes.",
                           "Change the metric used for model choice and see whether the chosen value changes.",
                           "Replace the validation sweep with `GridSearchCV` and compare the chosen value."],
                lib_md="The estimator is wrapped in a pipeline so imputation, scaling and encoding are fitted on training rows only.",
                train_note="", num=num)
    spec.update(kw)
    spec["folder"] = f"{BASE}/02_classification/{spec['id']}"
    SPECS.append(spec)


TELCO = dict(dataset="telco_churn", select="PR_AUC", scoring="average_precision", scale=False)
TELCO_PROBLEM = dict(
    context="Subscription businesses lose revenue when customers leave, and winning a new customer costs more than keeping one.",
    problem="Predict which telecom customers will churn from contract, billing and service information.",
    matters="A ranked list of at risk customers lets a retention team spend a limited budget on the right people.",
    why_ml="Churn depends on many interacting attributes, and the history of past customers is available.")
TELCO_PREP_NOTE = ("\n- `TotalCharges` is missing for new customers, so the median imputer is fitted on training rows only.\n"
                   "- Churn is the minority class, so we select models on PR-AUC (area under the precision recall curve) instead of accuracy.")

cls("01", id="logistic_regression", title="Logistic Regression: Diagnosing Breast Tumours", level="Beginner",
    tagline="Model the probability of malignancy and learn how regularisation and thresholds change decisions.",
    name="LogisticRegression", dataset="breast_cancer", prereq=["Linear regression notebook", "Probability basics"],
    objectives=["How the sigmoid turns a linear score into a probability", "What log loss and maximum likelihood mean",
                "How the decision threshold changes precision and recall", "How regularisation strength C acts"],
    problem=dict(context="Fine needle aspiration is a quick, low cost test. Cell nuclei in the sample are measured from digitised images.",
                 problem="Classify a tumour as malignant or benign from 30 nucleus measurements.",
                 matters="A missed malignancy costs far more than a false alarm, so the error trade-off must be explicit.",
                 why_ml="Many measurements together separate the classes better than any single one.",
                 why_algo="Logistic regression gives calibrated style probabilities and readable coefficients, which clinicians prefer."),
    papers=["wolberg1990", "hastie2009"],
    paper_relation="Wolberg and Mangasarian used earlier cytology features from the same research group. The 30 feature version used here was released later. We use a different model, so the paper is context, not a baseline to match.",
    math=r"""**Model.** $P(y=1\mid x) = \sigma(\beta_0 + x^\top\beta)$ with $\sigma(z) = \dfrac{1}{1+e^{-z}}$.

**Objective (regularised negative log likelihood).**

$$J(\beta) = -\sum_{i}\big[y_i \log p_i + (1-y_i)\log(1-p_i)\big] + \frac{1}{2C}\lVert\beta\rVert_2^2, \quad p_i = \sigma(\beta_0 + x_i^\top\beta)$$

**Gradient.** $\nabla J = A^\top (p - y) + \beta/C$. The objective is convex, so gradient methods reach the global optimum.

**Odds interpretation.** $e^{\beta_j}$ is the multiplicative change in odds for a one unit increase of feature $j$.

**Assumptions.** Log odds linear in features, independent rows, no perfect separation (or use regularisation).

**Limitations.** Linear decision boundary, needs scaling for regularisation, probabilities can be miscalibrated under strong penalty.""",
    dataset_md=DS_BC, eda_note="Many nucleus measures are strongly correlated (radius, perimeter, area).",
    concept="The model draws a hyperplane in feature space. The distance to the plane, passed through the sigmoid, is the predicted probability of malignancy.",
    scratch_md="We implement gradient descent on the log loss with NumPy and compare with scikit-learn.",
    scratch='''ytr = y_train.values.astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticScratch:
    def fit(self, X, y, lr=0.1, steps=1500, l2=0.0):
        A = np.c_[np.ones(len(X)), X]
        w = np.zeros(A.shape[1])
        self.loss = []
        for _ in range(steps):
            p = sigmoid(A @ w)
            w -= lr * (A.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]])
            self.loss.append(-np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))
        self.w = w
        return self

    def predict_proba(self, X):
        return sigmoid(np.c_[np.ones(len(X)), X] @ self.w)

mine = LogisticScratch().fit(Xtr, ytr)
ref = LogisticRegression(C=1e4, max_iter=5000).fit(Xtr, ytr)
agree = np.mean((mine.predict_proba(Xva) > 0.5) == ref.predict(Xva))
print("validation accuracy scratch:", round(float(np.mean((mine.predict_proba(Xva) > 0.5) == y_val.values)), 4),
      "| sklearn:", round(float(ref.score(Xva, y_val)), 4), "| prediction agreement:", round(float(agree), 4))
plt.plot(mine.loss); plt.xlabel("Gradient descent step"); plt.ylabel("Training log loss (nats)")
plt.title("Log loss during training"); plt.show()''',
    imports="from sklearn.linear_model import LogisticRegression",
    params="| Parameter | Meaning |\n|---|---|\n| `C` | Inverse regularisation strength. Smaller C means a stronger penalty. |\n| `l1_ratio` | 0 for L2, 1 for L1 (with the `saga` solver). |\n| `class_weight` | Reweight classes for imbalance. |\n| `max_iter` | Solver iterations. |",
    model="LogisticRegression(max_iter=5000)", sweep=("model__C", [0.001, 0.01, 0.1, 1, 10, 100]),
    experiments=[
        dict(title="Decision threshold", md="Move the probability threshold and watch precision and recall. We use cross validated probabilities on the training plus validation rows, never the test set.",
             code='''from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score
proba = cross_val_predict(final_model, X_trainval, y_trainval, cv=5, method="predict_proba")[:, 1]
rows = [{"threshold": t, "precision": precision_score(y_trainval, proba >= t), "recall": recall_score(y_trainval, proba >= t)} for t in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]]
display(pd.DataFrame(rows).set_index("threshold").round(3))'''),
        dict(title="L1 versus L2 penalty", md="L1 sets coefficients to zero. Count the zeros at the same C.",
             code='''for name, kw in [("L2", dict(l1_ratio=0.0)), ("L1", dict(l1_ratio=1.0))]:
    m = LogisticRegression(C=0.1, solver="saga", max_iter=5000, **kw).fit(Xtr, y_train)
    print(f"{name}: zero coefficients {(m.coef_ == 0).sum()} of {m.coef_.size}, validation accuracy {m.score(Xva, y_val):.3f}")'''),
        dict(title="Reliability of the probabilities", md="A calibration curve compares predicted probability with the observed share of malignant cases.",
             code='''from sklearn.calibration import calibration_curve
prob = final_model.predict_proba(X_test)[:, 1]
frac, mean_pred = calibration_curve(y_test, prob, n_bins=5, strategy="quantile")
plt.plot(mean_pred, frac, marker="o", label="model"); plt.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
plt.xlabel("Predicted probability"); plt.ylabel("Observed fraction malignant"); plt.legend(); plt.title("Calibration (5 quantile bins, test set)"); plt.show()''')],
    discussion="Wolberg and Mangasarian reported strong separation of benign and malignant samples with earlier features. Our model reaches a similar qualitative result on the later 30 feature release, but with a different model and split, so the numbers are not comparable.",
    improvements="Calibrate probabilities, choose the threshold from the cost of errors, and use repeated stratified cross validation because the test set is small.",
    takeaways=["Logistic regression is a linear model of log odds.", "Regularisation strength C is tuned on validation data.", "The threshold is a decision, not a model property."],
    use_cases="Medical screening scores, credit scoring, first baseline for any binary problem.", limitations="Linear boundary and sensitivity to scaling.",
    next="K-Nearest Neighbors.")

cls("02", id="k_nearest_neighbors", title="K-Nearest Neighbors: Classifying Wine Cultivars", level="Beginner",
    tagline="Classify by voting among the closest examples and see why feature scaling decides everything.",
    name="KNeighborsClassifier", dataset="wine", select="F1", scoring="f1_macro", prereq=["Distance and vectors", "Confusion matrix basics"],
    objectives=["How the nearest neighbour rule works", "Why unscaled features break distance based models", "How k trades bias against variance", "Why high dimensions hurt KNN"],
    problem=dict(context="Wine authenticity and quality control use chemical analysis to tell the grape cultivar.",
                 problem="Identify the cultivar of an Italian wine from 13 chemical measurements.",
                 matters="Labelling fraud and blending errors are detected from chemistry without a tasting panel.",
                 why_ml="Cultivars form groups in chemical space.",
                 why_algo="KNN makes no model assumption, which makes it a good first check of whether classes are separable by distance."),
    papers=["cover1967", "hastie2009"],
    paper_relation="Cover and Hart prove the asymptotic error bound of the nearest neighbour rule. Our small sample is far from that asymptotic regime.",
    math=r"""**Rule.** For a query $x$ find the $k$ training points with smallest distance $d(x, x_i)$ and predict the majority label:

$$\hat y(x) = \arg\max_c \sum_{i \in \mathcal{N}_k(x)} \mathbb{1}[y_i = c]$$

**Distance.** Euclidean $d(x,x') = \lVert x - x' \rVert_2$ or Manhattan $\lVert x - x' \rVert_1$.

**Asymptotic bound.** For $k=1$ and $n \to \infty$, the error is at most $2 R^*(1-R^*)$ with $R^*$ the Bayes error, so at most twice the Bayes error (Cover and Hart, 1967).

**Assumptions.** Nearby points share labels, all features are on comparable scales.

**Limitations.** Prediction cost grows with data size. Distances lose meaning in many dimensions. No training, so no interpretable parameters.""",
    dataset_md=DS_WINE, eda_note="Compare the ranges of `proline` (hundreds to over a thousand) and `hue` (below 2).",
    concept="The model is the training set. A new point is labelled by its neighbours, so the geometry of the feature space is everything.",
    scratch_md="KNN is short enough to write in a few NumPy lines.",
    scratch='''ytr, yva = y_train.values, y_val.values

def knn_predict(X_fit, y_fit, X_query, k=5):
    dist = np.sqrt(((X_query[:, None, :] - X_fit[None, :, :]) ** 2).sum(axis=2))
    nearest = np.argsort(dist, axis=1)[:, :k]
    votes = y_fit[nearest]
    return np.array([np.bincount(row).argmax() for row in votes])

mine = knn_predict(Xtr, ytr, Xva, k=5)
ref = KNeighborsClassifier(n_neighbors=5).fit(Xtr, ytr).predict(Xva)
print("agreement with sklearn:", np.mean(mine == ref), "| validation accuracy:", np.mean(mine == yva))''',
    imports="from sklearn.neighbors import KNeighborsClassifier",
    params="| Parameter | Meaning |\n|---|---|\n| `n_neighbors` | Number of voters $k$. |\n| `weights` | `uniform` or `distance` weighting. |\n| `metric` / `p` | Distance type. |",
    model="KNeighborsClassifier()", sweep=("model__n_neighbors", [1, 3, 5, 7, 9, 11, 15, 21]),
    experiments=[
        dict(title="Scaled versus unscaled features", md="Repeat the cross validation with and without standardisation.",
             code=CV_F1 + '''for scaled in [True, False]:
    mean, std = cv_f1(make_model_pipeline(X_trainval, KNeighborsClassifier(n_neighbors=5), scale=scaled), scoring="f1_macro")
    print(f"scaled={scaled}: CV macro F1 {mean:.3f} +- {std:.3f}")'''),
        dict(title="Curse of dimensionality", md="Append pure noise features and watch the score fall, because noise dominates the distance.",
             code=CV_F1 + '''rng = np.random.default_rng(SEED)
for extra in [0, 10, 50, 200]:
    X_noise = X_trainval.copy()
    for j in range(extra):
        X_noise[f"noise{j}"] = rng.normal(size=len(X_noise))
    mean, std = cv_f1(make_model_pipeline(X_noise, KNeighborsClassifier(n_neighbors=5)), scoring="f1_macro") if False else (None, None)
    from sklearn.model_selection import cross_val_score
    s = cross_val_score(make_model_pipeline(X_noise, KNeighborsClassifier(n_neighbors=5)), X_noise, y_trainval, cv=5, scoring="f1_macro")
    print(f"{extra} noise features: CV macro F1 {s.mean():.3f} +- {s.std(ddof=1):.3f}")'''),
        dict(title="Weights and metric", md="Compare uniform and distance weighting, Euclidean and Manhattan distance.",
             code=CV_F1 + '''for weights in ["uniform", "distance"]:
    for p in [1, 2]:
        mean, std = cv_f1(make_model_pipeline(X_trainval, KNeighborsClassifier(n_neighbors=7, weights=weights, p=p)), scoring="f1_macro")
        print(f"weights={weights}, p={p}: CV macro F1 {mean:.3f} +- {std:.3f}")''')],
    discussion="Cover and Hart bound the 1-NN error asymptotically. Our k sweep shows the practical side: with 178 rows the best k is found empirically and results vary between splits, which the cross validation spread makes visible.",
    improvements="Use feature weighting or metric learning, approximate neighbour search for large data, and repeated cross validation for small samples.",
    takeaways=["Always scale features for KNN.", "k controls smoothness.", "Noise features hurt distance based models."],
    use_cases="Small data baselines, recommendation neighbours, anomaly scoring.", limitations="Slow at prediction time, weak in high dimensions.",
    next="Decision Tree.")

cls("03", id="decision_tree", title="Decision Tree: Readable Rules for Customer Churn",
    tagline="Grow a tree of if-then rules, then prune it so it generalises.", name="DecisionTreeClassifier",
    objectives=["How a tree chooses splits with Gini impurity", "Why unpruned trees overfit", "How to read the rules of a small tree", "How to prune with depth limits and cost complexity"],
    problem=dict(TELCO_PROBLEM, why_algo="A shallow tree gives rules a retention manager can read and challenge."),
    papers=["quinlan1986", "hastie2009"],
    paper_relation="Quinlan describes the ID3 family with information gain. scikit-learn implements CART style trees with Gini or entropy, so the principle is the same but the algorithm differs in details.",
    math=r"""**Impurity.** For class shares $p_k$ at a node, Gini is $G = \sum_k p_k(1-p_k)$ and entropy is $H = -\sum_k p_k \log_2 p_k$.

**Split choice.** For feature $j$ and threshold $t$, split rows into left $L$ and right $R$ and minimise

$$\frac{|L|}{|N|} G(L) + \frac{|R|}{|N|} G(R)$$

**Growth.** Recursive greedy splitting until a stopping rule (depth, minimum leaf size) or pure leaves.

**Cost complexity pruning.** Minimise $R(T) + \alpha |T|$ where $R(T)$ is the training error and $|T|$ the number of leaves.

**Assumptions.** None on the distribution.

**Limitations.** Greedy and unstable (small data changes give different trees), axis aligned splits only, high variance when deep.""",
    dataset_md=DS_TELCO, prep_note=TELCO_PREP_NOTE,
    concept="Each internal node asks one yes or no question about one feature, and each leaf stores the class share of the training rows that arrive there.",
    scratch_md="We grow a small tree with Gini impurity on quantile thresholds and compare it with scikit-learn at the same depth.",
    scratch='''from sklearn.metrics import roc_auc_score
ytr = y_train.values
names = prep.get_feature_names_out()

def gini(y):
    p = y.mean()
    return 2 * p * (1 - p)

def best_split(X, y, min_leaf=30):
    best = (None, None, gini(y))
    for j in range(X.shape[1]):
        for t in np.unique(np.quantile(X[:, j], np.linspace(0.1, 0.9, 9))):
            left = X[:, j] <= t
            if left.sum() < min_leaf or (~left).sum() < min_leaf:
                continue
            impurity = (left.sum() * gini(y[left]) + (~left).sum() * gini(y[~left])) / len(y)
            if impurity < best[2]:
                best = (j, t, impurity)
    return best

def grow(X, y, depth):
    if depth == 0 or len(np.unique(y)) == 1:
        return ("leaf", y.mean())
    j, t, _ = best_split(X, y)
    if j is None:
        return ("leaf", y.mean())
    left = X[:, j] <= t
    return ("split", j, t, grow(X[left], y[left], depth - 1), grow(X[~left], y[~left], depth - 1))

def score_row(node, row):
    while node[0] == "split":
        node = node[3] if row[node[1]] <= node[2] else node[4]
    return node[1]

tree = grow(Xtr, ytr, depth=3)
print("root split on:", names[tree[1]], "<=", round(float(tree[2]), 3))
mine = np.array([score_row(tree, r) for r in Xva])
ref = DecisionTreeClassifier(max_depth=3, min_samples_leaf=30, random_state=SEED).fit(Xtr, ytr)
print("validation ROC-AUC scratch:", round(roc_auc_score(y_val, mine), 3), "| sklearn:", round(roc_auc_score(y_val, ref.predict_proba(Xva)[:, 1]), 3))''',
    imports="from sklearn.tree import DecisionTreeClassifier, export_text", **TELCO,
    params="| Parameter | Meaning |\n|---|---|\n| `max_depth` | Longest root to leaf path. |\n| `min_samples_leaf` | Smallest allowed leaf. |\n| `criterion` | `gini` or `entropy`. |\n| `ccp_alpha` | Cost complexity pruning strength. |",
    model="DecisionTreeClassifier(min_samples_leaf=20, random_state=SEED)", sweep=("model__max_depth", [1, 2, 3, 4, 5, 6, 8, 12]),
    experiments=[
        dict(title="Read the rules", md="Print the text rules of a depth 3 tree.",
             code='''small = make_model_pipeline(X_train, DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=SEED), scale=False).fit(X_train, y_train)
print(export_text(small.named_steps["model"], feature_names=list(small.named_steps["prep"].get_feature_names_out()), max_depth=3))'''),
        dict(title="Cost complexity pruning", md="Grow a deep tree and prune it. Cross validated PR-AUC is computed for several alphas.",
             code='''from sklearn.model_selection import cross_val_score
for alpha in [0.0, 0.0005, 0.002, 0.005, 0.01]:
    s = cross_val_score(make_model_pipeline(X_trainval, DecisionTreeClassifier(ccp_alpha=alpha, random_state=SEED), scale=False), X_trainval, y_trainval, cv=5, scoring="average_precision")
    print(f"ccp_alpha={alpha}: CV PR-AUC {s.mean():.3f} +- {s.std(ddof=1):.3f}")'''),
        dict(title="Class weights for the minority class", md="Balanced class weights raise recall at the cost of precision.",
             code='''from sklearn.metrics import precision_score, recall_score
for cw in [None, "balanced"]:
    m = make_model_pipeline(X_train, DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, class_weight=cw, random_state=SEED), scale=False).fit(X_train, y_train)
    p = m.predict(X_val)
    print(f"class_weight={cw}: precision {precision_score(y_val, p):.3f}, recall {recall_score(y_val, p):.3f}")''')],
    discussion="Quinlan presents trees as compact rule sets learned from examples. Our printed rules show the same readability on churn data. We did not use information gain exactly as in ID3.",
    improvements="Use ensembles (random forest, boosting) for accuracy and keep a shallow tree as an explanation surrogate.",
    takeaways=["Trees give readable rules.", "Depth and leaf size control overfitting.", "Single trees are unstable."],
    use_cases="Rule based decisions, explanation, base learner for ensembles.", limitations="High variance and axis aligned splits.",
    next="Random Forest.")

cls("04", id="random_forest", title="Random Forest: Stable Churn Prediction", minutes=2,
    tagline="Average many randomised trees to cut variance, and use out-of-bag data as a free validation set.", name="RandomForestClassifier",
    objectives=["Why bagging reduces variance", "What random feature subsets add", "How out of bag error works", "Why impurity importance can mislead"],
    problem=dict(TELCO_PROBLEM, why_algo="A forest keeps the flexibility of trees and removes most of their instability."),
    papers=["breiman2001", "grinsztajn2022"],
    paper_relation="Breiman introduces random forests. Grinsztajn et al. benchmark tree ensembles on tabular data. We use one dataset only, so we cannot test their general claim.",
    math=r"""**Ensemble.** Train $B$ trees, tree $b$ on a bootstrap sample $\mathcal{D}_b$ and, at every split, on a random subset of $m$ features. Predict by averaging class probabilities:

$$\hat p(y=1 \mid x) = \frac1B \sum_{b=1}^{B} \hat p_b(y=1 \mid x)$$

**Variance of an average.** For $B$ identically distributed trees with variance $\sigma^2$ and pairwise correlation $\rho$:

$$\operatorname{Var}\Big(\frac1B\sum_b T_b\Big) = \rho\,\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

Random feature subsets lower $\rho$, bagging lowers the second term.

**Out of bag (OOB).** About 37 percent of rows are absent from each bootstrap sample. Predicting each row only with trees that did not see it gives an honest error estimate.

**Limitations.** Less interpretable than one tree, large models, impurity importance favours high cardinality and continuous features.""",
    dataset_md=DS_TELCO, prep_note=TELCO_PREP_NOTE,
    concept="Each tree overfits differently. Averaging cancels much of that noise while keeping the signal that all trees share.",
    scratch_md="We build a bagged forest of 40 trees from scikit-learn trees and compare with one tree. This shows the variance reduction directly.",
    scratch='''from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
ytr = y_train.values
rng = np.random.default_rng(SEED)
trees = []
for b in range(40):
    idx = rng.integers(0, len(Xtr), len(Xtr))     # bootstrap rows
    t = DecisionTreeClassifier(max_features="sqrt", min_samples_leaf=3, random_state=b).fit(Xtr[idx], ytr[idx])
    trees.append(t)
forest_proba = np.mean([t.predict_proba(Xva)[:, 1] for t in trees], axis=0)
single_proba = trees[0].predict_proba(Xva)[:, 1]
print("validation ROC-AUC, single tree:", round(roc_auc_score(y_val, single_proba), 3))
print("validation ROC-AUC, forest of 40:", round(roc_auc_score(y_val, forest_proba), 3))''',
    imports="from sklearn.ensemble import RandomForestClassifier", **TELCO,
    params="| Parameter | Meaning |\n|---|---|\n| `n_estimators` | Number of trees. More is rarely harmful. |\n| `max_features` | Features tried at each split. |\n| `min_samples_leaf` | Leaf size, smooths predictions. |\n| `oob_score` | Compute out of bag score. |\n| `class_weight` | Reweight classes. |",
    model="RandomForestClassifier(n_estimators=200, min_samples_leaf=3, n_jobs=2, random_state=SEED)",
    sweep=("model__max_features", [0.1, 0.2, 0.3, 0.5, 0.8]),
    experiments=[
        dict(title="Out of bag score versus number of trees", md="OOB accuracy is computed without any validation split.",
             code='''rows = []
for n in [25, 50, 100, 200, 400]:
    m = RandomForestClassifier(n_estimators=n, min_samples_leaf=3, oob_score=True, n_jobs=2, random_state=SEED).fit(Xtr, y_train)
    rows.append({"trees": n, "oob_accuracy": m.oob_score_, "val_accuracy": m.score(Xva, y_val)})
display(pd.DataFrame(rows).set_index("trees").round(4))'''),
        dict(title="Impurity versus permutation importance", md="Impurity importance is computed on training data and can favour continuous columns. Compare the rankings.",
             code='''rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, n_jobs=2, random_state=SEED).fit(Xtr, y_train)
names = prep.get_feature_names_out()
imp_tree = pd.Series(rf.feature_importances_, index=names).sort_values(ascending=False).head(8)
perm_val = permutation_importance(rf, Xva, y_val, scoring="average_precision", n_repeats=5, random_state=SEED)
imp_perm = pd.Series(perm_val.importances_mean, index=names).sort_values(ascending=False).head(8)
display(pd.concat([imp_tree.rename("impurity (train)").reset_index(), imp_perm.rename("permutation (validation)").reset_index()], axis=1).round(4))'''),
        dict(title="Forest versus single tree and class weights", md="Compare cross validated PR-AUC of a single tree, a forest and a balanced forest.",
             code='''from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
candidates = {"single tree": DecisionTreeClassifier(min_samples_leaf=20, random_state=SEED),
              "forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=3, n_jobs=2, random_state=SEED),
              "forest balanced": RandomForestClassifier(n_estimators=150, min_samples_leaf=3, class_weight="balanced", n_jobs=2, random_state=SEED)}
for name, m in candidates.items():
    s = cross_val_score(make_model_pipeline(X_trainval, m, scale=False), X_trainval, y_trainval, cv=5, scoring="average_precision")
    print(f"{name}: CV PR-AUC {s.mean():.3f} +- {s.std(ddof=1):.3f}")''')],
    discussion="Breiman shows that generalisation error depends on tree strength and correlation. Our bagging scratch cell shows the correlation effect numerically: averaging trees gives a higher validation AUC than a single tree.",
    improvements="Tune min_samples_leaf and max_features jointly, calibrate probabilities, and compare with gradient boosting under identical splits.",
    takeaways=["Averaging trees reduces variance.", "OOB score is a free validation estimate.", "Permutation importance is safer than impurity importance."],
    use_cases="Strong default for tabular classification.", limitations="Large models, importance caveats.",
    next="Support Vector Machine.")

cls("05", id="support_vector_machine", title="Support Vector Machine: Reading Handwritten Digits", minutes=2,
    tagline="Find the maximum margin boundary and bend it with the RBF kernel.", name="SVC", dataset="digits", select="F1", scoring="f1_macro",
    perm=False, prereq=["Logistic regression notebook", "Vectors and dot products"],
    objectives=["What the margin is and why maximising it helps", "How the kernel trick gives nonlinear boundaries", "How C and gamma control flexibility", "How support vectors define the model"],
    problem=dict(context="Postal services, banks and forms processing read handwritten digits automatically.",
                 problem="Classify 8 by 8 pixel images of handwritten digits into ten classes.",
                 matters="Every misread digit is a mis-sorted letter or a wrong amount.",
                 why_ml="Writing styles vary too much for hand made rules.",
                 why_algo="SVMs were among the strongest digit classifiers in the 1990s and remain strong on small image sets."),
    papers=["cortes1995", "hastie2009"],
    paper_relation="Cortes and Vapnik demonstrated support vector networks on handwritten digits (USPS and NIST data). We use the UCI optical digits, which are smaller and different, so no number is comparable.",
    math=r"""**Hard margin.** Find $w, b$ maximising the margin $2/\lVert w\rVert$ subject to $y_i(w^\top x_i + b) \ge 1$.

**Soft margin primal.**

$$\min_{w,b,\xi}\; \tfrac12\lVert w\rVert^2 + C\sum_i \xi_i \quad \text{s.t.}\quad y_i(w^\top\phi(x_i)+b)\ge 1-\xi_i,\; \xi_i \ge 0$$

**Dual and kernel.** $f(x) = \sum_i \alpha_i y_i K(x_i, x) + b$, with RBF kernel $K(x,x') = \exp(-\gamma\lVert x-x'\rVert^2)$. Only points with $\alpha_i > 0$ (support vectors) matter.

**Multiclass.** scikit-learn trains one versus one classifiers and votes.

**Limitations.** Training cost grows quickly with rows, needs scaling, and raw scores are not probabilities (Platt scaling is used when `probability=True`).""",
    dataset_md=DS_DIGITS, eda_note="Pixel values run from 0 to 16 and some border pixels are always zero.",
    concept="Only points near the boundary shape it. The kernel measures similarity, so the boundary can curve around digit classes.",
    scratch_md="We implement a linear SVM by sub-gradient descent on the hinge loss for the binary task 'is this an 8?' and compare with `LinearSVC`.",
    scratch='''from sklearn.svm import LinearSVC
ytr8 = np.where(y_train.values == 8, 1.0, -1.0)
yva8 = np.where(y_val.values == 8, 1.0, -1.0)
w, b, lam, lr = np.zeros(Xtr.shape[1]), 0.0, 0.01, 0.05
for epoch in range(300):
    margin = ytr8 * (Xtr @ w + b)
    active = margin < 1                                  # points inside margin
    grad_w = lam * w - (ytr8[active, None] * Xtr[active]).sum(axis=0) / len(Xtr)
    grad_b = -ytr8[active].sum() / len(Xtr)
    w -= lr * grad_w
    b -= lr * grad_b
pred = np.sign(Xva @ w + b)
ref = LinearSVC(C=1.0, max_iter=20000).fit(Xtr, ytr8)
print("validation accuracy for 'is it an 8', scratch:", round(float(np.mean(pred == yva8)), 4), "| LinearSVC:", round(float(ref.score(Xva, yva8)), 4))
print("share of training points inside the margin:", round(float(active.mean()), 3))''',
    imports="from sklearn.svm import SVC",
    params="| Parameter | Meaning |\n|---|---|\n| `C` | Cost of margin violations. Large C fits training data harder. |\n| `gamma` | RBF reach. Large gamma means very local boundaries. |\n| `kernel` | `rbf`, `linear`, `poly`. |\n| `probability` | Fit Platt scaling for probabilities (slower). |",
    model="SVC(kernel=\"rbf\", gamma=\"scale\", probability=True, random_state=SEED)", sweep=("model__C", [0.1, 1, 10, 100]),
    experiments=[
        dict(title="C and gamma grid", md="Cross validated macro F1 over a small grid (no probability fitting, to save time).",
             code='''from sklearn.model_selection import cross_val_score
rows = []
for C in [0.1, 1, 10]:
    for gamma in [0.0005, 0.002, 0.008, 0.03]:
        s = cross_val_score(make_model_pipeline(X_trainval, SVC(C=C, gamma=gamma)), X_trainval, y_trainval, cv=3, scoring="f1_macro")
        rows.append({"C": C, "gamma": gamma, "cv_f1": s.mean()})
display(pd.DataFrame(rows).pivot(index="C", columns="gamma", values="cv_f1").round(3))'''),
        dict(title="Kernels", md="Compare linear, polynomial and RBF kernels.",
             code='''from sklearn.model_selection import cross_val_score
for kernel in ["linear", "poly", "rbf"]:
    s = cross_val_score(make_model_pipeline(X_trainval, SVC(kernel=kernel)), X_trainval, y_trainval, cv=5, scoring="f1_macro")
    print(f"{kernel}: CV macro F1 {s.mean():.3f} +- {s.std(ddof=1):.3f}")'''),
        dict(title="Support vectors versus C", md="A larger C means fewer margin violations and usually fewer support vectors.",
             code='''for C in [0.1, 1, 10, 100]:
    m = SVC(C=C, gamma="scale").fit(Xtr, y_train)
    print(f"C={C}: support vectors {m.n_support_.sum()} of {len(Xtr)}, validation accuracy {m.score(Xva, y_val):.3f}")''')],
    discussion="Cortes and Vapnik reported strong digit recognition with polynomial support vector networks. Our digits are a different, smaller dataset, so we can only note that kernel SVMs again perform strongly on digit images.",
    improvements="Use data augmentation, larger images (MNIST), and nested cross validation for C and gamma.",
    takeaways=["Margin maximisation gives good generalisation.", "Scale features and tune C and gamma.", "Support vectors are the only points that matter."],
    use_cases="Small image sets, text classification, bioinformatics.", limitations="Poor scaling to very large data.",
    next="Naive Bayes.")

cls("06", id="naive_bayes", title="Naive Bayes: Handwritten Digits from Pixel Statistics", minutes=2,
    tagline="Classify with Bayes rule and an independence assumption, and learn why it still works.", name="BernoulliNB", dataset="mnist_5000", select="F1", scoring="f1_macro",
    perm=False, scale=False, prereq=["Probability basics", "Bayes rule"],
    objectives=["How Bayes rule gives a classifier", "What the naive independence assumption means", "Why Laplace smoothing is needed", "Why naive Bayes can classify well with poor probabilities"],
    problem=dict(context="Naive Bayes is a classic method for text filtering and sparse count data. Binary pixels give the same structure on images.",
                 problem="Classify MNIST digits using only whether each pixel is on.",
                 matters="A fast, tiny model is useful as a baseline and when training data is scarce.",
                 why_ml="We estimate per class pixel probabilities from examples.",
                 why_algo="Naive Bayes trains with one pass of counting. Spam detection with the SMS Spam Collection would be the textbook case, but that dataset could not be downloaded in the build environment, so images are used."),
    papers=["domingos1997", "hastie2009"],
    paper_relation="Domingos and Pazzani explain why naive Bayes often classifies well despite dependent features. Pixels are strongly dependent, so this notebook is a direct test of that idea.",
    math=r"""**Bayes rule.** $P(c \mid x) \propto P(c)\, P(x \mid c)$.

**Naive assumption.** Features are independent given the class: $P(x \mid c) = \prod_{j=1}^p P(x_j \mid c)$.

**Bernoulli model.** For binary features, $P(x_j = 1 \mid c) = \theta_{cj}$, estimated with Laplace smoothing

$$\hat\theta_{cj} = \frac{N_{cj} + \alpha}{N_c + 2\alpha}$$

**Decision.** $\hat c = \arg\max_c \Big[\log P(c) + \sum_j x_j \log\theta_{cj} + (1-x_j)\log(1-\theta_{cj})\Big]$.

**Limitations.** Independence is false for pixels, so probabilities are over confident and poorly calibrated, but the ranking of classes can still be right.""",
    dataset_md=DS_MNIST, eda_note="Many pixels near the image border are always zero.",
    concept="Each class has its own map of how likely each pixel is to be on. A new image is scored against each map and the best fitting class wins.",
    scratch_md="Bernoulli naive Bayes needs only counting and logs.",
    scratch='''ytr, yva = y_train.values, y_val.values
Btr, Bva = (Xtr > 127).astype(float), (Xva > 127).astype(float)   # pixel on or off
classes = np.unique(ytr)
prior = np.array([(ytr == c).mean() for c in classes])
alpha = 1.0
theta = np.array([(Btr[ytr == c].sum(axis=0) + alpha) / ((ytr == c).sum() + 2 * alpha) for c in classes])
log_post = Bva @ np.log(theta).T + (1 - Bva) @ np.log(1 - theta).T + np.log(prior)
mine = classes[log_post.argmax(axis=1)]
ref = BernoulliNB(alpha=1.0, binarize=127).fit(Xtr, ytr)
print("agreement with sklearn:", np.mean(mine == ref.predict(Xva)), "| validation accuracy:", np.mean(mine == yva))''',
    imports="from sklearn.naive_bayes import BernoulliNB",
    params="| Parameter | Meaning |\n|---|---|\n| `alpha` | Laplace smoothing. Zero allows zero probabilities. |\n| `binarize` | Threshold turning pixels into on or off. |\n| `fit_prior` | Learn class priors. |",
    model="BernoulliNB(binarize=127)", sweep=("model__alpha", [0.01, 0.1, 0.5, 1, 5]),
    experiments=[
        dict(title="Class probability maps", md="Show the learned pixel probabilities for three digits.",
             code='''nb = final_model.named_steps["model"]
fig, axes = plt.subplots(1, 3, figsize=(9, 3))
for ax, digit in zip(axes, [0, 1, 8]):
    ax.imshow(np.exp(nb.feature_log_prob_[digit]).reshape(28, 28), cmap="viridis"); ax.set_title(f"P(pixel on | digit {digit})"); ax.axis("off")
plt.show()'''),
        dict(title="Binarisation threshold", md="The threshold decides what counts as ink.",
             code='''from sklearn.model_selection import cross_val_score
for thr in [30, 80, 127, 180]:
    s = cross_val_score(make_model_pipeline(X_trainval, BernoulliNB(binarize=thr), scale=False), X_trainval, y_trainval, cv=5, scoring="f1_macro")
    print(f"binarize={thr}: CV macro F1 {s.mean():.3f} +- {s.std(ddof=1):.3f}")'''),
        dict(title="Naive Bayes versus logistic regression", md="Discriminative models do not assume independence. Compare cross validated scores.",
             code='''from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
for name, m in [("BernoulliNB", BernoulliNB(binarize=127)), ("Logistic regression", LogisticRegression(max_iter=300))]:
    s = cross_val_score(make_model_pipeline(X_trainval, m, scale=(name != "BernoulliNB")), X_trainval, y_trainval, cv=3, scoring="f1_macro")
    print(f"{name}: CV macro F1 {s.mean():.3f} +- {s.std(ddof=1):.3f}")''')],
    discussion="Domingos and Pazzani argue that naive Bayes can be optimal under zero one loss even when independence fails. The comparison with logistic regression above shows how large the practical gap is on strongly dependent pixels.",
    improvements="Use Gaussian or multinomial variants where they fit the features and calibrate probabilities afterwards.",
    takeaways=["Training is counting.", "Smoothing avoids zero probabilities.", "Good ranking does not need good probabilities."],
    use_cases="Spam filtering, document classification, quick baselines.", limitations="Poor probability calibration, ignores feature dependence.",
    next="XGBoost.")

cls("07", id="xgboost", title="XGBoost: Regularised Boosting for Churn", minutes=2,
    tagline="Use second order boosting with built-in regularisation and early stopping.", name="XGBClassifier",
    objectives=["What makes XGBoost different from plain gradient boosting", "How regularisation terms enter the split gain", "How to use early stopping without leaking the test set"],
    problem=dict(TELCO_PROBLEM, why_algo="XGBoost is a widely used, regularised boosting library with strong defaults on tabular data. Credit default prediction (UCI) is the textbook case, but that data could not be downloaded here."),
    papers=["chen2016", "chen2016arxiv", "friedman2001", "grinsztajn2022"],
    paper_relation="Chen and Guestrin describe the system. We use the open source package at default style settings on a small dataset, so none of their scale results are tested.",
    math=r"""**Objective at round $t$.**

$$\mathcal{L}^{(t)} = \sum_i \ell\big(y_i, \hat y_i^{(t-1)} + f_t(x_i)\big) + \Omega(f_t), \qquad \Omega(f) = \gamma T + \tfrac12 \lambda \sum_{j=1}^{T} w_j^2$$

with $T$ leaves and leaf weights $w_j$.

**Second order approximation.** With $g_i, h_i$ the first and second derivative of the loss, the optimal leaf weight is $w_j^* = -G_j/(H_j + \lambda)$ and a split gain is

$$\text{Gain} = \tfrac12\Big[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\Big] - \gamma$$

**Logistic loss.** $g_i = p_i - y_i$, $h_i = p_i(1-p_i)$.

**Limitations.** Many hyperparameters, needs early stopping or tuning, less interpretable than one tree.""",
    dataset_md=DS_TELCO, prep_note=TELCO_PREP_NOTE,
    concept="Instead of fitting residuals only, XGBoost uses curvature (second derivative) of the loss to choose splits and leaf values, and penalises complex trees directly.",
    scratch_md="Not implemented from scratch: a faithful version needs sparsity aware split finding and a lot of engineering. We compute the gradient and Hessian that drive the algorithm for logistic loss.",
    scratch='''p = np.clip(np.full(len(y_train), y_train.mean()), 1e-6, 1 - 1e-6)    # first round prediction
g = p - y_train.values
h = p * (1 - p)
print("round 1 gradient for a churner:", round(float(g[y_train.values == 1][0]), 3), "| for a stayer:", round(float(g[y_train.values == 0][0]), 3))
print("Hessian (same for all rows at round 1):", round(float(h[0]), 3))
G, H = g.sum(), h.sum()
print("single leaf optimal weight w = -G/(H+lambda), lambda=1:", round(float(-G / (H + 1)), 4))''',
    imports="from xgboost import XGBClassifier", **TELCO,
    params="| Parameter | Meaning |\n|---|---|\n| `n_estimators` | Boosting rounds. |\n| `learning_rate` | Shrinkage. |\n| `max_depth` | Tree depth. |\n| `subsample`, `colsample_bytree` | Row and column sampling. |\n| `reg_lambda`, `gamma` | L2 on leaf weights and minimum split gain. |",
    model="XGBClassifier(learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, eval_metric=\"logloss\", tree_method=\"hist\", n_jobs=2, random_state=SEED)",
    sweep=("model__n_estimators", [25, 50, 100, 200, 400]),
    experiments=[
        dict(title="Early stopping on the validation rows", md="Stop when validation log loss stops improving. The test set is not used.",
             code='''m = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", early_stopping_rounds=30, tree_method="hist", n_jobs=2, random_state=SEED)
m.fit(Xtr, y_train, eval_set=[(Xva, y_val)], verbose=False)
print("best iteration:", m.best_iteration, "| validation log loss:", round(m.best_score, 4))
hist = m.evals_result()["validation_0"]["logloss"]
plt.plot(hist); plt.axvline(m.best_iteration, color="red", linestyle="--"); plt.xlabel("Boosting round"); plt.ylabel("Validation log loss"); plt.title("Early stopping"); plt.show()'''),
        dict(title="Regularisation", md="Vary `reg_lambda` and `gamma`.",
             code='''from sklearn.model_selection import cross_val_score
for lam, gam in [(0, 0), (1, 0), (10, 0), (1, 1), (10, 1)]:
    m = clone(final_model).set_params(model__reg_lambda=lam, model__gamma=gam)
    s = cross_val_score(m, X_trainval, y_trainval, cv=5, scoring="average_precision")
    print(f"lambda={lam}, gamma={gam}: CV PR-AUC {s.mean():.3f} +- {s.std(ddof=1):.3f}")'''),
        dict(title="Importance types", md="Gain based importance and permutation importance may rank features differently.",
             code='''names = prep.get_feature_names_out()
m = clone(final_model.named_steps["model"]).fit(Xtr, y_train)
gain = pd.Series(m.get_booster().get_score(importance_type="gain")).rename(lambda k: names[int(k[1:])]).sort_values(ascending=False).head(6)
perm_v = pd.Series(permutation_importance(m, Xva, y_val, scoring="average_precision", n_repeats=5, random_state=SEED).importances_mean, index=names).sort_values(ascending=False).head(6)
display(pd.concat([gain.rename("gain").reset_index(), perm_v.rename("permutation").reset_index()], axis=1).round(4))''')],
    discussion="Chen and Guestrin report that regularisation and second order information improve boosting. Our regularisation experiment checks the direction of that claim on a small dataset, where gains are small and within the cross validation spread.",
    improvements="Tune with random search or Bayesian search, use the native categorical support, and compare with LightGBM and CatBoost on identical folds.",
    takeaways=["XGBoost adds regularisation and second order steps.", "Use early stopping on validation data.", "Compare importance types."],
    use_cases="Tabular competitions and production scoring.", limitations="Tuning effort, less transparent.",
    next="LightGBM.")

cls("08", id="lightgbm", title="LightGBM: Fast Leaf-Wise Boosting for Churn", minutes=2,
    tagline="Grow trees leaf by leaf on histogram bins and handle categories natively.", name="LGBMClassifier",
    objectives=["How leaf-wise growth differs from level-wise growth", "Why num_leaves is the main complexity control", "How native categorical features compare with one-hot encoding"],
    problem=dict(TELCO_PROBLEM, why_algo="LightGBM trains fast and handles categorical columns without one-hot expansion."),
    papers=["ke2017", "friedman2001", "grinsztajn2022"],
    paper_relation="Ke et al. introduce LightGBM. That entry could not be verified in the build environment, so check it before citing. Our dataset is small, so speed claims are not tested.",
    math=r"""**Boosting objective.** Same additive model as gradient boosting with a second order approximation of the loss.

**Leaf-wise growth.** At each step split the leaf with the largest gain, no matter its depth. This reduces loss faster than level-wise growth at the same number of leaves but can overfit small data, so `num_leaves` and `min_child_samples` matter.

**Histogram split finding.** Features are binned (default 255 bins). Split search costs $O(\text{bins})$ per feature.

**GOSS.** Gradient based one side sampling keeps rows with large gradients and samples small gradient rows.

**EFB.** Exclusive feature bundling merges mutually exclusive sparse features.

**Limitations.** Leaf-wise growth overfits small datasets without constraints.""",
    dataset_md=DS_TELCO, prep_note=TELCO_PREP_NOTE,
    concept="Instead of growing the whole next level, the tree keeps splitting the single best leaf. The shape is lopsided, but each split buys the most loss reduction.",
    scratch_md="Not implemented from scratch: histogram binning and leaf-wise search are engineering optimisations. We illustrate leaf-wise gain selection on one split.",
    scratch='''g = Xtr[:, 0] * 0 + (y_train.values - y_train.mean())
names = prep.get_feature_names_out()
gains = []
for j in range(Xtr.shape[1]):
    thr = np.median(Xtr[:, j])
    left = Xtr[:, j] <= thr
    if left.all() or (~left).all():
        continue
    gl, gr = g[left].sum(), g[~left].sum()
    gains.append((names[j], gl ** 2 / left.sum() + gr ** 2 / (~left).sum() - g.sum() ** 2 / len(g)))
print("best first splits by variance reduction at the median:")
display(pd.DataFrame(gains, columns=["feature", "gain"]).sort_values("gain", ascending=False).head(5).round(2))''',
    imports="from lightgbm import LGBMClassifier", **TELCO,
    params="| Parameter | Meaning |\n|---|---|\n| `num_leaves` | Maximum leaves per tree, the main complexity knob. |\n| `learning_rate`, `n_estimators` | Shrinkage and rounds. |\n| `min_child_samples` | Minimum rows in a leaf. |\n| `subsample`, `colsample_bytree` | Row and column sampling. |",
    model="LGBMClassifier(n_estimators=200, learning_rate=0.05, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=2, verbose=-1, random_state=SEED)",
    sweep=("model__num_leaves", [4, 8, 15, 31, 63]),
    experiments=[
        dict(title="Native categorical features", md="Pass string columns as pandas categories and compare with one-hot encoding by cross validated PR-AUC.",
             code='''from sklearn.model_selection import cross_val_score
Xc = X_trainval.copy()
for col in Xc.select_dtypes("object"):
    Xc[col] = Xc[col].astype("category")
params = dict(n_estimators=150, learning_rate=0.05, num_leaves=15, n_jobs=2, verbose=-1, random_state=SEED)
native = cross_val_score(LGBMClassifier(**params), Xc, y_trainval, cv=5, scoring="average_precision")
onehot = cross_val_score(make_model_pipeline(X_trainval, LGBMClassifier(**params), scale=False), X_trainval, y_trainval, cv=5, scoring="average_precision")
print(f"native categorical: CV PR-AUC {native.mean():.3f} +- {native.std(ddof=1):.3f}")
print(f"one-hot: CV PR-AUC {onehot.mean():.3f} +- {onehot.std(ddof=1):.3f}")'''),
        dict(title="Leaves versus depth", md="Cap depth while keeping many leaves to see how the two controls interact.",
             code='''from sklearn.model_selection import cross_val_score
for leaves, depth in [(31, -1), (31, 3), (8, -1), (63, 6)]:
    m = clone(final_model).set_params(model__num_leaves=leaves, model__max_depth=depth)
    s = cross_val_score(m, X_trainval, y_trainval, cv=5, scoring="average_precision")
    print(f"num_leaves={leaves}, max_depth={depth}: CV PR-AUC {s.mean():.3f} +- {s.std(ddof=1):.3f}")'''),
        dict(title="Early stopping callback", md="Stop on validation log loss.",
             code='''import lightgbm as lgb
m = LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=15, n_jobs=2, verbose=-1, random_state=SEED)
m.fit(Xtr, y_train, eval_set=[(Xva, y_val)], eval_metric="logloss", callbacks=[lgb.early_stopping(30, verbose=False)])
print("best iteration:", m.best_iteration_)''')],
    discussion="Ke et al. report that histogram and sampling techniques give large speedups. We do not measure scale effects here. The leaves versus depth experiment checks the qualitative point that leaf count controls complexity.",
    improvements="Tune `min_child_samples` and regularisation, and run repeated cross validation because differences between boosters on 7000 rows are often within noise.",
    takeaways=["num_leaves is the key knob.", "Native categoricals can replace one-hot.", "Early stopping uses validation data only."],
    use_cases="Large tabular data and fast experiments.", limitations="Overfits small data when leaves are not limited.",
    next="CatBoost.")

cls("09", id="catboost", title="CatBoost: Boosting with Categorical Features", minutes=3,
    tagline="Use ordered boosting and ordered target statistics to treat categories safely.", name="CatBoostClassifier",
    objectives=["Why target encoding can leak and how ordered statistics avoid it", "How symmetric trees work", "How to pass categorical columns natively"],
    problem=dict(TELCO_PROBLEM, why_algo="Most churn columns are categorical, which is where CatBoost is designed to help."),
    papers=["prokhorenkova2018", "friedman2001"],
    paper_relation="Prokhorenkova et al. introduce CatBoost. We use the library at short training length and compare native categoricals with one-hot encoding, a small check of their motivation.",
    math=r"""**Prediction shift.** Standard boosting computes gradients on the same rows it was trained on, which biases them. Ordered boosting trains with models that never saw the row they score.

**Ordered target statistic.** For a categorical value $c$ and a random permutation, row $i$ is encoded as

$$\hat x_i = \frac{\sum_{j \prec i,\; c_j = c_i} y_j + a\,p}{\sum_{j \prec i,\; c_j = c_i} 1 + a}$$

using only rows before $i$ in the permutation, with prior $p$ and weight $a$. This avoids target leakage.

**Oblivious trees.** All nodes at the same depth use the same split, which gives fast, regular predictors.

**Limitations.** Slower training than histogram libraries on large data, more memory.""",
    dataset_md=DS_TELCO, prep_note=TELCO_PREP_NOTE,
    concept="CatBoost converts each category into a number using only earlier rows, so the encoding cannot memorise the row's own label.",
    scratch_md="Not implemented from scratch: ordered boosting is subtle. We implement the ordered target statistic for one column to show how it avoids leakage.",
    scratch='''col = "Contract"
y_arr = y_train.values
rng = np.random.default_rng(SEED)
order = rng.permutation(len(X_train))
prior, weight = y_arr.mean(), 1.0
sums, counts, encoded = {}, {}, np.zeros(len(order))
for position, i in enumerate(order):
    value = X_train[col].iloc[i]
    encoded[i] = (sums.get(value, 0) + weight * prior) / (counts.get(value, 0) + weight)   # earlier rows only
    sums[value] = sums.get(value, 0) + y_arr[i]
    counts[value] = counts.get(value, 0) + 1
naive = X_train.groupby(col).apply(lambda g: y_arr[X_train.index.get_indexer(g.index)].mean())
naive_encoded = X_train[col].map(naive).values
print("correlation with own label, ordered encoding:", round(float(np.corrcoef(encoded, y_arr)[0, 1]), 3))
print("correlation with own label, naive mean encoding:", round(float(np.corrcoef(naive_encoded, y_arr)[0, 1]), 3))''',
    imports="from catboost import CatBoostClassifier", **TELCO,
    params="| Parameter | Meaning |\n|---|---|\n| `iterations` | Number of trees. |\n| `depth` | Depth of symmetric trees. |\n| `learning_rate` | Shrinkage. |\n| `cat_features` | Columns treated as categories. |\n| `l2_leaf_reg` | L2 penalty. |",
    model="CatBoostClassifier(iterations=150, learning_rate=0.08, depth=5, random_seed=SEED, verbose=0, thread_count=2, allow_writing_files=False)",
    sweep=("model__depth", [3, 4, 6]),
    experiments=[
        dict(title="Native categoricals versus one-hot", md="Compare cross validated PR-AUC (3 folds to limit run time).",
             code='''from sklearn.model_selection import cross_val_score
cat_cols = list(X_trainval.select_dtypes("object").columns)
Xn = X_trainval.copy()
Xn[cat_cols] = Xn[cat_cols].astype(str)
Xn = Xn.fillna(Xn.median(numeric_only=True))
params = dict(iterations=100, learning_rate=0.1, depth=5, random_seed=SEED, verbose=0, thread_count=2, allow_writing_files=False)
native = cross_val_score(CatBoostClassifier(cat_features=cat_cols, **params), Xn, y_trainval, cv=3, scoring="average_precision")
onehot = cross_val_score(make_model_pipeline(X_trainval, CatBoostClassifier(**params), scale=False), X_trainval, y_trainval, cv=3, scoring="average_precision")
print(f"native categorical: CV PR-AUC {native.mean():.3f} +- {native.std(ddof=1):.3f}")
print(f"one-hot: CV PR-AUC {onehot.mean():.3f} +- {onehot.std(ddof=1):.3f}")'''),
        dict(title="Best iteration on a validation set", md="Use the validation rows to choose the number of trees.",
             code='''m = CatBoostClassifier(iterations=600, learning_rate=0.05, depth=5, random_seed=SEED, verbose=0, thread_count=2, allow_writing_files=False, early_stopping_rounds=40)
m.fit(Xtr, y_train, eval_set=(Xva, y_val))
print("best iteration:", m.get_best_iteration())'''),
        dict(title="Learning rate and iterations", md="A lower learning rate needs more trees.",
             code='''for lr, its in [(0.3, 60), (0.1, 150), (0.03, 500)]:
    m = CatBoostClassifier(iterations=its, learning_rate=lr, depth=5, random_seed=SEED, verbose=0, thread_count=2, allow_writing_files=False).fit(Xtr, y_train)
    p = m.predict_proba(Xva)[:, 1]
    from sklearn.metrics import average_precision_score
    print(f"lr={lr}, iterations={its}: validation PR-AUC {average_precision_score(y_val, p):.3f}")''')],
    discussion="Prokhorenkova et al. argue that ordered statistics reduce target leakage. Our scratch cell shows the mechanism: the ordered encoding is less correlated with the row's own label than naive mean encoding. This is an illustration on one column, not a reproduction.",
    improvements="Tune depth, l2_leaf_reg and learning rate with early stopping, and use native categoricals in the final model.",
    takeaways=["Naive target encoding leaks.", "Ordered statistics fix that.", "CatBoost is strongest when many columns are categorical."],
    use_cases="Datasets with many categorical columns.", limitations="Slower on large data.",
    next="AdaBoost.")

cls("10", id="adaboost", title="AdaBoost: Boosting Stumps on Breast Cancer Data",
    tagline="Reweight the hard examples after each round and combine weak learners.", name="AdaBoostClassifier", dataset="breast_cancer", select="F1", scoring="f1",
    prereq=["Decision tree notebook", "Logistic regression notebook"],
    objectives=["How example weights shift towards mistakes", "How the weak learner votes are weighted", "Why AdaBoost is sensitive to label noise"],
    problem=dict(context="A single simple rule on one nucleus measurement is a weak classifier of tumours.",
                 problem="Combine many one-split rules into a strong malignancy classifier.",
                 matters="Boosting shows how weak signals can be combined into an accurate decision.",
                 why_ml="No single feature separates the classes.",
                 why_algo="AdaBoost is the original boosting algorithm and has a clean theory."),
    papers=["freund1997", "friedman2001"],
    paper_relation="Freund and Schapire introduce AdaBoost and bound its error. Friedman later reads boosting as stage-wise optimisation of a loss. We use stumps on a different dataset from the paper.",
    math=r"""**Weights.** Start with $w_i = 1/n$.

**Round $m$.** Fit a weak learner $h_m$ with weighted error $\epsilon_m = \sum_i w_i \mathbb{1}[h_m(x_i) \ne y_i] / \sum_i w_i$. Set

$$\alpha_m = \tfrac12 \ln\frac{1-\epsilon_m}{\epsilon_m}, \qquad w_i \leftarrow w_i\, e^{-\alpha_m y_i h_m(x_i)}\ \text{then normalise}$$

with labels $y_i \in \{-1, +1\}$.

**Prediction.** $H(x) = \operatorname{sign}\big(\sum_m \alpha_m h_m(x)\big)$.

**Link to loss.** This minimises exponential loss $e^{-y f(x)}$ stage by stage.

**Limitations.** Exponential loss is sensitive to outliers and mislabelled rows.""",
    dataset_md=DS_BC,
    concept="Rows that earlier stumps misclassify get more weight, so the next stump must deal with them. Strong learners are built from weak ones.",
    scratch_md="We implement discrete AdaBoost with scikit-learn stumps as the weak learner.",
    scratch='''ypm = np.where(y_train.values == 1, 1, -1)
yvpm = np.where(y_val.values == 1, 1, -1)
w = np.ones(len(ypm)) / len(ypm)
stumps, alphas, val_acc = [], [], []
for m in range(60):
    stump = DecisionTreeClassifier(max_depth=1, random_state=SEED).fit(Xtr, ypm, sample_weight=w)
    pred = stump.predict(Xtr)
    err = w[pred != ypm].sum() / w.sum()
    a = 0.5 * np.log((1 - err) / (err + 1e-12))
    w = w * np.exp(-a * ypm * pred)
    w = w / w.sum()                              # renormalise
    stumps.append(stump); alphas.append(a)
    score = sum(al * s.predict(Xva) for al, s in zip(alphas, stumps))
    val_acc.append(np.mean(np.sign(score) == yvpm))
print("validation accuracy after 60 stumps (scratch):", round(val_acc[-1], 4))
ref = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=60, random_state=SEED).fit(Xtr, y_train)
print("sklearn validation accuracy:", round(float(ref.score(Xva, y_val)), 4))
plt.plot(val_acc); plt.xlabel("Boosting round"); plt.ylabel("Validation accuracy"); plt.title("Scratch AdaBoost"); plt.show()''',
    imports="from sklearn.ensemble import AdaBoostClassifier\nfrom sklearn.tree import DecisionTreeClassifier",
    params="| Parameter | Meaning |\n|---|---|\n| `n_estimators` | Number of weak learners. |\n| `learning_rate` | Shrinks each learner's vote. |\n| `estimator` | Weak learner, stumps by default. |",
    model="AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), random_state=SEED)",
    sweep=("model__n_estimators", [10, 25, 50, 100, 200, 400]),
    experiments=[
        dict(title="Error against number of rounds", md="Staged accuracy on training and validation rows.",
             code='''ada = clone(final_model.named_steps["model"]).fit(Xtr, y_train)
tr = list(ada.staged_score(Xtr, y_train)); va = list(ada.staged_score(Xva, y_val))
plt.plot(tr, label="train"); plt.plot(va, label="validation"); plt.xlabel("Boosting round"); plt.ylabel("Accuracy"); plt.legend(); plt.title("Staged accuracy"); plt.show()'''),
        dict(title="Label noise", md="Flip 10 percent of training labels and test on clean validation labels. AdaBoost versus a random forest.",
             code='''from sklearn.ensemble import RandomForestClassifier
rng = np.random.default_rng(SEED)
flip = rng.random(len(y_train)) < 0.10
y_noisy = np.where(flip, 1 - y_train.values, y_train.values)
for name, m in [("AdaBoost", AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=200, random_state=SEED)), ("Random forest", RandomForestClassifier(n_estimators=200, random_state=SEED))]:
    clean = clone(m).fit(Xtr, y_train).score(Xva, y_val)
    noisy = clone(m).fit(Xtr, y_noisy).score(Xva, y_val)
    print(f"{name}: validation accuracy clean labels {clean:.3f}, 10% flipped labels {noisy:.3f}")'''),
        dict(title="Weak learner depth", md="Stumps versus depth 3 trees.",
             code='''from sklearn.model_selection import cross_val_score
for depth in [1, 2, 3]:
    s = cross_val_score(make_model_pipeline(X_trainval, AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=depth), n_estimators=100, random_state=SEED)), X_trainval, y_trainval, cv=5, scoring="f1")
    print(f"depth {depth}: CV F1 {s.mean():.3f} +- {s.std(ddof=1):.3f}")''')],
    discussion="Freund and Schapire show that training error falls quickly with rounds and give generalisation bounds. Our staged accuracy plot shows the empirical side on a different dataset. The noise experiment tests the commonly cited weakness to label noise on this data only.",
    improvements="Compare with gradient boosting under the same splits, use early stopping, and try the SAMME.R style real boosting where available.",
    takeaways=["Boosting reweights hard rows.", "Stumps are enough weak learners.", "Label noise hurts exponential loss."],
    use_cases="Small clean datasets, feature selection with stumps.", limitations="Sensitive to noisy labels.",
    next="Extra Trees.")

cls("11", id="extra_trees", title="Extra Trees: Randomised Splits on Digit Images", minutes=2,
    tagline="Pick split thresholds at random to push variance down even further than a random forest.", name="ExtraTreesClassifier", dataset="digits",
    select="F1", scoring="f1_macro", perm=False,
    objectives=["How extremely randomised trees choose splits", "How the bias variance balance differs from random forests", "How to visualise importance on image pixels"],
    problem=dict(context="Digit images have many correlated pixels, where very random splits can still find useful structure.",
                 problem="Classify 8 by 8 handwritten digits into ten classes.",
                 matters="Cheap and strong classifiers are useful baselines for image like data.",
                 why_ml="Writing varies too much for rules.",
                 why_algo="Extra trees train faster than random forests because they skip the threshold search."),
    papers=["breiman2001", "hastie2009"],
    paper_relation="Extra trees were introduced by Geurts, Ernst and Wehenkel (2006), which is not in our verified list. We cite Breiman for the ensemble framework and Hastie et al. for the comparison. Check Geurts et al. before citing it as the original.",
    math=r"""**Split rule.** At a node, draw $K$ random features. For each, draw a threshold uniformly between the minimum and maximum value of that feature in the node, and keep the best of the $K$ random splits by impurity decrease.

**Ensemble.** No bootstrap by default, trees use the full training set. Average predicted probabilities as in random forests.

**Bias and variance.** Random thresholds raise the bias of each tree a little and lower the correlation between trees, so the ensemble variance drops.

**Limitations.** Needs many trees, importance has the same caveats as in random forests.""",
    dataset_md=DS_DIGITS,
    concept="Instead of searching for the best threshold, the algorithm draws one at random and keeps the best of a few draws. Many trees average the randomness away.",
    scratch_md="Not implemented from scratch: it is the random forest of the previous notebook with a different split rule. The experiments compare the two directly.",
    scratch=None,
    imports="from sklearn.ensemble import ExtraTreesClassifier",
    params="| Parameter | Meaning |\n|---|---|\n| `n_estimators` | Number of trees. |\n| `max_features` | Features drawn at each node. |\n| `bootstrap` | Off by default. |\n| `min_samples_leaf` | Leaf size. |",
    model="ExtraTreesClassifier(n_estimators=300, n_jobs=2, random_state=SEED)", sweep=("model__max_features", [0.05, 0.1, 0.2, 0.4, 0.7]),
    experiments=[
        dict(title="Importance on the pixel grid", md="Plot impurity importance as an 8 by 8 image.",
             code='''et = final_model.named_steps["model"]
plt.imshow(et.feature_importances_.reshape(8, 8), cmap="magma"); plt.colorbar(label="Impurity importance")
plt.title("Where the trees look (8 by 8 pixel grid)"); plt.axis("off"); plt.show()'''),
        dict(title="Extra trees versus random forest", md="Cross validated score and fit time.",
             code='''import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
for name, m in [("ExtraTrees", ExtraTreesClassifier(n_estimators=200, n_jobs=2, random_state=SEED)), ("RandomForest", RandomForestClassifier(n_estimators=200, n_jobs=2, random_state=SEED))]:
    start = time.time()
    s = cross_val_score(m, X_trainval, y_trainval, cv=5, scoring="f1_macro")
    print(f"{name}: CV macro F1 {s.mean():.3f} +- {s.std(ddof=1):.3f}, {time.time() - start:.1f} s")'''),
        dict(title="Number of trees", md="Validation score against the ensemble size.",
             code='''rows = []
for n in [5, 10, 25, 50, 100, 300]:
    m = ExtraTreesClassifier(n_estimators=n, n_jobs=2, random_state=SEED).fit(Xtr, y_train)
    rows.append({"trees": n, "val_accuracy": m.score(Xva, y_val)})
display(pd.DataFrame(rows).set_index("trees").round(4))''')],
    discussion="The original extra trees paper is not in the verified list, so we make no claim about its results. Breiman's analysis of tree correlation explains why added randomness helps an ensemble, and the tree count experiment shows the averaging effect.",
    improvements="Compare against gradient boosting and a small neural network on the same folds.",
    takeaways=["Random thresholds are cheap and effective.", "More trees reduce variance.", "Importance can be shown on the image grid."],
    use_cases="Fast strong baselines.", limitations="Less accurate than tuned boosting on many tabular tasks.",
    next="Multi-Layer Perceptron.")

cls("12", id="mlp", title="Multi-Layer Perceptron: A Small Neural Network for Digits", minutes=2,
    tagline="Stack linear layers with nonlinearities and train them with backpropagation.", name="MLPClassifier", dataset="digits",
    select="F1", scoring="f1_macro", perm=False, prereq=["Logistic regression notebook", "Chain rule of derivatives"],
    objectives=["How a hidden layer creates features", "How backpropagation computes gradients", "Why neural networks vary between random seeds", "How early stopping and L2 regularisation help"],
    problem=dict(context="Digit recognition is the classic test of neural networks.",
                 problem="Classify 8 by 8 digit images with a one hidden layer network.",
                 matters="The same building blocks scale to deep networks for images and text.",
                 why_ml="Features are learned, not designed.",
                 why_algo="A small MLP is the simplest neural network that is nonlinear."),
    papers=["hastie2009", "grinsztajn2022"],
    paper_relation="Hastie et al. cover neural networks and backpropagation. Grinsztajn et al. found tree models stronger than neural networks on tabular data, but our data are images, where neural networks are usually strong.",
    math=r"""**Forward pass.** $h = \max(0, W_1 x + b_1)$, $z = W_2 h + b_2$, $p = \operatorname{softmax}(z)$.

**Loss.** Cross entropy $\ell = -\sum_k y_k \log p_k$ plus an L2 term $\tfrac{\alpha}{2}\lVert W \rVert^2$.

**Backpropagation.** $\partial\ell/\partial z = p - y$, $\partial\ell/\partial W_2 = h^\top (p-y)$, then $\partial\ell/\partial h = (p-y) W_2^\top$ and $\partial\ell/\partial z_1 = \partial\ell/\partial h \odot \mathbb{1}[z_1 > 0]$.

**Optimisation.** Gradient descent variants (Adam in scikit-learn). The loss is non convex, so results depend on the seed.

**Limitations.** Needs scaling, many hyperparameters, no guarantee of the global optimum.""",
    dataset_md=DS_DIGITS,
    concept="The hidden layer learns stroke like detectors. The output layer combines them into class scores.",
    scratch_md="We train a one hidden layer network with full batch gradient descent in NumPy.",
    scratch='''ytr, yva = y_train.values, y_val.values
K, hidden = 10, 32
Y = np.eye(K)[ytr]
rng = np.random.default_rng(SEED)
W1, b1 = rng.normal(0, 0.1, (Xtr.shape[1], hidden)), np.zeros(hidden)
W2, b2 = rng.normal(0, 0.1, (hidden, K)), np.zeros(K)
losses = []
for epoch in range(300):
    Z1 = Xtr @ W1 + b1
    A1 = np.maximum(0, Z1)
    Z2 = A1 @ W2 + b2
    P = np.exp(Z2 - Z2.max(axis=1, keepdims=True)); P /= P.sum(axis=1, keepdims=True)
    losses.append(-np.mean(np.log(P[np.arange(len(ytr)), ytr] + 1e-12)))
    dZ2 = (P - Y) / len(ytr)
    dW2, db2 = A1.T @ dZ2, dZ2.sum(axis=0)
    dZ1 = (dZ2 @ W2.T) * (Z1 > 0)
    dW1, db1 = Xtr.T @ dZ1, dZ1.sum(axis=0)
    W1 -= 0.5 * dW1; b1 -= 0.5 * db1; W2 -= 0.5 * dW2; b2 -= 0.5 * db2
pred = (np.maximum(0, Xva @ W1 + b1) @ W2 + b2).argmax(axis=1)
print("validation accuracy, scratch network:", round(float(np.mean(pred == yva)), 4))
plt.plot(losses); plt.xlabel("Epoch"); plt.ylabel("Training cross entropy (nats)"); plt.title("Scratch MLP training loss"); plt.show()''',
    imports="from sklearn.neural_network import MLPClassifier",
    params="| Parameter | Meaning |\n|---|---|\n| `hidden_layer_sizes` | Units per hidden layer. |\n| `alpha` | L2 penalty. |\n| `learning_rate_init` | Adam step size. |\n| `early_stopping` | Hold out data to stop training. |\n| `max_iter` | Epoch limit. |",
    model="MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=500, early_stopping=True, random_state=SEED)",
    sweep=("model__hidden_layer_sizes", [(8,), (16,), (32,), (64,), (128,)]),
    experiments=[
        dict(title="Seed variability", md="Train the same network with five seeds and report the spread of validation accuracy.",
             code='''accs = []
for seed in range(5):
    m = MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=500, early_stopping=True, random_state=seed).fit(Xtr, y_train)
    accs.append(m.score(Xva, y_val))
print("validation accuracy by seed:", np.round(accs, 4), "| mean", round(float(np.mean(accs)), 4), "| std", round(float(np.std(accs, ddof=1)), 4))'''),
        dict(title="Regularisation strength", md="Vary the L2 penalty.",
             code='''for alpha in [1e-6, 1e-3, 1e-1, 1.0, 3.0]:
    m = MLPClassifier(hidden_layer_sizes=(64,), alpha=alpha, max_iter=500, random_state=SEED).fit(Xtr, y_train)
    print(f"alpha={alpha}: train accuracy {m.score(Xtr, y_train):.3f}, validation accuracy {m.score(Xva, y_val):.3f}")'''),
        dict(title="Learning curve of the loss", md="Plot the loss recorded during training.",
             code='''m = MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=300, random_state=SEED).fit(Xtr, y_train)
plt.plot(m.loss_curve_); plt.xlabel("Epoch"); plt.ylabel("Training loss (cross entropy)"); plt.title("sklearn MLP loss curve"); plt.show()''')],
    discussion="The tabular benchmark of Grinsztajn et al. does not apply to images, so we make no claim about it here. Our seed experiment illustrates why single run neural network results should be reported with a spread.",
    improvements="Use convolutional networks for larger images, tune the learning rate, and report several seeds.",
    takeaways=["A hidden layer adds nonlinearity.", "Backpropagation is the chain rule.", "Report several seeds."],
    use_cases="Images, audio, text, and as a flexible baseline.", limitations="Non convex, seed dependent.",
    next="Unsupervised learning: K-Means.")
