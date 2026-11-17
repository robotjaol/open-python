"""Regression extras: polynomial, ridge, lasso, elastic net, SVR, gradient boosting, hist gradient boosting."""

from specs_common import DS_DIAB, DS_MPG, REG_INTERP, REG_METRICS
from specs_supervised import BASE, CV_RMSE, SPECS


def reg(num, **kw):
    spec = dict(task="regression", select="RMSE", scoring="neg_root_mean_squared_error", minutes=1,
                level="Intermediate", prereq=["Linear regression notebook", "Train, validation and test splits"],
                skills=["Choose a hyperparameter on validation data", "Compare with a baseline", "Run a sensitivity experiment"],
                metrics_md=REG_METRICS, interpret_md=REG_INTERP,
                exercises=["Repeat the notebook with `SEED = 7` and report how much the headline metric changes.",
                           "Add one more feature or remove one and explain the change using cross validation.",
                           "Replace the validation sweep with `GridSearchCV` and compare the chosen value."],
                lib_md="The estimator is wrapped in a pipeline so imputation, scaling and encoding are fitted on training rows only.",
                train_note="", num=num)
    spec.update(kw)
    spec["folder"] = f"{BASE}/01_regression/{spec['id']}"
    SPECS.append(spec)


reg("02", id="polynomial_regression", title="Polynomial Regression: Fuel Economy versus Horsepower", level="Beginner",
    tagline="Use one feature to see underfitting and overfitting as the polynomial degree grows.",
    name="PolynomialRegression", dataset="auto_mpg", unit="(miles per gallon)",
    prep="data = data[['horsepower', 'mpg']].copy()   # one feature on purpose",
    objectives=["How polynomial features turn a linear model into a curve", "How the degree trades bias against variance",
                "How to pick the degree on validation data and not on the test set"],
    problem=dict(context="Engine power and fuel use are linked by physics, but the link is curved, not straight.",
                 problem="Predict mpg from horsepower alone and decide how flexible the curve should be.",
                 matters="Choosing the wrong flexibility gives either a model that misses real structure or one that memorises noise.",
                 why_ml="The shape of the curve is learned from cars on the road.",
                 why_algo="Polynomial regression is the cleanest way to see the bias variance trade-off: one input, one knob (degree)."),
    papers=["hastie2009"], paper_relation="Hastie et al. discuss basis expansions and the bias variance trade-off, which this notebook shows on real data.",
    math=r"""**Model.** Expand one input $x$ into powers and fit a linear model on them:

$$\hat y = \beta_0 + \beta_1 x + \beta_2 x^2 + \dots + \beta_d x^d$$

**Objective.** Ordinary least squares on the expanded features, $\min_\beta \lVert y - \Phi\beta \rVert_2^2$, where $\Phi_{ij} = x_i^{\,j}$.

**Bias and variance.** For expected squared error at a point, $\mathbb{E}[(y-\hat f)^2] = \text{bias}^2 + \text{variance} + \sigma^2$. Low degree: high bias. High degree: high variance.

**Assumptions.** The true curve is well approximated by a low degree polynomial inside the data range.

**Limitations.** High degree terms explode outside the data range, and the columns $x^j$ are strongly correlated (we standardise after expansion to help the solver).""",
    dataset_md=DS_MPG + " Here we keep only `horsepower` and `mpg`.",
    concept="A degree $d$ polynomial has $d+1$ parameters. Too few parameters cannot follow the curve (underfit), too many follow the noise (overfit).",
    scratch_md="Not implemented from scratch: it is the linear regression notebook applied to expanded columns. The sweep below is the educational part.",
    scratch=None,
    imports="from sklearn.linear_model import LinearRegression\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import PolynomialFeatures",
    params="| Parameter | Meaning |\n|---|---|\n| `degree` | Highest power of the input. The main complexity knob. |\n| `include_bias` | Whether to add a constant column. Off here because the model has its own intercept. |",
    model="make_pipeline(PolynomialFeatures(degree=2, include_bias=False), StandardScaler(), LinearRegression())",
    sweep=("model__polynomialfeatures__degree", [1, 2, 3, 4, 5, 6, 8, 10]),
    experiments=[
        dict(title="Training versus validation error by degree", md="Plot both errors against the degree. The gap between them grows when the model overfits.",
             code='''rows = []
for degree in range(1, 11):
    m = clone(pipe).set_params(model__polynomialfeatures__degree=degree).fit(X_train, y_train)
    rows.append({"degree": degree, "train_RMSE": evaluate_estimator(m, X_train, y_train, TASK)["RMSE"],
                 "val_RMSE": evaluate_estimator(m, X_val, y_val, TASK)["RMSE"]})
curve = pd.DataFrame(rows).set_index("degree")
curve.plot(marker="o")
plt.ylabel("RMSE (mpg)"); plt.title("Bias variance: error versus polynomial degree"); plt.show()
display(curve.round(3))'''),
        dict(title="Fitted curves", md="Draw the fitted curve for three degrees over the data.",
             code='''grid = pd.DataFrame({"horsepower": np.linspace(X_trainval["horsepower"].min(), X_trainval["horsepower"].max(), 200)})
plt.scatter(X_trainval["horsepower"], y_trainval, s=12, alpha=0.5, label="train and validation rows")
for degree in [1, 2, 8]:
    m = clone(pipe).set_params(model__polynomialfeatures__degree=degree).fit(X_trainval, y_trainval)
    plt.plot(grid["horsepower"], m.predict(grid), label=f"degree {degree}")
plt.xlabel("Horsepower"); plt.ylabel("mpg"); plt.legend(); plt.title("Polynomial fits"); plt.show()'''),
        dict(title="Add regularisation to a high degree", md="Degree 8 overfits. Ridge shrinkage can tame it. Compare cross validated RMSE.",
             code=CV_RMSE + '''from sklearn.linear_model import Ridge
raw = cv_rmse(make_pipeline(PolynomialFeatures(8, include_bias=False), StandardScaler(), LinearRegression()))
shrunk = cv_rmse(make_pipeline(PolynomialFeatures(8, include_bias=False), StandardScaler(), Ridge(alpha=10)))
print(f"degree 8 plain: {raw[0]:.3f} +- {raw[1]:.3f}")
print(f"degree 8 ridge: {shrunk[0]:.3f} +- {shrunk[1]:.3f}")''')],
    discussion="The polynomial view of regression is textbook material in Hastie et al. Our sweep reproduces the qualitative U shape of validation error that they describe, on a different dataset.",
    improvements="Use splines or a generalised additive model for smoother local flexibility, and cross validation with more folds for the degree choice.",
    takeaways=["Degree is a complexity knob.", "Choose it on validation data.", "Extrapolation with high degree polynomials is dangerous."],
    use_cases="Calibration curves, simple trend fits, teaching bias and variance.", limitations="Global basis, poor extrapolation, correlated columns.",
    next="Ridge Regression.")

reg("03", id="ridge_regression", title="Ridge Regression: Disease Progression with Correlated Predictors",
    tagline="Shrink coefficients with an L2 penalty and watch the bias variance trade-off on diabetes data.",
    name="Ridge", dataset="diabetes", unit="(progression score)",
    objectives=["What the L2 penalty does to coefficients", "Why scaling matters before penalised regression", "How to read a coefficient path"],
    problem=dict(context="Clinical studies measure many correlated blood markers per patient.",
                 problem="Predict a quantitative measure of diabetes progression one year after baseline from ten baseline features.",
                 matters="Stable and accurate risk scores help target follow up, and unstable coefficients make a score untrustworthy.",
                 why_ml="Markers interact and no closed form biological model is available.",
                 why_algo="The serum measures are strongly correlated, which is exactly where ridge stabilises least squares."),
    papers=["hoerl1970", "efron2004"],
    paper_relation="Hoerl and Kennard introduce ridge for nearly collinear predictors. Efron et al. use the same diabetes data (442 patients) to introduce least angle regression, so the data matches the literature in structure, though not the exact protocol.",
    math=r"""**Objective.**

$$\hat\beta^{\text{ridge}} = \arg\min_\beta \; \lVert y - A\beta \rVert_2^2 + \lambda \lVert \beta \rVert_2^2, \qquad \lambda \ge 0$$

The intercept is not penalised.

**Closed form.** $\hat\beta = (X^\top X + \lambda I)^{-1} X^\top y$. Adding $\lambda I$ makes the matrix invertible even when $X^\top X$ is nearly singular.

**Effect.** In the SVD basis, each component is shrunk by $d_j^2 / (d_j^2 + \lambda)$, so directions of low variance are shrunk most.

**Assumptions.** Same as least squares, plus standardised features so that the penalty treats them equally.

**Limitations.** Never sets a coefficient exactly to zero, so it does not select features. $\lambda$ must be tuned.""",
    dataset_md=DS_DIAB,
    concept="Ridge trades a little bias for a large drop in variance. As $\\lambda$ grows all coefficients move smoothly toward zero.",
    scratch_md="The closed form is short, so we implement it and compare with scikit-learn.",
    scratch='''ytr = y_train.values

def ridge_fit(X, y, alpha):
    A = np.c_[np.ones(len(X)), X]
    penalty = alpha * np.eye(A.shape[1])
    penalty[0, 0] = 0                      # keep intercept free
    return np.linalg.solve(A.T @ A + penalty, A.T @ y)

mine = ridge_fit(Xtr, ytr, alpha=10.0)
reference = Ridge(alpha=10.0).fit(Xtr, ytr)
print("max coefficient gap vs sklearn:", np.abs(mine[1:] - reference.coef_).max())''',
    imports="from sklearn.linear_model import Ridge",
    params="| Parameter | Meaning |\n|---|---|\n| `alpha` | Penalty strength $\\lambda$. Larger means more shrinkage. |\n| `solver` | Numerical method. `auto` is fine here. |",
    model="Ridge(alpha=1.0)", sweep=("model__alpha", [0.001, 0.01, 0.1, 1, 3, 10, 30, 100]),
    experiments=[
        dict(title="Coefficient path", md="Plot every coefficient against $\\lambda$ on a log axis.",
             code='''alphas = np.logspace(-3, 3, 40)
paths = np.array([Ridge(alpha=a).fit(Xtr, y_train).coef_ for a in alphas])
names = feature_names(clone(pipe).fit(X_train, y_train))
plt.plot(alphas, paths)
plt.xscale("log"); plt.xlabel("alpha (penalty strength)"); plt.ylabel("Coefficient (progression units per std)")
plt.legend([n.split("__")[-1] for n in names], ncol=2, fontsize=7); plt.title("Ridge coefficient paths"); plt.show()'''),
        dict(title="Ridge versus ordinary least squares under added collinearity", md="Add a noisy copy of `bmi` and compare cross validated RMSE of OLS and ridge.",
             code=CV_RMSE + '''from sklearn.linear_model import LinearRegression
rng = np.random.default_rng(SEED)
X_dup = X_trainval.copy()
X_dup["bmi_copy"] = X_dup["bmi"] + rng.normal(0, 0.002, len(X_dup))
for name, model in [("OLS", LinearRegression()), ("Ridge alpha=10", Ridge(alpha=10))]:
    mean, std = cv_rmse(make_model_pipeline(X_dup, model), X_dup)
    print(f"{name}: CV RMSE {mean:.2f} +- {std:.2f}")'''),
        dict(title="Why scaling matters", md="Fit ridge with and without scaling on a version of the data where one feature is multiplied by 1000. The penalty is not scale invariant.",
             code=CV_RMSE + '''X_big = X_trainval.copy()
X_big["bmi"] = X_big["bmi"] * 1000
for scaled in [True, False]:
    mean, std = cv_rmse(make_model_pipeline(X_big, Ridge(alpha=1.0), scale=scaled), X_big)
    print(f"scaled={scaled}: CV RMSE {mean:.2f} +- {std:.2f}")''')],
    discussion="Hoerl and Kennard motivate ridge for near collinear predictors, and our coefficient path shows the smooth shrinkage they describe. Efron et al. analyse lasso type paths on the same patients, which the lasso notebook picks up.",
    improvements="Select alpha with `RidgeCV` and nested cross validation, and report intervals with a bootstrap.",
    takeaways=["Ridge shrinks but does not select.", "Scale features first.", "Tune alpha on validation or cross validation data."],
    use_cases="Many correlated predictors, small samples, stable scoring models.", limitations="No feature selection, linear only.",
    next="Lasso Regression.")

reg("04", id="lasso_regression", title="Lasso Regression: Sparse Models for Disease Progression",
    tagline="Use the L1 penalty to set coefficients to exactly zero and select features.",
    name="Lasso", dataset="diabetes", unit="(progression score)",
    objectives=["How the L1 penalty creates sparse models", "How coordinate descent and soft thresholding work", "How to count selected features along the path"],
    problem=dict(context="Clinical teams prefer a score built from few measurements, because each measurement costs time and money.",
                 problem="Predict diabetes progression with as few baseline features as possible.",
                 matters="A short list of predictors is easier to collect, audit and explain.",
                 why_ml="The relevant subset of markers is not known in advance.",
                 why_algo="Lasso fits and selects in one step."),
    papers=["tibshirani1996", "efron2004"],
    paper_relation="Tibshirani introduces the lasso and Efron et al. give the path algorithm and analyse these same 442 patients. Our notebook uses a different split and software, so selected sets may differ.",
    math=r"""**Objective.**

$$\hat\beta^{\text{lasso}} = \arg\min_\beta \; \frac{1}{2n}\lVert y - A\beta \rVert_2^2 + \alpha \lVert \beta \rVert_1$$

(scikit-learn scales the squared error by $1/2n$.)

**Soft thresholding.** For one standardised feature the solution is $\hat\beta_j = \operatorname{sign}(z_j)\,\max(|z_j| - \alpha, 0)$, where $z_j$ is the least squares estimate. Small coefficients become exactly zero.

**Optimisation.** Coordinate descent updates one coefficient at a time using the soft threshold.

**Assumptions.** Sparse truth, standardised features.

**Limitations.** Among correlated features it tends to pick one and drop the others, and the selected set can change with the sample.""",
    dataset_md=DS_DIAB,
    concept="The L1 ball has corners on the axes, so the constrained optimum often lies where some coefficients are zero.",
    scratch_md="We implement cyclic coordinate descent with soft thresholding and check it against scikit-learn.",
    scratch='''ytr = y_train.values

def soft_threshold(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0)

def lasso_cd(X, y, alpha, sweeps=300):
    n, p = X.shape
    w, b = np.zeros(p), y.mean()
    for _ in range(sweeps):
        for j in range(p):
            residual = y - b - X @ w + X[:, j] * w[j]
            w[j] = soft_threshold(X[:, j] @ residual / n, alpha) / (X[:, j] @ X[:, j] / n)
        b = np.mean(y - X @ w)
    return w, b

w_mine, b_mine = lasso_cd(Xtr, ytr, alpha=1.0)
reference = Lasso(alpha=1.0, max_iter=10000).fit(Xtr, ytr)
print("max coefficient gap vs sklearn:", np.abs(w_mine - reference.coef_).max())
print("non zero coefficients (scratch, sklearn):", int((w_mine != 0).sum()), int((reference.coef_ != 0).sum()))''',
    imports="from sklearn.linear_model import Lasso",
    params="| Parameter | Meaning |\n|---|---|\n| `alpha` | L1 strength. Larger means fewer features. |\n| `max_iter` | Coordinate descent sweeps. Raise if a convergence warning appears. |",
    model="Lasso(alpha=0.1, max_iter=10000)", sweep=("model__alpha", [0.001, 0.01, 0.1, 0.3, 1, 3, 10]),
    experiments=[
        dict(title="Number of selected features along the path", md="Count non zero coefficients as alpha grows.",
             code='''alphas = np.logspace(-3, 1.5, 40)
counts = [int((Lasso(alpha=a, max_iter=10000).fit(Xtr, y_train).coef_ != 0).sum()) for a in alphas]
plt.semilogx(alphas, counts, marker="o"); plt.xlabel("alpha"); plt.ylabel("Non zero coefficients")
plt.title("Sparsity versus penalty strength"); plt.show()'''),
        dict(title="Selection stability under resampling", md="Fit lasso on 30 bootstrap samples and count how often each feature is selected. Features chosen in most samples are more trustworthy.",
             code='''rng = np.random.default_rng(SEED)
chosen = np.zeros(Xtr.shape[1])
for _ in range(30):
    idx = rng.integers(0, len(Xtr), len(Xtr))
    chosen += Lasso(alpha=1.0, max_iter=10000).fit(Xtr[idx], y_train.values[idx]).coef_ != 0
freq = pd.Series(chosen / 30, index=X_train.columns).sort_values(ascending=False)
display(freq.round(2).to_frame("selection frequency"))'''),
        dict(title="Lasso versus ridge on prediction", md="Compare cross validated RMSE of the best lasso and ridge alphas.",
             code=CV_RMSE + '''from sklearn.linear_model import Ridge
for name, model in [("Lasso alpha=1", Lasso(alpha=1.0, max_iter=10000)), ("Ridge alpha=10", Ridge(alpha=10))]:
    mean, std = cv_rmse(make_model_pipeline(X_trainval, model))
    print(f"{name}: CV RMSE {mean:.2f} +- {std:.2f}")''')],
    discussion="Tibshirani describes sparse solutions and Efron et al. show the piecewise linear path on the diabetes patients. Our coefficient counts plot shows the same qualitative path. We make no claim about matching their selected variable list.",
    improvements="Use `LassoCV` with repeated cross validation, stability selection, or the relaxed lasso to reduce shrinkage bias on kept features.",
    takeaways=["L1 gives exact zeros.", "Selection can be unstable across samples.", "Standardise before penalising."],
    use_cases="High dimensional data, feature screening, interpretable scores.", limitations="Arbitrary pick among correlated features.",
    next="Elastic Net.")

reg("05", id="elastic_net", title="Elastic Net: Mixing L1 and L2 Penalties",
    tagline="Balance sparsity and stability when predictors are correlated.",
    name="ElasticNet", dataset="diabetes", unit="(progression score)",
    objectives=["How the mixing ratio blends lasso and ridge", "Why the elastic net handles correlated groups better than lasso", "How to tune two parameters sensibly"],
    problem=dict(context="Serum measures `s1` to `s6` move together, so a method that picks one and drops the rest is fragile.",
                 problem="Predict diabetes progression with a model that is sparse but keeps correlated groups together.",
                 matters="Stable selection matters when the selected list is used in a clinical score.",
                 why_ml="Correlation structure is learned from data.",
                 why_algo="The elastic net was designed for exactly this case."),
    papers=["zou2005", "efron2004"],
    paper_relation="Zou and Hastie propose the elastic net for correlated predictors. We apply it to the correlated serum measures of the diabetes data without reproducing their experiments.",
    math=r"""**Objective.**

$$\min_\beta \; \frac{1}{2n}\lVert y - A\beta \rVert_2^2 + \alpha\Big(\rho \lVert \beta \rVert_1 + \frac{1-\rho}{2}\lVert \beta \rVert_2^2\Big)$$

where $\rho \in [0,1]$ is `l1_ratio`. $\rho=1$ is the lasso and $\rho \to 0$ approaches ridge.

**Grouping effect.** The quadratic term makes the objective strictly convex, so highly correlated features receive similar coefficients.

**Optimisation.** Coordinate descent with a soft threshold divided by $1 + \alpha(1-\rho)$.

**Limitations.** Two parameters to tune, still linear.""",
    dataset_md=DS_DIAB,
    concept="Think of the elastic net as the lasso with a ridge stabiliser. The L2 term spreads weight over correlated features, the L1 term still removes weak ones.",
    scratch_md="Not implemented from scratch: the update is the lasso coordinate descent from the previous notebook with a modified denominator.",
    scratch=None,
    imports="from sklearn.linear_model import ElasticNet, Lasso",
    params="| Parameter | Meaning |\n|---|---|\n| `alpha` | Overall penalty strength. |\n| `l1_ratio` | Share of L1 in the penalty. |",
    model="ElasticNet(alpha=0.1, max_iter=10000)", sweep=("model__l1_ratio", [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]),
    experiments=[
        dict(title="Grouping effect on correlated copies", md="Duplicate `bmi` with tiny noise. Lasso tends to keep one copy, the elastic net shares weight.",
             code='''rng = np.random.default_rng(SEED)
Z = StandardScaler().fit_transform(X_trainval)
copy = Z[:, [list(X_trainval.columns).index("bmi")]] + rng.normal(0, 0.01, (len(Z), 1))
Z2 = np.c_[Z, copy]
bmi_cols = [list(X_trainval.columns).index("bmi"), Z2.shape[1] - 1]
for name, m in [("Lasso", Lasso(alpha=0.3, max_iter=10000)), ("ElasticNet l1_ratio=0.3", ElasticNet(alpha=0.3, l1_ratio=0.3, max_iter=10000))]:
    coef = m.fit(Z2, y_trainval).coef_
    print(name, "coefficients on bmi and its copy:", coef[bmi_cols].round(2))'''),
        dict(title="Two parameter grid", md="Cross validated RMSE over alpha and l1_ratio.",
             code='''from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(make_model_pipeline(X_trainval, ElasticNet(max_iter=10000)),
                    {"model__alpha": [0.01, 0.1, 0.3, 1], "model__l1_ratio": [0.1, 0.5, 0.9]},
                    scoring="neg_root_mean_squared_error", cv=5).fit(X_trainval, y_trainval)
res = pd.DataFrame(grid.cv_results_)
display(res.pivot_table(index="param_model__alpha", columns="param_model__l1_ratio", values="mean_test_score").mul(-1).round(2))
print("best:", grid.best_params_)'''),
        dict(title="Sparsity of the three penalties", md="Count zero coefficients for ridge, lasso and elastic net at similar strength.",
             code='''from sklearn.linear_model import Ridge
for name, m in [("Ridge", Ridge(alpha=10)), ("Lasso", Lasso(alpha=1.0, max_iter=10000)), ("ElasticNet", ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000))]:
    coef = m.fit(Xtr, y_train).coef_
    print(f"{name}: zero coefficients {(coef == 0).sum()} of {len(coef)}")''')],
    discussion="Zou and Hastie argue that the elastic net selects correlated groups together. Our duplicate feature experiment is a small controlled check of that claim, not a reproduction of their data.",
    improvements="Use `ElasticNetCV`, repeated cross validation and compare with group lasso if feature groups are known.",
    takeaways=["Elastic net sits between lasso and ridge.", "It shares weight across correlated features.", "Tune alpha and l1_ratio together."],
    use_cases="Genomics, finance and any wide correlated data.", limitations="Two parameters, linear only.",
    next="Support Vector Regression.")

reg("06", id="support_vector_regression", title="Support Vector Regression: Fuel Economy with an Error Tube",
    tagline="Fit a flexible kernel regressor that ignores small errors.", name="SVR", dataset="auto_mpg", unit="(miles per gallon)",
    objectives=["What the epsilon insensitive loss is", "How the RBF kernel gives nonlinear fits", "How C, epsilon and gamma interact"],
    problem=dict(context="Fuel economy depends on weight, power and year in a nonlinear way.",
                 problem="Predict mpg with a model that can bend, without hand crafting polynomial terms.",
                 matters="Better predictions improve design studies and fleet planning.",
                 why_ml="The shape of the relation is unknown.",
                 why_algo="SVR is a strong kernel method for small and medium tabular data and ignores errors inside a tolerance tube."),
    papers=["smola2004", "cortes1995"],
    paper_relation="Smola and Scholkopf review support vector regression. Cortes and Vapnik introduced the classification machine from which it descends.",
    math=r"""**Epsilon insensitive loss.** $L_\varepsilon(y, f) = \max(0, |y - f| - \varepsilon)$.

**Primal problem.**

$$\min_{w,b,\xi,\xi^*}\; \tfrac12 \lVert w \rVert^2 + C\sum_i (\xi_i + \xi_i^*) \quad \text{s.t.}\quad |y_i - w^\top\phi(x_i) - b| \le \varepsilon + \xi_i$$

**Prediction.** $f(x) = \sum_i (\alpha_i - \alpha_i^*) K(x_i, x) + b$ with the RBF kernel $K(x,x') = \exp(-\gamma \lVert x - x' \rVert^2)$.

**Assumptions.** Smooth target, features on comparable scales.

**Limitations.** Training time grows between quadratically and cubically with rows. Needs scaling and tuning of $C$, $\varepsilon$, $\gamma$.""",
    dataset_md=DS_MPG,
    concept="Only points on or outside the tube become support vectors, so the model is sparse in the data. Larger $C$ punishes errors more and gives a wigglier fit.",
    scratch_md="Not implemented from scratch: solving the dual quadratic program is a numerical optimisation task that adds little beyond understanding the loss. We plot the loss instead.",
    scratch='''r = np.linspace(-3, 3, 200)
plt.plot(r, r ** 2, label="squared loss")
plt.plot(r, np.abs(r), label="absolute loss")
plt.plot(r, np.maximum(0, np.abs(r) - 0.5), label="epsilon insensitive (epsilon=0.5)")
plt.xlabel("Residual (mpg)"); plt.ylabel("Loss"); plt.legend(); plt.title("Loss functions"); plt.show()''',
    imports="from sklearn.svm import SVR",
    params="| Parameter | Meaning |\n|---|---|\n| `C` | Cost of errors outside the tube. |\n| `epsilon` | Tube half width in target units. |\n| `gamma` | RBF width. `scale` uses 1 / (n_features * variance). |\n| `kernel` | `rbf`, `linear`, `poly`. |",
    model="SVR(kernel=\"rbf\", epsilon=0.5, gamma=\"scale\")", sweep=("model__C", [0.1, 1, 3, 10, 30, 100, 300]),
    experiments=[
        dict(title="Epsilon and support vectors", md="Larger epsilon means fewer support vectors and a smoother fit.",
             code=CV_RMSE + '''for eps in [0.1, 0.5, 1.0, 2.0, 4.0]:
    m = clone(final_model).set_params(model__epsilon=eps).fit(X_trainval, y_trainval)
    mean, std = cv_rmse(clone(final_model).set_params(model__epsilon=eps))
    print(f"epsilon={eps}: support vectors {m.named_steps['model'].support_.size} of {len(X_trainval)}, CV RMSE {mean:.2f} +- {std:.2f}")'''),
        dict(title="Kernel choice", md="Compare linear, polynomial and RBF kernels with cross validation.",
             code=CV_RMSE + '''for kernel in ["linear", "poly", "rbf"]:
    mean, std = cv_rmse(clone(final_model).set_params(model__kernel=kernel))
    print(f"{kernel}: CV RMSE {mean:.2f} +- {std:.2f}")'''),
        dict(title="Gamma sensitivity", md="A very large gamma memorises points. Compare train and validation error.",
             code='''rows = []
for gamma in [0.001, 0.01, 0.1, 1, 10]:
    m = clone(final_model).set_params(model__gamma=gamma).fit(X_train, y_train)
    rows.append({"gamma": gamma, "train_RMSE": evaluate_estimator(m, X_train, y_train, TASK)["RMSE"], "val_RMSE": evaluate_estimator(m, X_val, y_val, TASK)["RMSE"]})
display(pd.DataFrame(rows).set_index("gamma").round(3))''')],
    discussion="Smola and Scholkopf describe the epsilon tube and sparsity. Our epsilon experiment shows the support vector count shrinking as epsilon grows, matching the described mechanism on different data.",
    improvements="Tune C, epsilon and gamma jointly with random search, and compare with gradient boosting on the same folds.",
    takeaways=["Scale features before SVR.", "Epsilon controls sparsity.", "Gamma and C control flexibility."],
    use_cases="Small and medium nonlinear regression problems.", limitations="Poor scaling to large data, no native uncertainty.",
    next="Gradient Boosting Regressor.")

reg("07", id="gradient_boosting_regressor", title="Gradient Boosting: Sequential Trees on Disease Progression",
    tagline="Build an additive model by fitting each tree to the errors of the previous ones.", name="GradientBoostingRegressor", dataset="diabetes", unit="(progression score)",
    scale=False, minutes=2,
    objectives=["How boosting fits residuals stage by stage", "The roles of learning rate, depth and number of trees", "How to detect overfitting with staged predictions"],
    problem=dict(context="Disease progression may depend on interactions between markers that a linear model cannot express.",
                 problem="Predict progression with a model that can capture nonlinear effects and interactions.",
                 matters="Higher accuracy lets clinicians rank patients for earlier intervention.",
                 why_ml="Interactions are unknown in advance.",
                 why_algo="Gradient boosting is a strong default for tabular data (see Grinsztajn et al. 2022)."),
    papers=["friedman2001", "grinsztajn2022"],
    paper_relation="Friedman defines gradient boosting, which scikit-learn implements. Grinsztajn et al. compare tree ensembles with neural networks on many tabular sets. This notebook uses one small dataset only.",
    math=r"""**Additive model.** $F_M(x) = F_0(x) + \nu \sum_{m=1}^{M} h_m(x)$ with shrinkage $\nu$ (learning rate).

**Stage $m$.** Compute pseudo residuals $r_{im} = -\partial L(y_i, F)/\partial F \,\big|_{F=F_{m-1}(x_i)}$, fit a small tree $h_m$ to them, and add it with a step size.

**Squared error loss.** The pseudo residual is the ordinary residual $y_i - F_{m-1}(x_i)$.

**Assumptions.** Differentiable loss, weak learners with low depth.

**Limitations.** Sequential training, sensitive to learning rate and number of trees, can overfit noisy data.""",
    dataset_md=DS_DIAB,
    concept="Each tree corrects what the ensemble so far gets wrong. A small learning rate with many trees usually generalises better than a large step.",
    scratch_md="With squared error the algorithm is a short loop: fit a shallow tree to the residuals and add a shrunk copy.",
    scratch='''from sklearn.tree import DecisionTreeRegressor
ytr, yva = y_train.values, y_val.values
nu, trees, pred_tr, pred_va = 0.1, [], np.full(len(ytr), ytr.mean()), np.full(len(yva), ytr.mean())
val_curve = []
for m in range(200):
    tree = DecisionTreeRegressor(max_depth=2, random_state=SEED).fit(Xtr, ytr - pred_tr)   # fit residuals
    pred_tr += nu * tree.predict(Xtr)
    pred_va += nu * tree.predict(Xva)
    val_curve.append(np.sqrt(np.mean((pred_va - yva) ** 2)))
print("scratch validation RMSE after 200 stages:", round(val_curve[-1], 3))
ref = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=2, random_state=SEED).fit(Xtr, ytr)
print("sklearn validation RMSE:", round(float(np.sqrt(np.mean((ref.predict(Xva) - yva) ** 2))), 3))
plt.plot(val_curve); plt.xlabel("Boosting stage"); plt.ylabel("Validation RMSE"); plt.title("Scratch boosting"); plt.show()''',
    imports="from sklearn.ensemble import GradientBoostingRegressor",
    params="| Parameter | Meaning |\n|---|---|\n| `n_estimators` | Number of stages. |\n| `learning_rate` | Shrinkage $\\nu$. |\n| `max_depth` | Depth of each tree. |\n| `subsample` | Row fraction per tree. Below 1 gives stochastic boosting. |",
    model="GradientBoostingRegressor(learning_rate=0.05, max_depth=2, subsample=0.8, random_state=SEED)",
    sweep=("model__n_estimators", [25, 50, 100, 200, 400, 800]),
    experiments=[
        dict(title="Staged error and overfitting", md="Use `staged_predict` to plot train and test RMSE by number of trees.",
             code='''gbr = final_model.named_steps["model"]
prep_final = final_model.named_steps["prep"]
tr = [np.sqrt(np.mean((p - y_trainval.values) ** 2)) for p in gbr.staged_predict(prep_final.transform(X_trainval))]
te = [np.sqrt(np.mean((p - y_test.values) ** 2)) for p in gbr.staged_predict(prep_final.transform(X_test))]
plt.plot(tr, label="train + validation"); plt.plot(te, label="test (look only)")
plt.xlabel("Number of trees"); plt.ylabel("RMSE (progression score)"); plt.legend(); plt.title("Staged error"); plt.show()'''),
        dict(title="Learning rate against number of trees", md="Compare learning rates at a fixed budget of trees.",
             code=CV_RMSE + '''for lr in [0.01, 0.05, 0.2]:
    mean, std = cv_rmse(clone(final_model).set_params(model__learning_rate=lr, model__n_estimators=200))
    print(f"learning_rate={lr}: CV RMSE {mean:.2f} +- {std:.2f}")'''),
        dict(title="Tree depth", md="Deeper trees model interactions but overfit faster.",
             code=CV_RMSE + '''for depth in [1, 2, 3, 5]:
    mean, std = cv_rmse(clone(final_model).set_params(model__max_depth=depth))
    print(f"max_depth={depth}: CV RMSE {mean:.2f} +- {std:.2f}")''')],
    discussion="Friedman presents boosting as functional gradient descent, which our scratch loop implements for squared error. Grinsztajn et al. report an advantage of tree ensembles on tabular data. Our single dataset cannot confirm or refute that benchmark result.",
    improvements="Use early stopping on a validation fold, tune with random search, and compare with HistGradientBoosting, XGBoost and LightGBM.",
    takeaways=["Boosting fits residuals sequentially.", "Small learning rate plus many trees is a safe default.", "Use staged predictions to see overfitting."],
    use_cases="Tabular prediction with interactions.", limitations="Sequential, many hyperparameters.",
    next="Histogram Gradient Boosting, then XGBoost.")

reg("08", id="hist_gradient_boosting", title="HistGradientBoosting: Fast Boosting on Fuel Economy",
    tagline="Bin features into histograms to train boosted trees quickly and handle missing values natively.",
    name="HistGradientBoostingRegressor", dataset="auto_mpg", unit="(miles per gallon)", scale=False,
    objectives=["How histogram binning speeds up split search", "Early stopping with an internal validation split", "Native handling of missing values"],
    problem=dict(context="Fuel economy models are rebuilt as new vehicle data arrives, so training speed matters.",
                 problem="Predict mpg with a fast boosted tree model.",
                 matters="Fast retraining supports frequent updates and larger searches.",
                 why_ml="Nonlinear interactions between weight, power and year.",
                 why_algo="HistGradientBoosting is scikit-learn's fast boosting implementation inspired by LightGBM."),
    papers=["friedman2001", "ke2017"],
    paper_relation="Friedman defines gradient boosting. Ke et al. describe the histogram technique used in LightGBM, which scikit-learn's implementation follows in spirit. The Ke et al. entry is not verified in the build environment.",
    math=r"""**Binning.** Each feature is mapped to at most 255 integer bins by quantiles. Split search scans bin edges only, so cost per feature is $O(\text{bins})$ per node after one pass to build gradient histograms.

**Boosting step.** Same additive model as gradient boosting: $F_m = F_{m-1} + \nu\, h_m$ where $h_m$ fits gradients and Hessians of the loss.

**Missing values.** At each split rows with missing values are sent to the side that lowers the loss.

**Limitations.** Binning loses fine resolution on very small data, and early stopping uses part of the training data.""",
    dataset_md=DS_MPG,
    concept="Histograms replace sorting. Instead of testing every unique value as a threshold, the algorithm tests at most a few hundred bin edges.",
    scratch_md="Not implemented from scratch: the histogram build is an engineering optimisation. We demonstrate binning with NumPy instead.",
    scratch='''edges = np.quantile(X_train["weight"], np.linspace(0, 1, 11))
bins = np.digitize(X_train["weight"], edges[1:-1])
print("unique weight values:", X_train["weight"].nunique(), "-> bins:", len(np.unique(bins)))
print(pd.Series(bins).value_counts().sort_index().to_string())''',
    imports="from sklearn.ensemble import HistGradientBoostingRegressor",
    params="| Parameter | Meaning |\n|---|---|\n| `max_iter` | Number of trees. |\n| `learning_rate` | Shrinkage. |\n| `max_leaf_nodes` | Tree size. |\n| `early_stopping` | Stop on an internal validation score. |\n| `l2_regularization` | Penalty on leaf values. |",
    model="HistGradientBoostingRegressor(learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=10, random_state=SEED)",
    sweep=("model__max_iter", [25, 50, 100, 200, 400]),
    experiments=[
        dict(title="Speed versus GradientBoosting", md="Time both on the same data.",
             code='''import time
from sklearn.ensemble import GradientBoostingRegressor
for name, m in [("HistGB", HistGradientBoostingRegressor(max_iter=200, random_state=SEED)), ("GradientBoosting", GradientBoostingRegressor(n_estimators=200, random_state=SEED))]:
    start = time.time(); m.fit(Xtr, y_train); print(f"{name}: {time.time() - start:.2f} s, val RMSE {np.sqrt(np.mean((m.predict(Xva) - y_val.values) ** 2)):.3f}")'''),
        dict(title="Native missing values", md="Hide 15 percent of horsepower values and let the model handle it, compared with median imputation.",
             code=CV_RMSE + '''rng = np.random.default_rng(SEED)
X_hole = X_trainval.copy()
X_hole.loc[rng.random(len(X_hole)) < 0.15, "horsepower"] = np.nan
native = cv_rmse(HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, random_state=SEED), X_hole)
imputed = cv_rmse(make_model_pipeline(X_hole, HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, random_state=SEED), scale=False), X_hole)
print(f"native NaN handling: {native[0]:.2f} +- {native[1]:.2f}")
print(f"median imputation: {imputed[0]:.2f} +- {imputed[1]:.2f}")'''),
        dict(title="Early stopping", md="Let the model stop itself and see how many iterations it keeps.",
             code='''m = HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.05, early_stopping=True, validation_fraction=0.2, n_iter_no_change=20, random_state=SEED).fit(Xtr, y_train)
print("iterations used:", m.n_iter_, "of 1000")''')],
    discussion="Ke et al. report large speedups from histogram methods on big data. Our dataset has under 400 rows, so any timing difference here only demonstrates the mechanism, not their scale results.",
    improvements="Tune with random search over leaf size and regularisation and compare with LightGBM directly.",
    takeaways=["Histograms speed up split search.", "NaN handling is built in.", "Early stopping saves tuning work."],
    use_cases="Medium to large tabular data.", limitations="Less transparent than a single tree.",
    next="Logistic Regression (classification).")
