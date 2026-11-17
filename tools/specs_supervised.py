"""Supervised learning notebook specs."""

LEVEL_BEGINNER = "Beginner"
LEVEL_MID = "Intermediate"
BASE = "01_supervised_learning"

CV_RMSE = '''from sklearn.model_selection import cross_val_score
def cv_rmse(model, X_data=None, y_data=None):
    scores = cross_val_score(model, X_data if X_data is not None else X_trainval,
                             y_data if y_data is not None else y_trainval,
                             cv=5, scoring="neg_root_mean_squared_error")
    return -scores.mean(), scores.std(ddof=1)
'''

CV_F1 = '''from sklearn.model_selection import cross_val_score
def cv_f1(model, scoring="f1"):
    scores = cross_val_score(model, X_trainval, y_trainval, cv=5, scoring=scoring)
    return scores.mean(), scores.std(ddof=1)
'''

SPECS = []


def add(**spec):
    SPECS.append(spec)


# ------------------------------------------------------------------ 1 linear regression
add(
    id="linear_regression", num="01", folder=f"{BASE}/01_regression/linear_regression",
    title="Linear Regression: Predicting Vehicle Fuel Economy", level=LEVEL_BEGINNER,
    tagline="Fit and diagnose ordinary least squares on real car data, from the normal equations to residual checks.",
    name="LinearRegression", task="regression", dataset="auto_mpg", unit="(miles per gallon)", minutes=1,
    prereq=["Python and pandas basics", "Matrices and the dot product", "Mean and variance"],
    objectives=["How least squares chooses coefficients and why the normal equations work",
                "Why we split the data before scaling and fitting",
                "How to read RMSE, MAE and R-squared and compare them with a dummy baseline",
                "How collinearity and nonlinearity show up in the diagnostics"],
    skills=["Build a leakage free regression pipeline", "Write OLS from scratch with NumPy", "Diagnose residuals"],
    problem=dict(
        context="Fuel economy decides running cost and emissions of a vehicle. Engineers and regulators compare designs by miles per gallon (mpg).",
        problem="Predict the mpg of a car from engine and body attributes: cylinders, displacement, horsepower, weight, acceleration, model year and origin.",
        matters="A model that links design choices to fuel use supports early design trade studies and fleet level emission estimates.",
        why_ml="Many attributes interact and the relation is estimated from past cars, not derived from physics. A statistical fit is the quickest honest first model.",
        why_algo="Linear regression is the simplest model with interpretable coefficients. It sets the baseline that more flexible models must beat."),
    papers=["hastie2009", "hoerl1970"],
    paper_relation="The Auto MPG data was donated to the UCI repository in 1993 and is a standard regression teaching set. Hastie et al. give the least squares theory used here. Hoerl and Kennard motivate why collinear predictors, which this dataset has, make coefficients unstable.",
    math=r"""**Model.** For a row $x_i \in \mathbb{R}^p$ and target $y_i$:

$$y_i = \beta_0 + x_i^\top \beta + \varepsilon_i, \qquad \varepsilon_i \sim (0, \sigma^2)$$

**Objective (residual sum of squares).**

$$\hat\beta = \arg\min_{\beta} \; \sum_{i=1}^{n} \left(y_i - \beta_0 - x_i^\top \beta\right)^2 = \arg\min_{\beta}\; \lVert y - A\beta \rVert_2^2$$

where $A$ is the design matrix with a leading column of ones.

**Optimisation.** Setting the gradient to zero gives the normal equations

$$A^\top A\, \hat\beta = A^\top y \quad\Rightarrow\quad \hat\beta = (A^\top A)^{-1} A^\top y.$$

Gradient descent is the alternative: $\beta \leftarrow \beta - \eta \cdot \tfrac{2}{n} A^\top (A\beta - y)$.

**Notation.** $n$ rows, $p$ features, $\beta_0$ intercept, $\eta$ learning rate, $\varepsilon$ noise.

**Assumptions.** Linear relation in the parameters, errors with mean zero and constant variance, errors uncorrelated with the features, and (for standard errors) no perfect collinearity.

**Limitations.** Sensitive to outliers because errors are squared. Cannot capture curvature or interactions unless you add terms. Unstable coefficients when features are strongly correlated.""",
    dataset_md="**Auto MPG** (Quinlan, 1993) from the UCI Machine Learning Repository, licence CC BY 4.0, DOI 10.24432/C5859H. The UCI page lists 398 cars; this repository uses the 392 rows without missing horsepower as shipped in `mlxtend`. Target `mpg` (continuous). Features: `cylinders`, `displacement` (cubic inches), `horsepower`, `weight` (pounds), `acceleration` (seconds from 0 to 60 mph), `model_year` (1970 to 1982), `origin` (1 US, 2 Europe, 3 Japan, coded as numbers). All features are numeric. Known limits: cars from 1970 to 1982 only, so the model says nothing about modern vehicles.",
    eda_note="Look for strong correlation between `displacement`, `cylinders`, `horsepower` and `weight`, and for a curved relation between weight and mpg.",
    concept="Least squares draws the hyperplane that minimises the sum of squared vertical distances to the data. With standardised features each coefficient is the change in mpg for a one standard deviation change in that feature, holding the others fixed.",
    scratch_md="We implement the normal equations and gradient descent with NumPy and check that both match scikit-learn. Implementing it yourself teaches what `fit` does. For production use the library version.",
    scratch='''ytr, yva = y_train.values, y_val.values

class LinearRegressionScratch:
    def fit_normal(self, X, y):
        A = np.c_[np.ones(len(X)), X]
        self.w = np.linalg.solve(A.T @ A, A.T @ y)   # normal equations
        return self

    def fit_gd(self, X, y, lr=0.05, steps=1500):
        A = np.c_[np.ones(len(X)), X]
        w = np.zeros(A.shape[1])
        self.history = []
        for _ in range(steps):
            w -= lr * (2 / len(A)) * A.T @ (A @ w - y)
            self.history.append(np.mean((A @ w - y) ** 2))
        self.w = w
        return self

    def predict(self, X):
        return np.c_[np.ones(len(X)), X] @ self.w

normal = LinearRegressionScratch().fit_normal(Xtr, ytr)
descent = LinearRegressionScratch().fit_gd(Xtr, ytr)
reference = LinearRegression().fit(Xtr, ytr)
print("max coefficient gap, normal vs sklearn:", np.abs(normal.w[1:] - reference.coef_).max())
print("max coefficient gap, gradient descent vs sklearn:", np.abs(descent.w[1:] - reference.coef_).max())
print("validation RMSE scratch:", round(float(np.sqrt(np.mean((normal.predict(Xva) - yva) ** 2))), 4),
      "| sklearn:", round(float(np.sqrt(np.mean((reference.predict(Xva) - yva) ** 2))), 4))
plt.plot(descent.history)
plt.yscale("log")
plt.xlabel("Gradient descent step")
plt.ylabel("Training MSE (mpg squared)")
plt.title("Gradient descent convergence")
plt.show()''',
    imports="from sklearn.linear_model import LinearRegression",
    lib_md="`LinearRegression` solves least squares with an SVD based solver. We wrap it in a pipeline so that imputation and scaling are fitted on training rows only.",
    params="| Parameter | Meaning |\n|---|---|\n| `fit_intercept` | Learn $\\beta_0$. Set to `False` only when the target is already centred. |\n| `positive` | Force non negative coefficients. |\n\nOrdinary least squares has no regularisation strength to tune. The validation sweep below compares with and without an intercept to show why the intercept matters.",
    model="LinearRegression()", sweep=("model__fit_intercept", [True, False]), select="RMSE", scoring="neg_root_mean_squared_error",
    metrics_md="Regression metrics: MAE (average absolute error in mpg), RMSE (square root of mean squared error, same unit as mpg, penalises large errors), R-squared (share of variance explained) and MAPE (percentage error, valid because mpg is never zero).",
    interpret_md="Read the permutation importance chart as: how much does the error grow if one feature is shuffled? Because several features are correlated (weight, displacement, cylinders), importance is shared between them and individual coefficients can be misleading. Check the residual plot for curvature, which would mean a straight line is too simple.",
    experiments=[
        dict(title="Collinearity: variance inflation factors", md="Variance inflation factor (VIF) of feature $j$ is $1/(1-R_j^2)$, where $R_j^2$ comes from regressing feature $j$ on the others. Large values mean the coefficient is poorly determined. We then drop the worst feature and compare cross validated RMSE on the training plus validation rows.",
             code=CV_RMSE + '''Z = StandardScaler().fit_transform(X_trainval)
vif = {}
for j, name in enumerate(X_trainval.columns):
    others = np.delete(Z, j, axis=1)
    r2 = LinearRegression().fit(others, Z[:, j]).score(others, Z[:, j])
    vif[name] = 1 / (1 - r2)
vif = pd.Series(vif).sort_values(ascending=False)
display(vif.round(2).to_frame("VIF"))
worst = vif.index[0]
full = cv_rmse(make_model_pipeline(X_trainval, LinearRegression()))
reduced = cv_rmse(make_model_pipeline(X_trainval.drop(columns=[worst]), LinearRegression()), X_trainval.drop(columns=[worst]))
print(f"CV RMSE all features: {full[0]:.3f} +- {full[1]:.3f}")
print(f"CV RMSE without {worst}: {reduced[0]:.3f} +- {reduced[1]:.3f}")'''),
        dict(title="Transform the target", md="Fuel economy is bounded below and right skewed. Modelling $\\log(\\text{mpg})$ turns multiplicative effects into additive ones. We compare cross validated RMSE measured back on the mpg scale.",
             code=CV_RMSE + '''from sklearn.compose import TransformedTargetRegressor
plain = cv_rmse(make_model_pipeline(X_trainval, LinearRegression()))
logged = cv_rmse(TransformedTargetRegressor(make_model_pipeline(X_trainval, LinearRegression()), func=np.log, inverse_func=np.exp))
print(f"CV RMSE raw target: {plain[0]:.3f} +- {plain[1]:.3f}")
print(f"CV RMSE log target: {logged[0]:.3f} +- {logged[1]:.3f}")'''),
        dict(title="Add curvature with squared terms", md="If the residual plot bends, add squared terms. This moves towards polynomial regression (see that notebook).",
             code=CV_RMSE + '''from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
linear = cv_rmse(make_model_pipeline(X_trainval, LinearRegression()))
curved = cv_rmse(make_pipeline(build_preprocessor(X_trainval), PolynomialFeatures(2, include_bias=False), StandardScaler(), LinearRegression()))
print(f"CV RMSE linear: {linear[0]:.3f} +- {linear[1]:.3f}")
print(f"CV RMSE degree 2 terms: {curved[0]:.3f} +- {curved[1]:.3f}")''')],
    exercises=["Fit the model on `model_year >= 76` rows only and test on earlier years. What does this tell you about extrapolation?",
               "Add Huber loss (`HuberRegressor`) and compare the largest residuals.",
               "Compute 95 percent bootstrap intervals for the coefficients with `ml_lab.metrics.bootstrap_interval`."],
    discussion="Hastie et al. present least squares as the base case for linear methods. Our notebook shows the same normal equations numerically. The collinearity experiment connects to the motivation of Hoerl and Kennard for ridge regression.",
    improvements="Use regularised variants (ridge, lasso), model the log target, add interaction terms, or move to tree ensembles if curvature dominates.",
    takeaways=["Least squares has a closed form, and gradient descent approaches the same solution (the small gap is printed in the scratch cell).", "Always compare with a dummy baseline.", "Correlated features make single coefficients unreliable even when predictions are fine."],
    use_cases="Quick baselines, interpretable cost or demand models, and benchmarks for more complex models.",
    limitations="Linear in the parameters, sensitive to outliers, and unstable under collinearity.",
    next="Ridge Regression, which stabilises coefficients under collinearity.",
)
