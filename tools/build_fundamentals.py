"""Build fundamentals notebooks. Synthetic data is used only to explain maths."""
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "00_fundamentals"

PRE = '''import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from ml_lab.data import load_dataset
from ml_lab.utils import set_seed
from ml_lab import visualization as viz
SEED = 42
set_seed(SEED)
viz.apply_style()'''

NB = []


def nb(folder, num, title, tagline, objectives, cells, exercises, takeaways, next_nb):
    NB.append(dict(folder=folder, num=num, title=title, tagline=tagline, objectives=objectives, cells=cells,
                   exercises=exercises, takeaways=takeaways, next=next_nb))


def md(t):
    return ("md", t)


def code(t):
    return ("code", t)


nb("01_python_for_ml", "01", "Python for Machine Learning", "The Python habits that keep ML code readable, testable and reproducible.",
   ["Use functions, dictionaries and comprehensions for data work", "Write small functions with type hints and docstrings", "Set seeds and record the environment", "Time and profile a slow cell"],
   [md("## 1. Functions and data structures\n\nML code is mostly small functions that transform data. Keep each one short and give it one job."),
    code('''def rescale(values, low=0.0, high=1.0):
    """Map values linearly to [low, high]."""
    values = np.asarray(values, dtype=float)
    span = values.max() - values.min()
    return low + (values - values.min()) * (high - low) / span

print(rescale([3, 5, 9]))
config = {"seed": SEED, "test_size": 0.2, "model": "linear"}
print({k: v for k, v in config.items() if k != "model"})'''),
    md("## 2. Reproducibility basics\n\nA result is only useful if someone can run it again. Fix the seed, record package versions and keep settings in one dictionary."),
    code('''from ml_lab.utils import print_environment
print_environment()
a = np.random.default_rng(SEED).normal(size=3)
b = np.random.default_rng(SEED).normal(size=3)
print("same seed, same numbers:", bool(np.allclose(a, b)))'''),
    md("## 3. Vectorised code is faster than loops\n\nWe time the same sum of squares with a Python loop and with NumPy on one million numbers."),
    code('''import time
x = np.random.default_rng(SEED).normal(size=1_000_000)
t = time.perf_counter(); s1 = sum(v * v for v in x); loop = time.perf_counter() - t
t = time.perf_counter(); s2 = float(np.dot(x, x)); vec = time.perf_counter() - t
print(f"loop {loop:.3f} s, numpy {vec:.4f} s, ratio {loop / vec:.0f}x, same result: {abs(s1 - s2) < 1e-6 * s2}")'''),
    md("## 4. Testing a small function\n\nAn `assert` is the smallest test. The repository runs real tests with `pytest` (see `tests/`)."),
    code('''assert np.allclose(rescale([1, 2, 3]), [0, 0.5, 1])
assert rescale([10, 20], 0, 100).tolist() == [0.0, 100.0]
print("tests passed")''')],
   ["Write a function `zscore(values)` and test it against `sklearn.preprocessing.StandardScaler`.", "Time the loop and NumPy versions for 10 000 and 10 000 000 numbers and describe how the ratio changes."],
   ["Short functions are easier to test.", "Seeds and environment records make runs repeatable.", "Vectorise before optimising."], "NumPy and pandas")

nb("02_numpy_and_pandas", "02", "NumPy and pandas", "Arrays, tables, missing values and grouping on a real dataset.",
   ["Index, slice and broadcast arrays", "Select, filter and group a DataFrame", "Find and handle missing values", "Avoid common copy and view mistakes"],
   [md("## 1. Arrays and broadcasting\n\nBroadcasting lets arrays of different shapes combine without loops."),
    code('''A = np.arange(12).reshape(3, 4)
print(A, "\\nshape", A.shape)
print("column means:", A.mean(axis=0), "| row sums:", A.sum(axis=1))
print("centre columns by broadcasting:\\n", A - A.mean(axis=0))'''),
    md("## 2. A real table: Auto MPG\n\nThe dataset registry loads real data with a licence note."),
    code('''bundle = load_dataset("auto_mpg")
df = bundle.frame
display(df.head())
print(df.dtypes.to_dict())
print("missing values per column:", df.isna().sum().to_dict())'''),
    md("## 3. Select, filter, group"),
    code('''heavy = df[df["weight"] > df["weight"].median()]
print("heavy cars:", len(heavy), "| mean mpg heavy:", round(heavy["mpg"].mean(), 1), "| mean mpg light:", round(df[df["weight"] <= df["weight"].median()]["mpg"].mean(), 1))
display(df.groupby("origin")[["mpg", "horsepower", "weight"]].agg(["mean", "std"]).round(1))'''),
    md("## 4. Missing values\n\nWe poke holes into a copy to practise. Real imputation inside a pipeline is in the preprocessing module."),
    code('''holey = df.copy()
rng = np.random.default_rng(SEED)
holey.loc[rng.choice(holey.index, 30, replace=False), "horsepower"] = np.nan
print("missing:", int(holey["horsepower"].isna().sum()))
print("mean imputation changes the std from", round(df["horsepower"].std(), 1), "to", round(holey["horsepower"].fillna(holey["horsepower"].mean()).std(), 1))'''),
    md("## 5. Views and copies\n\nChanging a slice may or may not change the original. Use `.copy()` when you mean a copy."),
    code('''v = A[:, :2]
v[0, 0] = 99
print("original changed through a view:", A[0, 0] == 99)
c = A[:, :2].copy(); c[0, 0] = -1
print("original changed through a copy:", A[0, 0] == -1)''')],
   ["Compute the correlation of `mpg` with every other column and sort it.", "Show that median imputation changes the standard deviation less than a constant fill of 0."],
   ["Broadcasting replaces loops.", "Group and aggregate before modelling.", "Imputation changes distributions."], "Data visualization")

nb("03_data_visualization", "03", "Data Visualization", "Choose a plot for the question, then make it readable.",
   ["Match plot types to questions", "Label axes with units", "Read distributions, relationships and comparisons", "Avoid misleading axes"],
   [md("## 1. Distribution: histogram and box plot"),
    code('''df = load_dataset("auto_mpg").frame
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].hist(df["mpg"], bins=20); axes[0].set_xlabel("Fuel economy (miles per gallon)"); axes[0].set_ylabel("Cars"); axes[0].set_title("Distribution of mpg")
df.boxplot(column="mpg", by="origin", ax=axes[1]); axes[1].set_xlabel("Origin (1 US, 2 Europe, 3 Japan)"); axes[1].set_ylabel("mpg"); axes[1].set_title("mpg by origin"); fig.suptitle("")
plt.tight_layout(); plt.show()'''),
    md("## 2. Relationship: scatter plot with a trend line"),
    code('''fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(df["weight"], df["mpg"], s=12, alpha=0.6)
coef = np.polyfit(df["weight"], df["mpg"], 1)
xs = np.linspace(df["weight"].min(), df["weight"].max(), 50)
ax.plot(xs, np.polyval(coef, xs), color="red"); ax.set_xlabel("Weight (lb)"); ax.set_ylabel("mpg"); ax.set_title("Heavier cars use more fuel")
plt.show()'''),
    md("## 3. Correlation heatmap"),
    code('''corr = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=60, ha="right"); ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
plt.colorbar(im, label="Pearson correlation"); ax.set_title("Correlation matrix"); plt.show()'''),
    md("## 4. A misleading axis\n\nThe same two bars look very different when the axis does not start at zero."),
    code('''fig, axes = plt.subplots(1, 2, figsize=(8, 3))
vals = [df[df.origin == 1]["mpg"].mean(), df[df.origin == 3]["mpg"].mean()]
for ax, lim in zip(axes, [(0, 35), (19, 32)]):
    ax.bar(["US", "Japan"], vals); ax.set_ylim(*lim); ax.set_ylabel("Mean mpg"); ax.set_title(f"axis from {lim[0]}")
plt.tight_layout(); plt.show()''')],
   ["Make a violin plot of `horsepower` by `cylinders` and describe what it adds to a box plot.", "Find a pair of columns whose scatter plot shows a curve and plot it."],
   ["Pick the plot for the question.", "Always label units.", "Start bar axes at zero."], "Linear algebra")

nb("04_linear_algebra", "04", "Linear Algebra for ML", "Vectors, matrices, projections and the SVD, which sit under regression and PCA.",
   ["Compute dot products, norms and matrix products", "Solve a least squares problem with the normal equation", "See what the SVD does", "Connect these to regression and PCA"],
   [md("## 1. Vectors and matrices\n\nThe dot product measures alignment: $a\\cdot b = \\lVert a\\rVert \\lVert b\\rVert \\cos\\theta$."),
    code('''a, b = np.array([3.0, 4.0]), np.array([4.0, 3.0])
cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
print("norm of a:", np.linalg.norm(a), "| cosine:", round(float(cos), 3), "| angle (degrees):", round(float(np.degrees(np.arccos(cos))), 1))'''),
    md("## 2. Least squares\n\nLinear regression finds $\\beta$ that minimises $\\lVert y - X\\beta\\rVert^2$. The normal equation is $X^\\top X\\beta = X^\\top y$. We use the real Auto MPG data."),
    code('''df = load_dataset("auto_mpg").frame
X = np.c_[np.ones(len(df)), df[["weight", "horsepower"]].values]
y = df["mpg"].values
beta = np.linalg.solve(X.T @ X, X.T @ y)
print("coefficients (intercept, weight, horsepower):", np.round(beta, 4))
from sklearn.linear_model import LinearRegression
ref = LinearRegression().fit(df[["weight", "horsepower"]], y)
print("sklearn:", round(float(ref.intercept_), 4), np.round(ref.coef_, 4))'''),
    md("## 3. The SVD\n\nAny matrix factors as $A = U S V^\\top$. A low rank approximation keeps the largest singular values. We use the 8 by 8 digit images."),
    code('''digits = load_dataset("digits").X.values
U, S, Vt = np.linalg.svd(digits - digits.mean(axis=0), full_matrices=False)
for k in [2, 5, 10, 20]:
    approx = (U[:, :k] * S[:k]) @ Vt[:k]
    err = np.linalg.norm(digits - digits.mean(axis=0) - approx) / np.linalg.norm(digits - digits.mean(axis=0))
    print(f"rank {k}: relative error {err:.3f}")'''),
    md("## 4. A picture of rank 10"),
    code('''k = 10
approx = (U[:, :k] * S[:k]) @ Vt[:k] + digits.mean(axis=0)
fig, axes = plt.subplots(2, 6, figsize=(9, 3))
for i in range(6):
    axes[0, i].imshow(digits[i].reshape(8, 8), cmap="gray_r"); axes[0, i].axis("off")
    axes[1, i].imshow(approx[i].reshape(8, 8), cmap="gray_r"); axes[1, i].axis("off")
axes[0, 0].set_title("original", fontsize=8); axes[1, 0].set_title("rank 10", fontsize=8); plt.show()''')],
   ["Show that the normal equation fails when two columns are identical, and fix it with `np.linalg.lstsq`.", "Compute the condition number of `X.T @ X` for the Auto MPG design matrix before and after standardising."],
   ["Regression is a projection problem.", "SVD gives the best low rank approximation.", "Scale features to keep matrices well conditioned."], "Probability and statistics")

nb("05_probability_and_statistics", "05", "Probability and Statistics for ML", "Distributions, sampling error, confidence intervals and a permutation test.",
   ["Distinguish population and sample", "Build a bootstrap confidence interval", "Run a permutation test", "Explain why one test split is a noisy estimate"],
   [md("## 1. Sampling error\n\nWe draw samples of different sizes from the Auto MPG column and watch the mean settle. Treating the 392 cars as the population is a simplification for teaching."),
    code('''mpg = load_dataset("auto_mpg").frame["mpg"].values
rng = np.random.default_rng(SEED)
for n in [5, 20, 80, 320]:
    means = [rng.choice(mpg, n, replace=False).mean() for _ in range(500)]
    print(f"n={n}: std of the sample mean {np.std(means):.3f} (theory sigma/sqrt(n) = {mpg.std() / np.sqrt(n):.3f})")'''),
    md("## 2. Bootstrap interval for a mean"),
    code('''boot = [rng.choice(mpg, len(mpg), replace=True).mean() for _ in range(2000)]
print("mean mpg:", round(float(mpg.mean()), 2), "| 95% bootstrap interval:", np.round(np.percentile(boot, [2.5, 97.5]), 2))'''),
    md("## 3. Permutation test\n\nAre US cars (origin 1) different in mpg from Japanese cars (origin 3)? We shuffle the group labels many times and see how often a difference this large appears by chance."),
    code('''df = load_dataset("auto_mpg").frame
g1, g3 = df[df.origin == 1]["mpg"].values, df[df.origin == 3]["mpg"].values
obs = g3.mean() - g1.mean()
pool = np.r_[g1, g3]
diffs = []
for _ in range(5000):
    rng.shuffle(pool)
    diffs.append(pool[len(g1):].mean() - pool[:len(g1)].mean())
print("observed difference:", round(float(obs), 2), "| share of shuffles at least as extreme:", float(np.mean(np.abs(diffs) >= abs(obs))))'''),
    md("## 4. Why one test split is noisy\n\nThe same model scored on 30 different random splits gives a spread of test scores."),
    code('''from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
X, y = df[["weight", "horsepower", "model_year"]], df["mpg"]
rmses = []
for s in range(30):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=s)
    rmses.append(mean_squared_error(yte, LinearRegression().fit(Xtr, ytr).predict(Xte)) ** 0.5)
print("test RMSE over 30 splits: mean", round(float(np.mean(rmses)), 2), "| std", round(float(np.std(rmses, ddof=1)), 2), "| range", round(min(rmses), 2), "to", round(max(rmses), 2))''')],
   ["Repeat the permutation test for mpg of Europe versus Japan and compare p values.", "Plot the bootstrap distribution of the median."],
   ["Sample statistics have spread.", "Bootstrap and permutation tests need few assumptions.", "Report spread, not one number."], "Calculus and optimisation")

nb("06_calculus_and_optimization", "06", "Calculus and Optimisation", "Gradients and gradient descent, written by hand and checked numerically.",
   ["Take derivatives of a loss", "Check gradients numerically", "Run gradient descent and see the effect of the step size", "Understand why feature scaling speeds convergence"],
   [md("## 1. The loss and its gradient\n\nFor linear regression $L(\\beta) = \\frac1n\\lVert y - X\\beta\\rVert^2$ and $\\nabla L = -\\frac2n X^\\top(y - X\\beta)$. We check the formula with finite differences."),
    code('''df = load_dataset("auto_mpg").frame
X = (df[["weight", "horsepower"]].values - df[["weight", "horsepower"]].values.mean(axis=0)) / df[["weight", "horsepower"]].values.std(axis=0)
X = np.c_[np.ones(len(X)), X]; y = df["mpg"].values
loss = lambda b: np.mean((y - X @ b) ** 2)
grad = lambda b: -2 / len(y) * X.T @ (y - X @ b)
b0 = np.array([1.0, 2.0, -1.0])
num = np.array([(loss(b0 + 1e-6 * e) - loss(b0 - 1e-6 * e)) / 2e-6 for e in np.eye(3)])
print("analytic:", np.round(grad(b0), 4), "| numeric:", np.round(num, 4))'''),
    md("## 2. Gradient descent and the step size"),
    code('''def descend(lr, steps=200):
    b, hist = np.zeros(3), []
    for _ in range(steps):
        b = b - lr * grad(b); hist.append(loss(b))
    return b, hist
exact = np.linalg.solve(X.T @ X, X.T @ y)
fig, ax = plt.subplots(figsize=(6, 3.5))
for lr in [0.001, 0.05, 0.4]:
    b, h = descend(lr); ax.plot(h, label=f"lr={lr}")
    print(f"lr={lr}: final loss {h[-1]:.3f}, distance to exact solution {np.linalg.norm(b - exact):.4f}")
ax.set_yscale("log"); ax.set_xlabel("Step"); ax.set_ylabel("Mean squared error"); ax.legend(); ax.set_title("Step size effect"); plt.show()'''),
    md("## 3. Scaling and conditioning\n\nWithout scaling the loss surface is a long narrow valley and gradient descent needs a tiny step size."),
    code('''Xr = np.c_[np.ones(len(df)), df[["weight", "horsepower"]].values]
for name, M in [("scaled", X), ("raw", Xr)]:
    print(name, "condition number of X'X:", f"{np.linalg.cond(M.T @ M):.1f}")'''),
    md("## 4. Beyond gradient descent: Newton step on a simple function"),
    code('''f = lambda x: (x - 3) ** 2 + np.sin(5 * x) * 0.2
x = 0.0
for i in range(10):
    h = 1e-4
    g = (f(x + h) - f(x - h)) / (2 * h); H = (f(x + h) - 2 * f(x) + f(x - h)) / h ** 2
    x = x - g / H
print("minimiser found by Newton steps:", round(float(x), 4), "| f:", round(float(f(x)), 4))''')],
   ["Add momentum to `descend` and compare the loss after 50 steps.", "Show with a plot that the raw features need a learning rate below 1e-7 to avoid divergence."],
   ["Gradients are checkable.", "Step size decides stability.", "Scaling improves conditioning."], "Machine learning concepts")

nb("07_machine_learning_concepts", "07", "Machine Learning Concepts", "Overfitting, the bias variance trade-off, data leakage and validation.",
   ["Detect overfitting with train and validation scores", "Run a validation curve", "See data leakage inflate a score", "Choose between hold out and cross validation"],
   [md("## 1. Overfitting\n\nWe fit polynomials of growing degree on a noisy sample of Auto MPG and compare training and validation error."),
    code('''from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
df = load_dataset("auto_mpg").frame
Xtr, Xva, ytr, yva = train_test_split(df[["horsepower"]].values, df["mpg"].values, test_size=0.3, random_state=SEED)
rows = []
for d in [1, 2, 3, 5, 8, 12]:
    m = make_pipeline(StandardScaler(), PolynomialFeatures(d), LinearRegression()).fit(Xtr, ytr)
    rows.append({"degree": d, "train_rmse": mean_squared_error(ytr, m.predict(Xtr)) ** 0.5, "validation_rmse": mean_squared_error(yva, m.predict(Xva)) ** 0.5})
display(pd.DataFrame(rows).set_index("degree").round(3))'''),
    md("## 2. Data leakage\n\nWe use the breast cancer data with 500 noise features. Selecting the best features on all rows before splitting makes pure noise look predictive."),
    code('''from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
rng = np.random.default_rng(SEED)
Xn, yn = rng.normal(size=(100, 500)), rng.integers(0, 2, 100)      # labels are random
leak = SelectKBest(f_classif, k=10).fit_transform(Xn, yn)
print("leaky: select features on all rows, then cross validate:", round(cross_val_score(LogisticRegression(), leak, yn, cv=5).mean(), 3))
pipe = make_pipeline(SelectKBest(f_classif, k=10), LogisticRegression())
print("correct: selection inside each fold:", round(cross_val_score(pipe, Xn, yn, cv=5).mean(), 3), "(chance is about 0.5)")'''),
    md("## 3. Learning curve\n\nMore data helps only if the model has not reached its limit."),
    code('''from sklearn.model_selection import learning_curve
bc = load_dataset("breast_cancer")
sizes, tr, va = learning_curve(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), bc.X, bc.y, cv=5, train_sizes=np.linspace(0.1, 1, 6), random_state=SEED)
fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.plot(sizes, tr.mean(axis=1), marker="o", label="train"); ax.plot(sizes, va.mean(axis=1), marker="o", label="validation")
ax.set_xlabel("Training rows"); ax.set_ylabel("Accuracy"); ax.legend(); ax.set_title("Learning curve"); plt.show()'''),
    md("## 4. Hold out versus cross validation")
    , code('''from sklearn.model_selection import KFold
X, y = df[["weight", "horsepower", "model_year"]], df["mpg"]
for k in [3, 5, 10]:
    s = cross_val_score(LinearRegression(), X, y, cv=KFold(k, shuffle=True, random_state=SEED), scoring="neg_root_mean_squared_error")
    print(f"{k}-fold RMSE {-s.mean():.3f} +- {s.std(ddof=1):.3f}")''')],
   ["Repeat the leakage demo with 5000 features and report how the leaky score changes.", "Plot a validation curve for k in KNN on the wine data."],
   ["Training error alone misleads.", "Leakage can make noise look like signal.", "Cross validation reports spread."], "Feature engineering")

nb("08_feature_engineering", "08", "Feature Engineering", "Scaling, encoding, interactions and a leakage safe pipeline.",
   ["Scale and encode features inside a pipeline", "Create interaction and transformed features", "Compare feature sets with cross validation", "Avoid target leakage in encodings"],
   [md("## 1. A pipeline for mixed columns\n\nTelco churn has numeric and categorical columns and missing values. The preprocessor is fitted on training rows only."),
    code('''from ml_lab.preprocessing import split_three_way, build_preprocessor
bundle = load_dataset("telco_churn")
X, y = bundle.X, bundle.y
Xtr, Xva, Xte, ytr, yva, yte = split_three_way(X, y, task="classification", seed=SEED)
prep = build_preprocessor(Xtr)
Z = prep.fit_transform(Xtr)
print("raw columns:", X.shape[1], "| after encoding:", Z.shape[1])
print("missing values after imputation:", int(np.isnan(Z).sum()))'''),
    md("## 2. Transformations\n\nA skewed feature often works better after a log transform."),
    code('''from scipy.stats import skew
df = load_dataset("auto_mpg").frame
print("skew of displacement:", round(float(skew(df["displacement"])), 2), "| after log:", round(float(skew(np.log(df["displacement"]))), 2))'''),
    md("## 3. Does an engineered feature help?\n\nWe compare cross validated RMSE with and without `weight / horsepower` and `log(weight)`."),
    code('''from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
base = df[["weight", "horsepower", "model_year", "cylinders"]]
eng = base.assign(power_to_weight=df["horsepower"] / df["weight"], log_weight=np.log(df["weight"]))
for name, X_ in [("base", base), ("engineered", eng)]:
    s = cross_val_score(make_pipeline(StandardScaler(), LinearRegression()), X_, df["mpg"], cv=KFold(5, shuffle=True, random_state=SEED), scoring="neg_root_mean_squared_error")
    print(f"{name}: CV RMSE {-s.mean():.3f} +- {s.std(ddof=1):.3f}")'''),
    md("## 4. Target leakage in encoding\n\nEncoding a category with the mean of the target computed on all rows leaks the label. Inside cross validation use the training fold only. Here we use a high cardinality random column to make the effect visible.")
    , code('''rng = np.random.default_rng(SEED)
ids = rng.integers(0, 150, len(df))                       # random category, no signal
y = df["mpg"].values
leak_enc = pd.Series(y).groupby(ids).transform("mean").values
from sklearn.model_selection import cross_val_predict
pred = cross_val_predict(LinearRegression(), leak_enc.reshape(-1, 1), y, cv=5)
print("R2 of a random category with leaky target encoding:", round(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum(), 3), "(the true signal is zero)")''')],
   ["Add `model_year` as a categorical column and compare with the numeric version.", "Implement out-of-fold target encoding for `ids` and show that R2 falls to about zero."],
   ["Fit transformers on training rows only.", "Engineered features need cross validated proof.", "Target encoding must be out of fold."], "Supervised learning: linear regression")


def render(spec):
    folder = BASE / spec["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    cells = [new_markdown_cell(f"# {spec['title']}\n\n**Family:** Fundamentals | **Level:** Beginner | **Runs on:** CPU, under a minute\n\n{spec['tagline']}"),
             new_markdown_cell("## Learning objectives\n\n" + "\n".join(f"- {o}" for o in spec["objectives"])),
             new_code_cell(PRE)]
    for kind, text in spec["cells"]:
        cells.append(new_markdown_cell(text) if kind == "md" else new_code_cell(text))
    cells.append(new_markdown_cell("## Exercises\n\n" + "\n".join(f"{i}. {e}" for i, e in enumerate(spec["exercises"], 1))))
    cells.append(new_markdown_cell("## Summary\n\n" + "\n".join(f"- {t}" for t in spec["takeaways"]) + f"\n\n**Next:** {spec['next']}.\n\n"
                                   "**Data note.** Real datasets come from `ml_lab.data` with licence notes in `datasets/catalog`. Synthetic numbers appear only where a demonstration needs controlled data, and are labelled."))
    nbk = new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}})
    nbformat.validate(nbk)
    path = folder / f"{spec['num']}_{spec['folder'][3:]}.ipynb"
    nbformat.write(nbk, path)
    (folder / "README.md").write_text(f"# {spec['title']}\n\n{spec['tagline']}\n\nNotebook: [`{path.name}`]({path.name})\n\n## Objectives\n\n" + "\n".join(f"- {o}" for o in spec["objectives"]) + "\n")
    print("built", path.relative_to(ROOT))


for s in NB:
    render(s)
