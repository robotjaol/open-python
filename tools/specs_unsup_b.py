"""Unsupervised specs, part B."""

from specs_common import DS_BC, DS_DIGITS
from specs_unsup_a import UN, un

CL, DR, AN = "01_clustering", "02_dimensionality_reduction", "03_anomaly_detection"

un("06", CL, id="spectral_clustering", title="Spectral Clustering: Graph Cuts on Digit Images", level="Advanced",
   tagline="Cluster with the eigenvectors of a neighbour graph.", name="SpectralClustering", dataset="digits",
   objectives=["How a similarity graph defines clusters", "What the graph Laplacian eigenvectors do", "How the eigengap suggests k", "Why spectral methods need a good graph"],
   problem=dict(context="Digits of the same class lie on curved structures, not round balls.",
                problem="Cluster digit images with a nearest neighbour graph.",
                matters="Graph methods find non convex groups.",
                why_ml="Similarity is local, so a global centre model fails.",
                why_algo="Spectral clustering turns local similarity into global groups."),
   papers=["vonluxburg2007"],
   paper_relation="von Luxburg's tutorial explains the algorithm and its variants. We implement the normalised symmetric version.",
   math=r"""**Graph.** Affinity matrix $W$ from k nearest neighbours.

**Normalised Laplacian.** $L_{sym} = I - D^{-1/2} W D^{-1/2}$ with degree matrix $D$.

**Algorithm.** Take the $k$ eigenvectors of $L_{sym}$ with smallest eigenvalues, normalise rows to unit length, and run k-means on the rows.

**Eigengap.** A large jump between consecutive eigenvalues suggests the number of clusters.

**Limits.** Needs a good graph, $O(n^3)$ eigen decomposition for dense solvers, no out of sample prediction.""",
   dataset_md=DS_DIGITS, prep_md="We project to 20 principal components and build a 15 nearest neighbour graph. No labels are used.",
   prep='''X = PCA(20, random_state=SEED).fit_transform(X_raw.values)
print(X.shape)''',
   concept="Points connected through chains of neighbours end up close in the eigenvector space, where a simple k-means can split them.",
   scratch_md="We build the normalised Laplacian from a kNN graph, take eigenvectors and run k-means.",
   scratch='''from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans, SpectralClustering
W = kneighbors_graph(X, 15, include_self=False)
W = 0.5 * (W + W.T)
W = W.toarray()
deg = W.sum(axis=1)
Dm = np.diag(1 / np.sqrt(deg))
L = np.eye(len(W)) - Dm @ W @ Dm
vals, vecs = np.linalg.eigh(L)
print("smallest 12 eigenvalues:", np.round(vals[:12], 4))
def spectral_scratch(k):
    U = vecs[:, :k]
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    return KMeans(k, n_init=10, random_state=SEED).fit_predict(U)
mine = spectral_scratch(10)
ref = SpectralClustering(10, affinity="nearest_neighbors", n_neighbors=15, random_state=SEED).fit_predict(X)
print("ARI scratch vs reference digits:", round(external_scores(y_true, mine)["ARI"], 3), "| sklearn:", round(external_scores(y_true, ref)["ARI"], 3))''',
   params="| Parameter | Meaning |\n|---|---|\n| `n_clusters` | Groups. |\n| `affinity` | `nearest_neighbors` or `rbf`. |\n| `n_neighbors` | Graph size parameter. |\n| `assign_labels` | `kmeans` or `discretize`. |",
   train_md="We choose k from the eigengap and check the choice with silhouette. The reference digit label is not used for selection.",
   train='''gaps = np.diff(vals[:20])
K_GAP = int(np.argmax(gaps[1:]) + 2)
print("largest eigengap suggests k =", K_GAP)
rows = []
for k in [5, 8, 10, 12, 15]:
    lab = spectral_scratch(k)
    rows.append({"k": k, "silhouette": internal_scores(X, lab)["Silhouette"], "eigengap": float(vals[k] - vals[k - 1])})
sweep = pd.DataFrame(rows).set_index("k")
display(sweep.round(4))
BEST_K = int(sweep["silhouette"].idxmax())
model = SpectralClustering(BEST_K, affinity="nearest_neighbors", n_neighbors=15, random_state=SEED)
labels = model.fit_predict(X)
print("chosen k:", BEST_K)''',
   eval='''METRICS = {**internal_scores(X, labels), **external_scores(y_true, labels)}
CONFIG = {"k": BEST_K, "n_neighbors": 15, "pca_components": 20}
EXTRA = {"eigengap_k": K_GAP, "sweep": sweep.round(4).reset_index().to_dict("records")}''',
   viz='''fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(range(1, 21), vals[:20], marker="o"); axes[0].set_xlabel("Index"); axes[0].set_ylabel("Eigenvalue"); axes[0].set_title("Laplacian spectrum")
Z = PCA(2, random_state=SEED).fit_transform(X)
axes[1].scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", s=10); axes[1].set_title("Spectral clusters")
plt.tight_layout(); save_figure(fig, NOTEBOOK_ID, "overview"); plt.show()''',
   interpret_md="The crosstab shows which digits share a cluster.",
   interpret='''display(pd.crosstab(pd.Series(labels, name="cluster"), pd.Series(y_true, name="reference digit")))''',
   experiments=[
       dict(title="Neighbours in the graph", md="Too few neighbours disconnect the graph, too many blur it.",
            code='''for nn in [5, 15, 30, 60]:
    lab = SpectralClustering(BEST_K, affinity="nearest_neighbors", n_neighbors=nn, random_state=SEED).fit_predict(X)
    print(f"n_neighbors={nn}: ARI {external_scores(y_true, lab)['ARI']:.3f}, silhouette {internal_scores(X, lab)['Silhouette']:.3f}")'''),
       dict(title="Spectral versus k-means", md="Same k, same input.",
            code='''km = KMeans(BEST_K, n_init=10, random_state=SEED).fit_predict(X)
print("ARI k-means:", round(external_scores(y_true, km)["ARI"], 3), "| ARI spectral:", round(METRICS["ARI"], 3))'''),
       dict(title="RBF affinity", md="A dense Gaussian kernel affinity instead of a kNN graph.",
            code='''for gamma in [0.001, 0.005, 0.02]:
    lab = SpectralClustering(BEST_K, affinity="rbf", gamma=gamma, random_state=SEED).fit_predict(X)
    print(f"gamma={gamma}: ARI {external_scores(y_true, lab)['ARI']:.3f}")''')],
   discussion="von Luxburg stresses that the quality depends on the graph. The neighbour experiment shows this on digit data.",
   improvements="Use self tuning affinities and Nystrom approximation for large data.",
   takeaways=["The graph decides the result.", "Eigengap guides k.", "No prediction for new rows."],
   use_cases="Image segmentation, community detection.", limitations="Cubic cost, graph sensitivity.", next="PCA.")

un("07", DR, id="pca", title="PCA: Compressing Breast Cancer Measurements", level="Beginner",
   tagline="Find the directions of largest variance and keep the few that matter.", name="PCA", dataset="breast_cancer", task="dimensionality_reduction",
   objectives=["How PCA relates to the SVD", "How to read explained variance", "Why scaling decides the result", "How to fit PCA without leakage"],
   problem=dict(context="Thirty highly correlated measurements describe each tumour.",
                problem="Compress 30 measurements into a few components that keep most variance.",
                matters="Fewer, uncorrelated features simplify models and plots.",
                why_ml="Correlated columns carry the same information several times.",
                why_algo="PCA is the standard linear compression."),
   papers=["pearson1901"],
   paper_relation="Pearson posed the problem of the line or plane of closest fit. Modern PCA computes this with the SVD.",
   math=r"""**Centre** $X$, then $X = U S V^\top$ (SVD).

**Components** are rows of $V^\top$. Scores are $XV$. Variance along component $j$ is $s_j^2/(n-1)$.

**Explained variance ratio** $= s_j^2 / \sum_i s_i^2$.

**Reconstruction.** With $m$ components $\hat X = X V_m V_m^\top$ and the error is the sum of the dropped $s_j^2$.

**Limits.** Linear only, sensitive to scale, components are hard to interpret.""",
   dataset_md=DS_BC, prep_md="Split first: 70 percent of rows to fit the scaler and PCA, 30 percent held out to measure reconstruction error. This is how PCA must be used inside a prediction pipeline.",
   prep='''from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y_true, test_size=0.3, stratify=y_true, random_state=SEED)
scaler = StandardScaler().fit(X_tr)
Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)
print(Xs_tr.shape, Xs_te.shape)''',
   concept="The first component is the direction with the most variance, the second is the best direction orthogonal to it, and so on.",
   scratch_md="PCA from the SVD in four lines.",
   scratch='''Xc = Xs_tr - Xs_tr.mean(axis=0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
var_ratio_mine = S ** 2 / (S ** 2).sum()
ref = PCA().fit(Xs_tr)
print("max difference of explained variance ratio vs sklearn:", float(np.abs(var_ratio_mine - ref.explained_variance_ratio_).max()))
print("first component agrees up to sign:", bool(np.allclose(np.abs(Vt[0]), np.abs(ref.components_[0]))))''',
   params="| Parameter | Meaning |\n|---|---|\n| `n_components` | Count or variance share to keep. |\n| `whiten` | Scale components to unit variance. |\n| `svd_solver` | Solver. |",
   train_md="PCA is fitted on training rows only. The number of components is the smallest that explains 95 percent of the training variance.",
   train='''pca = PCA(random_state=SEED).fit(Xs_tr)
cum = np.cumsum(pca.explained_variance_ratio_)
N_COMP = int(np.searchsorted(cum, 0.95) + 1)
print("components for 95 percent variance:", N_COMP, "| for 99 percent:", int(np.searchsorted(cum, 0.99) + 1))
pca_k = PCA(N_COMP, random_state=SEED).fit(Xs_tr)
Z_tr, Z_te = pca_k.transform(Xs_tr), pca_k.transform(Xs_te)''',
   eval_md="Held out reconstruction error (mean squared error per feature, in standardised units) shows what the compression loses on new rows. A logistic regression on the components shows whether the label information survives. This label use is an evaluation of the compression, not part of PCA.",
   eval='''from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
recon = pca_k.inverse_transform(Z_te)
mse = float(((Xs_te - recon) ** 2).mean())
full = f1_score(y_te, LogisticRegression(max_iter=2000).fit(Xs_tr, y_tr).predict(Xs_te))
comp = f1_score(y_te, LogisticRegression(max_iter=2000).fit(Z_tr, y_tr).predict(Z_te))
METRICS = {"components": N_COMP, "variance_explained": float(cum[N_COMP - 1]), "heldout_reconstruction_MSE": mse, "F1_full_features": float(full), "F1_on_components": float(comp)}
CONFIG = {"variance_target": 0.95, "scaled": True}
EXTRA = {"cumulative_variance": [float(v) for v in cum[:10]]}''',
   viz='''fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].bar(range(1, 11), pca.explained_variance_ratio_[:10]); axes[0].set_xlabel("Component"); axes[0].set_ylabel("Explained variance ratio"); axes[0].set_title("Scree plot")
axes[1].plot(range(1, 31), cum, marker="o"); axes[1].axhline(0.95, color="red", linestyle="--"); axes[1].set_xlabel("Components"); axes[1].set_ylabel("Cumulative variance"); axes[1].set_title("Cumulative variance")
axes[2].scatter(Z_te[:, 0], Z_te[:, 1], c=y_te, cmap="coolwarm", s=18); axes[2].set_xlabel("PC1"); axes[2].set_ylabel("PC2"); axes[2].set_title("Held out rows, colour = diagnosis")
plt.tight_layout(); save_figure(fig, NOTEBOOK_ID, "overview"); plt.show()''',
   interpret_md="Loadings show which original measurements make up a component.",
   interpret='''load = pd.DataFrame(pca.components_[:2].T, index=X_raw.columns, columns=["PC1", "PC2"])
display(load.reindex(load.PC1.abs().sort_values(ascending=False).index).head(8).round(3))''',
   experiments=[
       dict(title="Scaled versus raw data", md="Without scaling the largest scale feature dominates.",
            code='''raw = PCA().fit(X_tr.values)
print("variance share of PC1, raw:", round(float(raw.explained_variance_ratio_[0]), 3), "| scaled:", round(float(pca.explained_variance_ratio_[0]), 3))
print("largest loading raw PC1:", X_raw.columns[np.abs(raw.components_[0]).argmax()])'''),
       dict(title="Reconstruction error against components", md="Held out error as components increase.",
            code='''for m in [1, 2, 5, 10, 20, 30]:
    p = PCA(m, random_state=SEED).fit(Xs_tr)
    print(m, round(float(((Xs_te - p.inverse_transform(p.transform(Xs_te))) ** 2).mean()), 4))'''),
       dict(title="Leakage check", md="Fitting PCA on all rows including the test rows makes the held out error look better.",
            code='''leaky = PCA(5, random_state=SEED).fit(np.vstack([Xs_tr, Xs_te]))
clean = PCA(5, random_state=SEED).fit(Xs_tr)
for name, p in [("fit on train only", clean), ("fit on train and test", leaky)]:
    print(name, round(float(((Xs_te - p.inverse_transform(p.transform(Xs_te))) ** 2).mean()), 4))''')],
   discussion="Pearson's idea of closest fit corresponds to the reconstruction error computed here. The leakage experiment shows the size of the optimistic bias on this data.",
   improvements="Use cross validated component selection, sparse PCA for readable loadings.",
   takeaways=["Scale first.", "Fit on training rows only.", "Components are directions, not features."],
   use_cases="Compression, denoising, visualisation, preprocessing.", limitations="Linear, hard to interpret.", next="t-SNE.")

un("08", DR, id="tsne", title="t-SNE: Mapping Digit Images in 2D", level="Intermediate",
   tagline="Embed high dimensional data so that neighbours stay neighbours.", name="TSNE", dataset="digits",
   objectives=["How t-SNE matches neighbour probabilities", "What perplexity means", "Why distances between clusters are not meaningful", "How to measure embedding quality"],
   problem=dict(context="Analysts want to look at high dimensional data in a plot.",
                problem="Show the structure of 1797 digit images in two dimensions.",
                matters="A good map reveals groups and mistakes before modelling.",
                why_ml="Raw pixels cannot be plotted.",
                why_algo="t-SNE preserves local neighbourhoods better than PCA."),
   papers=["vandermaaten2008"],
   paper_relation="van der Maaten and Hinton introduce t-SNE. That entry is not verified in the build environment, so check it before citing.",
   math=r"""**High dimensional affinities.** $p_{j|i} = \dfrac{\exp(-\lVert x_i-x_j\rVert^2/2\sigma_i^2)}{\sum_{k\ne i}\exp(-\lVert x_i-x_k\rVert^2/2\sigma_i^2)}$ with $\sigma_i$ set so that the perplexity $2^{H(P_i)}$ matches the target. Then $p_{ij} = (p_{j|i}+p_{i|j})/2n$.

**Low dimensional affinities.** Student t with one degree of freedom: $q_{ij} = \dfrac{(1+\lVert y_i-y_j\rVert^2)^{-1}}{\sum_{k\ne l}(1+\lVert y_k-y_l\rVert^2)^{-1}}$.

**Objective.** Minimise $KL(P\Vert Q) = \sum_{i\ne j} p_{ij}\log(p_{ij}/q_{ij})$ by gradient descent.

**Limits.** Non convex, cluster sizes and gaps are not reliable, no out of sample transform.""",
   dataset_md=DS_DIGITS, prep_md="Pixels are scaled to 0 to 1 and projected to 30 components first, which is the usual practice to cut noise and run time.",
   prep='''X = PCA(30, random_state=SEED).fit_transform(X_raw.values / 16.0)
print(X.shape)''',
   concept="Neighbours in the original space should be neighbours in the map. A heavy tailed kernel in 2D lets clusters separate.",
   scratch_md="A short NumPy t-SNE on 300 rows with a fixed sigma search, 250 gradient steps, and no early exaggeration tricks beyond a simple one.",
   scratch='''from sklearn.manifold import trustworthiness
def p_matrix(P, perplexity=30.0):
    n = len(P)
    D = ((P[:, None] - P[None]) ** 2).sum(axis=2)
    out = np.zeros((n, n))
    target = np.log(perplexity)
    for i in range(n):
        lo, hi, beta = 0.0, np.inf, 1.0
        for _ in range(50):
            pr = np.exp(-D[i] * beta); pr[i] = 0
            s = pr.sum() + 1e-12
            H = np.log(s) + beta * (D[i] * pr).sum() / s
            if H > target:
                lo = beta; beta = beta * 2 if hi == np.inf else (beta + hi) / 2
            else:
                hi = beta; beta = (beta + lo) / 2
        pr = np.exp(-D[i] * beta); pr[i] = 0
        out[i] = pr / pr.sum()
    return (out + out.T) / (2 * n)
sub = X[:300]
P = p_matrix(sub)
rng = np.random.default_rng(SEED)
Y = rng.normal(0, 1e-2, (300, 2)); vel = np.zeros_like(Y)
for it in range(250):
    d = ((Y[:, None] - Y[None]) ** 2).sum(axis=2)
    num = 1 / (1 + d); np.fill_diagonal(num, 0)
    Q = np.maximum(num / num.sum(), 1e-12)
    PQ = (4.0 if it < 50 else 1.0) * P - Q
    grad = 4 * ((PQ * num)[:, :, None] * (Y[:, None] - Y[None])).sum(axis=1)
    vel = (0.5 if it < 100 else 0.8) * vel - 100 * grad
    Y = Y + vel
print("trustworthiness of the scratch map (k=10):", round(float(trustworthiness(sub, Y, n_neighbors=10)), 3))''',
   params="| Parameter | Meaning |\n|---|---|\n| `perplexity` | Effective number of neighbours, usually 5 to 50. |\n| `learning_rate` | Step size. |\n| `max_iter` | Optimisation steps. |\n| `init` | `pca` is more stable than `random`. |",
   train_md="We run t-SNE for perplexity 5, 15, 30 and 50 and choose by trustworthiness (share of neighbours that are kept, 1 is best). The reference digit is not used for the choice.",
   train='''from sklearn.manifold import TSNE
embeds, rows = {}, []
for perp in [5, 15, 30, 50]:
    ts = TSNE(2, perplexity=perp, init="pca", random_state=SEED)
    emb = ts.fit_transform(X)
    embeds[perp] = emb
    rows.append({"perplexity": perp, "trustworthiness": trustworthiness(X, emb, n_neighbors=10), "kl_divergence": ts.kl_divergence_})
sweep = pd.DataFrame(rows).set_index("perplexity")
display(sweep.round(4))
BEST = int(sweep["trustworthiness"].idxmax())
emb = embeds[BEST]''',
   eval_md="Trustworthiness measures local structure. A 5-nearest-neighbour classifier trained on the embedding coordinates with cross validation shows whether the digit groups are separable in the map. This uses labels for evaluation only, and the embedding itself was fitted on all rows, so the number describes the map, not a prediction.",
   eval='''from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
knn_acc = cross_val_score(KNeighborsClassifier(5), emb, y_true, cv=5).mean()
pca2 = PCA(2, random_state=SEED).fit_transform(X)
METRICS = {"trustworthiness": float(sweep.loc[BEST, "trustworthiness"]), "trustworthiness_pca2": float(trustworthiness(X, pca2, n_neighbors=10)),
           "knn_accuracy_in_map": float(knn_acc), "knn_accuracy_pca2": float(cross_val_score(KNeighborsClassifier(5), pca2, y_true, cv=5).mean())}
CONFIG = {"perplexity": BEST, "init": "pca", "pca_components": 30}
EXTRA = {"sweep": sweep.round(4).reset_index().to_dict("records")}''',
   viz='''fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sc = axes[0].scatter(emb[:, 0], emb[:, 1], c=y_true, cmap="tab10", s=8); axes[0].set_title(f"t-SNE (perplexity {BEST}), colour = digit")
axes[0].legend(*sc.legend_elements(), title="digit", loc="best", fontsize=7)
axes[1].scatter(pca2[:, 0], pca2[:, 1], c=y_true, cmap="tab10", s=8); axes[1].set_title("PCA with 2 components")
plt.tight_layout(); save_figure(fig, NOTEBOOK_ID, "overview"); plt.show()''',
   interpret_md="Look for digits that sit between groups, and remember that gaps and sizes in the map are not reliable.",
   interpret='''from sklearn.neighbors import NearestNeighbors
idx = NearestNeighbors(n_neighbors=6).fit(emb).kneighbors(emb)[1][:, 1:]
purity = (y_true.values[idx] == y_true.values[:, None]).mean(axis=1)
print("share of rows whose 5 map neighbours all share their digit:", round(float((purity == 1).mean()), 3))
print("digits with the lowest neighbour purity:", pd.Series(purity, index=y_true.values).groupby(level=0).mean().sort_values().head(3).round(3).to_dict())''',
   experiments=[
       dict(title="Seed variability", md="Different seeds give different maps. Quality should be similar.",
            code='''for s in range(3):
    e = TSNE(2, perplexity=BEST, init="random", random_state=s).fit_transform(X)
    print("seed", s, "trustworthiness", round(float(trustworthiness(X, e, n_neighbors=10)), 4))'''),
       dict(title="Random versus PCA initialisation", md="PCA init keeps global layout more stable.",
            code='''for init in ["random", "pca"]:
    e = TSNE(2, perplexity=BEST, init=init, random_state=SEED).fit_transform(X)
    print(init, round(float(trustworthiness(X, e, n_neighbors=10)), 4))'''),
       dict(title="Distances between clusters", md="Distance between cluster centres in the map across perplexities. They change, so they carry no meaning.",
            code='''for perp, e in embeds.items():
    c0, c1 = e[y_true == 0].mean(axis=0), e[y_true == 1].mean(axis=0)
    spread = e.std()
    print(f"perplexity {perp}: distance between centres of digit 0 and 1 divided by map spread = {np.linalg.norm(c0 - c1) / spread:.2f}")''')],
   discussion="The original paper presents t-SNE as better than earlier methods at keeping local structure. The PCA comparison on trustworthiness checks that claim on digits only.",
   improvements="Use openTSNE for out of sample transforms and report several seeds.",
   takeaways=["Perplexity sets the neighbourhood size.", "Do not read distances between clusters.", "Use trustworthiness to compare maps."],
   use_cases="Visualisation of embeddings, quality checks.", limitations="No transform for new rows, non convex.", next="UMAP.")

un("09", DR, id="umap", title="UMAP: A Faster Map That Can Transform New Rows", level="Intermediate",
   tagline="Embed with a fuzzy neighbour graph and apply the map to unseen data.", name="UMAP", dataset="digits", extras="unsupervised",
   objectives=["How UMAP builds a fuzzy graph", "How n_neighbors and min_dist shape the map", "How to transform held out rows", "How UMAP compares with t-SNE and PCA"],
   problem=dict(context="Maps of embeddings must often be applied to new data as it arrives.",
                problem="Embed digit images and project held out digits into the same map.",
                matters="A transform lets a monitoring system place new items without refitting.",
                why_ml="Local structure of images is nonlinear.",
                why_algo="UMAP is fast, keeps more global structure than t-SNE and supports `transform`."),
   papers=["mcinnes2018", "mcinnes2018arxiv"],
   paper_relation="McInnes et al. introduce UMAP. We use the `umap-learn` package, version recorded in the experiment record.",
   math=r"""**Local fuzzy sets.** For point $i$ with nearest neighbour distance $\rho_i$, edge weight $w_{ij} = \exp\big(-(d_{ij}-\rho_i)/\sigma_i\big)$ with $\sigma_i$ such that $\sum_j w_{ij} = \log_2 k$. Symmetrise: $w = w_{ij} + w_{ji} - w_{ij}w_{ji}$.

**Low dimensional weights.** $q_{ij} = (1 + a\lVert y_i-y_j\rVert^{2b})^{-1}$ with $a, b$ fitted from `min_dist`.

**Objective.** Minimise the cross entropy between weights $w$ and $q$ by stochastic gradient descent with negative sampling.

**Limits.** Parameters shape the picture, global distances are only partly kept.""",
   dataset_md=DS_DIGITS, prep_md="Split first: 70 percent of rows fit the map, 30 percent are transformed to test the out of sample map. Pixels are scaled to 0 to 1.",
   prep='''from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X_raw.values / 16.0, y_true.values, test_size=0.3, stratify=y_true, random_state=SEED)
print(X_tr.shape, X_te.shape)''',
   concept="UMAP finds a low dimensional layout whose neighbour graph looks like the high dimensional one.",
   scratch_md="We compute the fuzzy edge weights for one row to show the local construction. The full optimisation is not reimplemented.",
   scratch='''from sklearn.neighbors import NearestNeighbors
K = 15
nn = NearestNeighbors(n_neighbors=K + 1).fit(X_tr)
dist, idx = nn.kneighbors(X_tr[:1])
d = dist[0, 1:]
rho = d[0]
lo, hi = 1e-3, 1e3
for _ in range(60):
    sigma = (lo + hi) / 2
    total = np.exp(-(d - rho) / sigma).sum()
    if total > np.log2(K):
        hi = sigma
    else:
        lo = sigma
w = np.exp(-(d - rho) / sigma)
print("rho:", round(float(rho), 3), "| sigma:", round(float(sigma), 3), "| sum of weights:", round(float(w.sum()), 3), "| target log2(k):", round(float(np.log2(K)), 3))
print("weights of the 15 neighbours:", np.round(w, 2))''',
   params="| Parameter | Meaning |\n|---|---|\n| `n_neighbors` | Local neighbourhood size. Larger values keep more global structure. |\n| `min_dist` | How tightly points may pack. |\n| `metric` | Distance. |\n| `random_state` | Fixes the result but turns off parallelism. |",
   train_md="We fit UMAP on training rows for a small grid of n_neighbors and min_dist and choose by trustworthiness on the training rows.",
   train='''import umap
from sklearn.manifold import trustworthiness
rows, models = [], {}
for nn_ in [5, 15, 50]:
    for md in [0.0, 0.1, 0.5]:
        m = umap.UMAP(n_neighbors=nn_, min_dist=md, random_state=SEED).fit(X_tr)
        models[(nn_, md)] = m
        rows.append({"n_neighbors": nn_, "min_dist": md, "trustworthiness": trustworthiness(X_tr, m.embedding_, n_neighbors=10)})
sweep = pd.DataFrame(rows)
display(sweep.round(4))
best = sweep.loc[sweep.trustworthiness.idxmax()]
model = models[(int(best.n_neighbors), float(best.min_dist))]
emb_tr = model.embedding_
emb_te = model.transform(X_te)''',
   eval_md="Trustworthiness on training rows, trustworthiness of the transformed held out rows, and a 5-nearest-neighbour classifier trained on training embeddings and scored on held out embeddings (labels used for evaluation only).",
   eval='''from sklearn.neighbors import KNeighborsClassifier
acc = KNeighborsClassifier(5).fit(emb_tr, y_tr).score(emb_te, y_te)
pca_m = PCA(2, random_state=SEED).fit(X_tr)
acc_pca = KNeighborsClassifier(5).fit(pca_m.transform(X_tr), y_tr).score(pca_m.transform(X_te), y_te)
METRICS = {"trustworthiness_train": float(best.trustworthiness), "trustworthiness_heldout": float(trustworthiness(X_te, emb_te, n_neighbors=10)),
           "knn_accuracy_heldout_umap": float(acc), "knn_accuracy_heldout_pca2": float(acc_pca)}
CONFIG = {"n_neighbors": int(best.n_neighbors), "min_dist": float(best.min_dist)}
EXTRA = {"sweep": sweep.round(4).to_dict("records")}''',
   viz='''fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(emb_tr[:, 0], emb_tr[:, 1], c=y_tr, cmap="tab10", s=8); axes[0].set_title("UMAP, training rows")
axes[1].scatter(emb_tr[:, 0], emb_tr[:, 1], c="lightgray", s=5)
axes[1].scatter(emb_te[:, 0], emb_te[:, 1], c=y_te, cmap="tab10", s=10); axes[1].set_title("Held out rows transformed into the same map")
plt.tight_layout(); save_figure(fig, NOTEBOOK_ID, "overview"); plt.show()''',
   interpret_md="Held out digits should land in the region of their own digit. Misplaced points are listed below.",
   interpret='''pred = KNeighborsClassifier(5).fit(emb_tr, y_tr).predict(emb_te)
wrong = np.where(pred != y_te)[0]
print("held out rows landing in another digit's region:", len(wrong), "of", len(y_te))
print(pd.crosstab(pd.Series(y_te[wrong], name="true digit"), pd.Series(pred[wrong], name="map region")).head(10))''',
   experiments=[
       dict(title="n_neighbors effect", md="Local versus global structure.",
            code='''for (nn_, md), m in models.items():
    if md == 0.1:
        print(f"n_neighbors={nn_}: trustworthiness {trustworthiness(X_tr, m.embedding_, n_neighbors=10):.4f}")'''),
       dict(title="Seed variability", md="Report the spread across seeds.",
            code='''vals = []
for s in range(3):
    m = umap.UMAP(n_neighbors=int(best.n_neighbors), min_dist=float(best.min_dist), random_state=s).fit(X_tr)
    vals.append(trustworthiness(X_tr, m.embedding_, n_neighbors=10))
print("trustworthiness by seed:", np.round(vals, 4))'''),
       dict(title="UMAP versus t-SNE run time", md="Wall clock on the same training rows.",
            code='''import time
from sklearn.manifold import TSNE
t = time.time(); umap.UMAP(random_state=SEED).fit(X_tr); a = time.time() - t
t = time.time(); TSNE(2, init="pca", random_state=SEED).fit(PCA(30, random_state=SEED).fit_transform(X_tr)); b = time.time() - t
print(f"UMAP {a:.1f} s, t-SNE {b:.1f} s on {len(X_tr)} rows (first UMAP call includes compilation time)")''')],
   discussion="The UMAP paper claims better run time and global structure than t-SNE. Our run time test covers one small dataset only, so it cannot confirm scaling claims.",
   improvements="Use supervised UMAP, report stability over seeds, and evaluate global structure with distance correlation.",
   takeaways=["UMAP supports transform.", "n_neighbors trades local for global structure.", "Always check the map with trustworthiness."],
   use_cases="Embedding visualisation, preprocessing before clustering.", limitations="Parameter dependent pictures.", next="Isolation Forest.")

un("10", AN, id="isolation_forest", title="Isolation Forest: Finding Malignant Outliers Among Benign Samples", level="Intermediate",
   tagline="Isolate anomalies with random splits and score by path length.", name="IsolationForest", dataset="breast_cancer",
   objectives=["Why anomalies are isolated in few splits", "How the anomaly score is normalised", "How to evaluate anomaly detection with rare labels", "How contamination affects the threshold"],
   problem=dict(context="Screening systems look for rare abnormal cases in a mostly normal population.",
                problem="Treat benign tumours as normal and a small share of malignant ones as anomalies, then rank rows without using labels for fitting.",
                matters="A ranking lets reviewers inspect the most unusual rows first.",
                why_ml="Anomalies are rare and labels are scarce.",
                why_algo="Isolation Forest is fast and needs no distance computation."),
   papers=["liu2008", "breunig2000"],
   paper_relation="Liu, Ting and Zhou introduce Isolation Forest. Breunig et al. introduce LOF, used here as a comparison.",
   math=r"""**Isolation tree.** Pick a random feature and a random split value between its minimum and maximum, recurse until each point is alone.

**Path length** $h(x)$ is the number of splits to isolate $x$. Anomalies have short paths.

**Score.** $s(x,n) = 2^{-E[h(x)]/c(n)}$ with $c(n) = 2H(n-1) - 2(n-1)/n$ and $H$ the harmonic number. Scores near 1 are anomalies, below 0.5 are normal.

**Limits.** Axis parallel splits, many irrelevant features dilute the signal.""",
   dataset_md=DS_BC, prep_md="We keep all 357 benign rows (normal) and a random 5 percent of the benign count in malignant rows (about 18 anomalies). Features are standardised with a scaler fitted on the training part. 70 percent of rows train the detector, 30 percent are scored. The label is used only to evaluate the ranking.",
   prep='''from sklearn.model_selection import train_test_split
rng = np.random.default_rng(SEED)
benign = np.where(y_true.values == 0)[0]
malignant = rng.choice(np.where(y_true.values == 1)[0], int(0.05 * len(benign)), replace=False)
rows = np.sort(np.r_[benign, malignant])
Xs, ys = X_raw.values[rows], y_true.values[rows]
X_tr, X_te, y_tr, y_te = train_test_split(Xs, ys, test_size=0.3, stratify=ys, random_state=SEED)
scaler = StandardScaler().fit(X_tr)
X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
print("rows:", len(rows), "| anomalies:", int(ys.sum()), "| test anomalies:", int(y_te.sum()))''',
   concept="Rows far from the crowd are separated by chance splits after few steps. Averaging many random trees makes the length stable.",
   scratch_md="A small isolation forest in NumPy, scored with the normalised path length.",
   scratch='''from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
def c_factor(n):
    return 0.0 if n <= 1 else 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n
def build(Xs_, depth, limit, rng):
    if depth >= limit or len(Xs_) <= 1:
        return ("leaf", len(Xs_))
    j = rng.integers(Xs_.shape[1])
    lo, hi = Xs_[:, j].min(), Xs_[:, j].max()
    if lo == hi:
        return ("leaf", len(Xs_))
    t = rng.uniform(lo, hi)
    m = Xs_[:, j] < t
    return ("node", j, t, build(Xs_[m], depth + 1, limit, rng), build(Xs_[~m], depth + 1, limit, rng))
def path(tree, x, d=0):
    if tree[0] == "leaf":
        return d + c_factor(tree[1])
    return path(tree[3] if x[tree[1]] < tree[2] else tree[4], x, d + 1)
rng = np.random.default_rng(SEED)
psi = 128
limit = int(np.ceil(np.log2(psi)))
trees = [build(X_tr[rng.choice(len(X_tr), min(psi, len(X_tr)), replace=False)], 0, limit, rng) for _ in range(100)]
mean_path = np.array([np.mean([path(t, x) for t in trees]) for x in X_te])
score_mine = 2 ** (-mean_path / c_factor(min(psi, len(X_tr))))
ref = IsolationForest(n_estimators=100, max_samples=128, random_state=SEED).fit(X_tr)
print("ROC-AUC scratch:", round(roc_auc_score(y_te, score_mine), 3), "| sklearn:", round(roc_auc_score(y_te, -ref.score_samples(X_te)), 3))''',
   params="| Parameter | Meaning |\n|---|---|\n| `n_estimators` | Number of trees. |\n| `max_samples` | Rows per tree. |\n| `contamination` | Expected anomaly share, sets the threshold only. |\n| `max_features` | Features per tree. |",
   train_md="We pick the number of trees from a small grid by the stability of the ranking (rank correlation between two seeds). Anomaly labels are not used.",
   train='''from scipy.stats import spearmanr
rows = []
for n in [25, 100, 300]:
    a = IsolationForest(n_estimators=n, random_state=0).fit(X_tr).score_samples(X_te)
    b = IsolationForest(n_estimators=n, random_state=1).fit(X_tr).score_samples(X_te)
    rows.append({"n_estimators": n, "rank_correlation_between_seeds": spearmanr(a, b)[0]})
stab = pd.DataFrame(rows).set_index("n_estimators")
display(stab.round(4))
model = IsolationForest(n_estimators=300, random_state=SEED).fit(X_tr)
score = -model.score_samples(X_te)''',
   eval_md="ROC-AUC and PR-AUC of the anomaly score against the held out labels, compared with a random ranking and with Local Outlier Factor. With about 5 test anomalies, these numbers are very uncertain, so we add a bootstrap interval.",
   eval='''from sklearn.metrics import average_precision_score
from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(X_tr)
score_lof = -lof.score_samples(X_te)
rng = np.random.default_rng(SEED)
boot = []
for _ in range(500):
    i = rng.integers(0, len(y_te), len(y_te))
    if y_te[i].sum() > 0 and y_te[i].sum() < len(i):
        boot.append(roc_auc_score(y_te[i], score[i]))
METRICS = {"ROC_AUC": roc_auc_score(y_te, score), "PR_AUC": average_precision_score(y_te, score), "PR_AUC_random": float(y_te.mean()),
           "ROC_AUC_LOF": roc_auc_score(y_te, score_lof), "ROC_AUC_boot_low": float(np.percentile(boot, 2.5)), "ROC_AUC_boot_high": float(np.percentile(boot, 97.5))}
CONFIG = {"n_estimators": 300, "anomaly_share": 0.05}
EXTRA = {"test_rows": int(len(y_te)), "test_anomalies": int(y_te.sum()), "stability": stab.round(4).reset_index().to_dict("records")}''',
   viz='''fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(score[y_te == 0], bins=20, alpha=0.7, label="benign"); axes[0].hist(score[y_te == 1], bins=10, alpha=0.7, label="malignant (anomaly)")
axes[0].set_xlabel("Anomaly score"); axes[0].set_ylabel("Rows"); axes[0].legend(); axes[0].set_title("Score distribution (test rows)")
Z = PCA(2, random_state=SEED).fit(X_tr).transform(X_te)
sc = axes[1].scatter(Z[:, 0], Z[:, 1], c=score, cmap="viridis", s=25); axes[1].set_title("Test rows, colour = anomaly score"); plt.colorbar(sc, ax=axes[1])
plt.tight_layout(); save_figure(fig, NOTEBOOK_ID, "overview"); plt.show()''',
   interpret_md="Which features make the top ranked rows unusual: compare their standardised values with the training mean.",
   interpret='''top = np.argsort(score)[-3:]
names = np.array(X_raw.columns)
for i in top:
    order = np.argsort(-np.abs(X_te[i]))[:3]
    print("row", i, "label", int(y_te[i]), "| most unusual features:", dict(zip(names[order], np.round(X_te[i][order], 1))))''',
   experiments=[
       dict(title="Contamination and threshold", md="The contamination parameter only sets the cut. Check precision and recall at three values.",
            code='''from sklearn.metrics import precision_score, recall_score
for c in [0.02, 0.05, 0.1]:
    thr = np.quantile(score, 1 - c)
    pred = (score >= thr).astype(int)
    print(f"contamination={c}: precision {precision_score(y_te, pred, zero_division=0):.2f}, recall {recall_score(y_te, pred):.2f}")'''),
       dict(title="Subsample size", md="max_samples per tree.",
            code='''for ms in [32, 64, 128, 256]:
    s = -IsolationForest(n_estimators=200, max_samples=ms, random_state=SEED).fit(X_tr).score_samples(X_te)
    print(f"max_samples={ms}: ROC-AUC {roc_auc_score(y_te, s):.3f}")'''),
       dict(title="Irrelevant features", md="Add noise columns and watch the score fall.",
            code='''rng = np.random.default_rng(SEED)
for extra in [0, 30, 100]:
    tr = np.c_[X_tr, rng.normal(size=(len(X_tr), extra))]; te = np.c_[X_te, rng.normal(size=(len(X_te), extra))]
    s = -IsolationForest(n_estimators=200, random_state=SEED).fit(tr).score_samples(te)
    print(f"{extra} noise columns: ROC-AUC {roc_auc_score(y_te, s):.3f}")''')],
   discussion="Liu et al. report that Isolation Forest ranks anomalies well with low cost. Our test has few anomalies, so the interval is wide and we cannot rank it against LOF with confidence.",
   improvements="Repeat the anomaly sampling many times and report the average, and use extended isolation forests.",
   takeaways=["Anomalies are isolated early.", "Contamination only sets a threshold.", "With rare labels report intervals."],
   use_cases="Fraud screening, sensor monitoring, data cleaning.", limitations="Axis parallel splits.", next="Reinforcement learning basics.",
   limits_extra="The anomaly set is constructed by sampling malignant rows, so this is a proxy for anomaly detection, not a real screening study.")
