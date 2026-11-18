"""Unsupervised specs, part A (clustering)."""

from specs_common import DS_BC, DS_DIGITS, DS_MPG, DS_WINE

UN = []
BASE = "02_unsupervised_learning"


def un(num, sub, **kw):
    spec = dict(num=num, level="Intermediate", minutes=1, extras="unsupervised",
                skills=["Pick settings without labels", "Check stability", "Compare with a reference labelling"],
                prereq=["Distances and vectors", "Standardising features"],
                exercises=["Repeat with `SEED = 7` and report which numbers move.",
                           "Change the number of rows by sampling 50 percent and compare the result.",
                           "Add pure noise columns and watch the internal scores."],
                eval_md="Internal scores need no labels: silhouette (higher is better, range -1 to 1), Davies-Bouldin (lower is better) "
                        "and Calinski-Harabasz (higher is better). External scores (ARI and NMI) compare with a reference label that was "
                        "never used for fitting. A reference label is not the truth about the data. It is one possible grouping.",
                lib_md="scikit-learn and scipy provide tested implementations.")
    spec.update(kw)
    spec["folder"] = f"{BASE}/{sub}/{spec['id']}"
    UN.append(spec)


CL = "01_clustering"

un("01", CL, id="kmeans", title="K-Means: Segmenting Wines by Chemistry", level="Beginner",
   tagline="Group rows around centres and learn how to pick k without labels.", name="KMeans", dataset="wine",
   objectives=["How Lloyd's algorithm alternates assignment and update", "Why k-means needs scaled features",
               "How to choose k with the elbow and silhouette", "Why results depend on initialisation"],
   problem=dict(context="Producers and retailers group products by measurable properties.",
                problem="Find groups of wines that are similar in 13 chemical measurements, without using the cultivar label.",
                matters="Groups guide blending, pricing and quality control.",
                why_ml="Many measurements together define similarity, which no single column does.",
                why_algo="K-means is the standard fast baseline for compact, similar sized groups."),
   papers=["lloyd1982", "rousseeuw1987"],
   paper_relation="Lloyd describes the iterative quantisation procedure now called k-means. Rousseeuw introduces the silhouette used here to compare values of k.",
   math=r"""**Objective.** Choose centres $\mu_1..\mu_k$ and assignments to minimise the within cluster sum of squares

$$J = \sum_{i=1}^{n} \lVert x_i - \mu_{c(i)} \rVert^2$$

**Lloyd steps.** Assign each point to its nearest centre, then set each centre to the mean of its points. $J$ never increases, so the loop converges to a local minimum.

**Silhouette.** For a point, $a$ is the mean distance to its own cluster and $b$ the mean distance to the nearest other cluster: $s = (b-a)/\max(a,b)$.

**Assumptions.** Roughly round clusters of similar size, features on one scale. **Limits.** Local minima, sensitive to outliers, k must be given.""",
   dataset_md=DS_WINE, eda_note="Compare the ranges of `proline` and `hue` to see why scaling matters.",
   prep_md="K-means uses Euclidean distance, so each column is standardised to mean 0 and standard deviation 1. No labels enter this step. "
           "There is no train and test split because we describe this dataset, not predict new rows. We check stability by resampling instead.",
   prep='''scaler = StandardScaler().fit(X_raw)
X = scaler.transform(X_raw)
print("scaled mean:", round(float(X.mean()), 3), "| scaled std:", round(float(X.std()), 3))''',
   concept="Each centre is a prototype. Points join the nearest prototype and prototypes move to the middle of their points until nothing changes.",
   scratch_md="We write Lloyd's algorithm in NumPy with random starting centres and keep the best of several starts.",
   scratch='''def kmeans_scratch(X, k, n_init=10, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        centres = X[rng.choice(len(X), k, replace=False)]
        for _ in range(iters):
            dist = ((X[:, None, :] - centres[None]) ** 2).sum(axis=2)
            labels = dist.argmin(axis=1)
            new = np.array([X[labels == j].mean(axis=0) if (labels == j).any() else centres[j] for j in range(k)])
            if np.allclose(new, centres):
                break
            centres = new
        inertia = ((X - centres[labels]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, labels, centres)
    return best

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
inertia_mine, labels_mine, _ = kmeans_scratch(X, 3)
ref = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit(X)
print("inertia scratch:", round(float(inertia_mine), 2), "| sklearn:", round(float(ref.inertia_), 2))
print("agreement of the two labellings (ARI):", round(adjusted_rand_score(labels_mine, ref.labels_), 3))''',
   params="| Parameter | Meaning |\n|---|---|\n| `n_clusters` | Number of groups k. |\n| `init` | `k-means++` spreads starting centres. |\n| `n_init` | Number of restarts, best inertia is kept. |\n| `max_iter` | Iteration limit per start. |",
   train_md="We try k from 2 to 8, record inertia and silhouette for each, and choose k by the highest silhouette. The reference label is not used for this choice.",
   train='''rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
    s = internal_scores(X, km.labels_)
    rows.append({"k": k, "inertia": km.inertia_, "silhouette": s["Silhouette"], "davies_bouldin": s["Davies_Bouldin"]})
sweep = pd.DataFrame(rows).set_index("k")
display(sweep.round(3))
BEST_K = int(sweep["silhouette"].idxmax())
print("chosen k by silhouette:", BEST_K)
model = KMeans(n_clusters=BEST_K, n_init=10, random_state=SEED).fit(X)
labels = model.labels_''',
   eval='''METRICS = {**internal_scores(X, labels), **external_scores(y_true, labels)}
rng = np.random.default_rng(SEED)
random_ari = external_scores(y_true, rng.integers(0, BEST_K, len(labels)))["ARI"]
CONFIG = {"k": BEST_K, "n_init": 10, "scaled": True}
EXTRA = {"random_labels_ARI": random_ari, "inertia": float(model.inertia_), "sweep": sweep.round(4).reset_index().to_dict("records")}
print("ARI of random labels (floor):", round(random_ari, 3))''',
   viz='''fig, axes = plt.subplots(1, 3, figsize=(14, 4))
sweep["inertia"].plot(marker="o", ax=axes[0]); axes[0].set_title("Elbow: inertia against k"); axes[0].set_ylabel("Inertia")
sweep["silhouette"].plot(marker="o", ax=axes[1], color="C1"); axes[1].set_title("Silhouette against k"); axes[1].set_ylabel("Silhouette")
Z = PCA(2, random_state=SEED).fit_transform(X)
axes[2].scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", s=18); axes[2].set_title(f"Clusters (k={BEST_K}) on first two components")
axes[2].set_xlabel("PC1"); axes[2].set_ylabel("PC2")
plt.tight_layout(); save_figure(fig, NOTEBOOK_ID, "overview"); plt.show()''',
   interpret_md="Cluster means in original units tell what each group is. Compare the groups with the reference cultivar only as a sanity check.",
   interpret='''profile = X_raw.assign(cluster=labels).groupby("cluster").mean().round(2)
display(profile.T.head(8))
display(pd.crosstab(pd.Series(labels, name="cluster"), pd.Series(y_true, name="reference label")))''',
   experiments=[
       dict(title="Initialisation sensitivity", md="Run a single random start 30 times and look at the spread of inertia.",
            code='''inertias = [KMeans(n_clusters=3, init="random", n_init=1, random_state=s).fit(X).inertia_ for s in range(30)]
print("single start inertia: min", round(min(inertias), 2), "max", round(max(inertias), 2), "distinct values", len(set(np.round(inertias, 3))))'''),
       dict(title="Scaled versus raw features", md="Cluster the raw columns and compare with the reference label.",
            code='''raw = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit(X_raw.values)
print("ARI raw features:", round(external_scores(y_true, raw.labels_)["ARI"], 3), "| ARI scaled:", round(external_scores(y_true, KMeans(n_clusters=3, n_init=10, random_state=SEED).fit(X).labels_)["ARI"], 3))'''),
       dict(title="Stability under resampling", md="Cluster 80 percent samples twice and compare the labellings on the shared rows.",
            code='''from sklearn.metrics import adjusted_rand_score
rng = np.random.default_rng(SEED)
aris = []
for _ in range(20):
    a = rng.choice(len(X), int(0.8 * len(X)), replace=False)
    b = rng.choice(len(X), int(0.8 * len(X)), replace=False)
    common = np.intersect1d(a, b)
    la = KMeans(BEST_K, n_init=5, random_state=0).fit(X[a]).predict(X[common])
    lb = KMeans(BEST_K, n_init=5, random_state=1).fit(X[b]).predict(X[common])
    aris.append(adjusted_rand_score(la, lb))
print("stability ARI mean", round(float(np.mean(aris)), 3), "std", round(float(np.std(aris, ddof=1)), 3))''')],
   discussion="Lloyd's procedure and Rousseeuw's silhouette work together here. Our sweep shows how the chosen k can differ between the elbow and the silhouette, and the stability experiment shows how much the grouping changes with the sample.",
   improvements="Use gap statistic or stability based selection of k, k-means++ with more restarts, and robust variants for outliers.",
   takeaways=["Scale features before k-means.", "Pick k with several criteria and a stability check.", "Compare with a reference label only as a sanity check."],
   use_cases="Customer segmentation, vector quantisation, initial grouping.", limitations="Round clusters only, k must be set.", next="Hierarchical clustering.")

un("02", CL, id="hierarchical_clustering", title="Hierarchical Clustering: A Tree of Wines", level="Beginner",
   tagline="Build a dendrogram and cut it at the level you need.", name="AgglomerativeClustering", dataset="wine",
   objectives=["How agglomerative merging works", "How linkage choice changes the tree", "How to read a dendrogram", "What the cophenetic correlation measures"],
   problem=dict(context="Taxonomies and product families are nested, not flat.",
                problem="Build a nested grouping of wines from chemical measurements.",
                matters="A tree lets an analyst choose the level of detail afterwards.",
                why_ml="Similarity from many measurements defines the tree.",
                why_algo="Hierarchical clustering needs no k in advance and shows structure at all scales."),
   papers=["ward1963", "rousseeuw1987"],
   paper_relation="Ward defines the merge criterion minimising the increase of within cluster variance. We use it as one of four linkages.",
   math=r"""**Procedure.** Start with every row as its own cluster. Repeatedly merge the two closest clusters until one remains.

**Linkages.** Single: smallest pairwise distance. Complete: largest. Average: mean pairwise. Ward: merge that increases total within cluster variance least,

$$\Delta(A,B) = \frac{|A||B|}{|A|+|B|}\lVert \bar x_A - \bar x_B \rVert^2$$

**Cophenetic correlation.** Correlation between original pairwise distances and the tree distances at which pairs first join. Close to 1 means the tree preserves distances.

**Limits.** $O(n^2)$ memory, merges cannot be undone, single linkage chains.""",
   dataset_md=DS_WINE, prep_md="Features are standardised because linkage uses Euclidean distance. No labels enter this step.",
   prep='''X = StandardScaler().fit_transform(X_raw)
print(X.shape)''',
   concept="The tree records the order and the height of every merge. Cutting at a height gives a flat clustering.",
   scratch_md="We implement naive single linkage on 40 rows and compare the merge heights with scipy.",
   scratch='''from scipy.cluster.hierarchy import linkage
sub = X[:40]
dist = np.sqrt(((sub[:, None] - sub[None]) ** 2).sum(axis=2))
np.fill_diagonal(dist, np.inf)
clusters = {i: [i] for i in range(len(sub))}
heights = []
while len(clusters) > 1:
    keys = list(clusters)
    best = (np.inf, None, None)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            d = min(dist[i, j] for i in clusters[keys[a]] for j in clusters[keys[b]])
            if d < best[0]:
                best = (d, keys[a], keys[b])
    heights.append(best[0])
    clusters[best[1]] = clusters[best[1]] + clusters.pop(best[2])
ref_heights = linkage(sub, method="single")[:, 2]
print("max difference of merge heights vs scipy:", float(np.abs(np.array(heights) - ref_heights).max()))''',
   params="| Parameter | Meaning |\n|---|---|\n| `method` | Linkage rule. |\n| `metric` | Distance, Euclidean for Ward. |\n| `n_clusters` / `t` | Where to cut. |",
   train_md="We build trees with four linkages, compare cophenetic correlation, and cut the Ward tree at k chosen by silhouette over k from 2 to 8.",
   train='''from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.spatial.distance import pdist
pair = pdist(X)
coph = {}
trees = {}
for method in ["single", "complete", "average", "ward"]:
    Z = linkage(X, method=method)
    trees[method] = Z
    coph[method] = cophenet(Z, pair)[0]
display(pd.Series(coph, name="cophenetic correlation").round(3).to_frame())
sil = {k: internal_scores(X, fcluster(trees["ward"], k, "maxclust"))["Silhouette"] for k in range(2, 9)}
BEST_K = max(sil, key=sil.get)
print("silhouette by k:", {k: round(v, 3) for k, v in sil.items()}, "| chosen k:", BEST_K)
labels = fcluster(trees["ward"], BEST_K, "maxclust") - 1''',
   eval='''METRICS = {**internal_scores(X, labels), **external_scores(y_true, labels), "cophenetic_ward": float(coph["ward"])}
CONFIG = {"linkage": "ward", "k": BEST_K}
EXTRA = {"cophenetic": {k: float(v) for k, v in coph.items()}}''',
   viz='''from scipy.cluster.hierarchy import dendrogram
fig, ax = plt.subplots(figsize=(11, 4))
dendrogram(trees["ward"], truncate_mode="lastp", p=20, ax=ax, color_threshold=0)
ax.set_title("Ward dendrogram (last 20 merges)"); ax.set_ylabel("Merge distance"); ax.set_xlabel("Cluster size or row")
save_figure(fig, NOTEBOOK_ID, "dendrogram"); plt.show()''',
   interpret_md="Long vertical gaps in the dendrogram mark natural cut levels. The cophenetic table shows how faithfully each linkage keeps the original distances.",
   interpret='''display(pd.crosstab(pd.Series(labels, name="cluster"), pd.Series(y_true, name="reference label")))
display(X_raw.assign(cluster=labels).groupby("cluster").mean().round(2).T.head(6))''',
   experiments=[
       dict(title="Linkage comparison", md="Cut every tree into three clusters and compare silhouette and ARI.",
            code='''for method, Z in trees.items():
    lab = fcluster(Z, 3, "maxclust") - 1
    s = internal_scores(X, lab)
    print(f"{method:9s} silhouette {s.get('Silhouette', float('nan')):.3f}  ARI {external_scores(y_true, lab)['ARI']:.3f}  sizes {np.bincount(lab)}")'''),
       dict(title="Chaining of single linkage", md="Single linkage tends to leave one big cluster and tiny leftovers.",
            code='''lab = fcluster(trees["single"], 5, "maxclust")
print("single linkage cluster sizes:", np.bincount(lab)[1:])
lab = fcluster(trees["ward"], 5, "maxclust")
print("ward cluster sizes:", np.bincount(lab)[1:])'''),
       dict(title="Distance metric", md="Compare Euclidean and Manhattan distance with average linkage.",
            code='''from sklearn.cluster import AgglomerativeClustering
for metric in ["euclidean", "manhattan", "cosine"]:
    lab = AgglomerativeClustering(n_clusters=3, metric=metric, linkage="average").fit_predict(X)
    print(f"{metric:10s} ARI {external_scores(y_true, lab)['ARI']:.3f}")''')],
   discussion="Ward's criterion gives compact clusters and our cophenetic table shows how it compares with other linkages on this data. The chaining experiment illustrates the known weakness of single linkage.",
   improvements="Use bootstrap support for tree branches and test linkage stability across resamples.",
   takeaways=["Linkage choice changes the tree.", "Cut where the dendrogram has a large gap.", "Check cophenetic correlation."],
   use_cases="Taxonomies, gene expression, product families.", limitations="Quadratic memory, greedy merges.", next="DBSCAN.")

un("03", CL, id="dbscan", title="DBSCAN: Density Clusters and Noise in Digit Images", level="Intermediate",
   tagline="Find dense regions of any shape and mark the rest as noise.", name="DBSCAN", dataset="digits",
   objectives=["Core, border and noise points", "How eps and min_samples are chosen", "Why DBSCAN needs scaled, low dimensional data", "How noise is reported"],
   problem=dict(context="Real data has dense groups of odd shapes and stray points that belong to no group.",
                problem="Group handwritten digit images by density in a 10 dimensional PCA space and flag stray images.",
                matters="Noise labelling lets an analyst review odd samples instead of forcing them into a group.",
                why_ml="Density based grouping does not assume round clusters.",
                why_algo="DBSCAN is the standard density method with a clear definition of noise."),
   papers=["schubert2017"],
   paper_relation="Schubert et al. revisit DBSCAN, explain when it works and how to set parameters. We follow their advice to inspect the k-distance plot.",
   math=r"""**Neighbourhood.** $N_\varepsilon(p) = \{q : d(p,q) \le \varepsilon\}$.

**Core point.** $|N_\varepsilon(p)| \ge \text{minPts}$. A border point lies in the neighbourhood of a core point but is not core itself. All others are noise.

**Cluster.** Maximal set of core points connected through neighbourhoods plus their border points.

**k-distance rule.** Sort the distance to the minPts-th neighbour for all points. A knee suggests $\varepsilon$.

**Limits.** One global density, distances lose meaning in high dimensions, results are sensitive to eps.""",
   dataset_md=DS_DIGITS, prep_md="Raw pixel space has 64 dimensions, which makes distances less informative. We project to 10 principal components fitted on all rows (descriptive analysis, no labels).",
   prep='''X = PCA(10, random_state=SEED).fit_transform(X_raw.values)
print("projected shape:", X.shape)''',
   concept="A dense region is a set of points that all have enough neighbours nearby. Clusters grow by walking from one core point to the next.",
   scratch_md="We implement DBSCAN with a neighbour list on 500 rows and compare with scikit-learn at the same settings.",
   scratch='''from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
sub = X[:500]
def dbscan_scratch(P, eps, min_pts):
    dist = np.sqrt(((P[:, None] - P[None]) ** 2).sum(axis=2))
    neigh = [np.where(dist[i] <= eps)[0] for i in range(len(P))]
    core = np.array([len(n) >= min_pts for n in neigh])
    labels = -np.ones(len(P), dtype=int)
    cid = 0
    for i in range(len(P)):
        if labels[i] != -1 or not core[i]:
            continue
        labels[i] = cid
        stack = [i]
        while stack:
            p = stack.pop()
            for q in neigh[p]:
                if labels[q] == -1:
                    labels[q] = cid
                    if core[q]:
                        stack.append(q)
        cid += 1
    return labels
eps_try = 20.0
mine = dbscan_scratch(sub, eps_try, 5)
ref = DBSCAN(eps=eps_try, min_samples=5).fit_predict(sub)
print("clusters scratch:", mine.max() + 1, "| sklearn:", ref.max() + 1, "| noise scratch:", int((mine == -1).sum()), "| sklearn:", int((ref == -1).sum()))''',
   params="| Parameter | Meaning |\n|---|---|\n| `eps` | Neighbourhood radius. |\n| `min_samples` | Neighbours needed for a core point. |\n| `metric` | Distance function. |",
   train_md="min_samples is set to 10 (twice the 5 usual for low dimensions is not needed here, we check 10 and 5 in an experiment). eps comes from a grid. We pick the eps with the best silhouette among settings that give at least 3 clusters and at most 30 percent noise.",
   train='''from sklearn.neighbors import NearestNeighbors
MIN_SAMPLES = 10
kdist = np.sort(NearestNeighbors(n_neighbors=MIN_SAMPLES).fit(X).kneighbors(X)[0][:, -1])
rows = []
for eps in np.linspace(8, 24, 17):
    lab = DBSCAN(eps=eps, min_samples=MIN_SAMPLES).fit_predict(X)
    s = internal_scores(X, lab)
    rows.append({"eps": round(eps, 1), "clusters": s["clusters"], "noise": s["noise_fraction"], "silhouette": s.get("Silhouette", np.nan)})
sweep = pd.DataFrame(rows)
display(sweep.round(3))
ok = sweep[(sweep.clusters >= 3) & (sweep.noise <= 0.30)]
BEST_EPS = float(ok.loc[ok.silhouette.idxmax(), "eps"]) if len(ok) else float(sweep.loc[sweep.silhouette.idxmax(), "eps"])
print("chosen eps:", BEST_EPS)
labels = DBSCAN(eps=BEST_EPS, min_samples=MIN_SAMPLES).fit_predict(X)''',
   eval='''METRICS = {**internal_scores(X, labels)}
mask = labels != -1
if mask.sum() > 1 and len(np.unique(labels[mask])) > 1:
    METRICS.update({k + "_nonnoise": v for k, v in external_scores(y_true[mask], labels[mask]).items()})
CONFIG = {"eps": BEST_EPS, "min_samples": MIN_SAMPLES, "pca_components": 10}
EXTRA = {"sweep": sweep.round(4).to_dict("records")}
print("external scores are computed on non noise rows only")''',
   viz='''fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(kdist); axes[0].axhline(BEST_EPS, color="red", linestyle="--"); axes[0].set_title(f"Sorted {MIN_SAMPLES}-distance"); axes[0].set_xlabel("Rows sorted"); axes[0].set_ylabel("Distance")
Z = PCA(2, random_state=SEED).fit_transform(X)
axes[1].scatter(Z[labels == -1, 0], Z[labels == -1, 1], c="lightgray", s=8, label="noise")
axes[1].scatter(Z[labels != -1, 0], Z[labels != -1, 1], c=labels[labels != -1], cmap="tab10", s=10)
axes[1].set_title("DBSCAN clusters, first two components"); axes[1].legend()
plt.tight_layout(); save_figure(fig, NOTEBOOK_ID, "overview"); plt.show()''',
   interpret_md="Noise rows are the digits the algorithm could not place in a dense group. Look at a few of them.",
   interpret='''noise_idx = np.where(labels == -1)[0][:10]
fig, axes = plt.subplots(1, len(noise_idx), figsize=(12, 1.6))
for ax, i in zip(np.atleast_1d(axes), noise_idx):
    ax.imshow(X_raw.values[i].reshape(8, 8), cmap="gray_r"); ax.axis("off"); ax.set_title(f"y={y_true.iloc[i]}", fontsize=8)
plt.show()
print("noise rows by reference digit:", pd.Series(y_true[labels == -1]).value_counts().sort_index().to_dict())''',
   experiments=[
       dict(title="Sensitivity to eps", md="A small change in eps can merge or split clusters.",
            code='''for eps in [BEST_EPS - 2, BEST_EPS, BEST_EPS + 2]:
    s = internal_scores(X, DBSCAN(eps=eps, min_samples=MIN_SAMPLES).fit_predict(X))
    print(f"eps={eps:.1f}: clusters {s['clusters']}, noise {s['noise_fraction']:.2f}")'''),
       dict(title="Dimensionality", md="Run DBSCAN on 2, 10, 30 and 64 dimensions with eps rescaled by the median neighbour distance.",
            code='''for d in [2, 10, 30]:
    P = PCA(d, random_state=SEED).fit_transform(X_raw.values)
    med = np.median(NearestNeighbors(n_neighbors=MIN_SAMPLES).fit(P).kneighbors(P)[0][:, -1])
    s = internal_scores(P, DBSCAN(eps=med * 1.1, min_samples=MIN_SAMPLES).fit_predict(P))
    print(f"{d} components: clusters {s['clusters']}, noise {s['noise_fraction']:.2f}")'''),
       dict(title="min_samples", md="Higher min_samples marks more points as noise.",
            code='''for m in [3, 5, 10, 20]:
    s = internal_scores(X, DBSCAN(eps=BEST_EPS, min_samples=m).fit_predict(X))
    print(f"min_samples={m}: clusters {s['clusters']}, noise {s['noise_fraction']:.2f}")''')],
   discussion="Schubert et al. stress that DBSCAN works with one global density and that parameter choice matters. Our eps sweep and the dimension experiment show the same sensitivity on digit data.",
   improvements="Use HDBSCAN for varying density, and evaluate stability across resamples.",
   takeaways=["eps and min_samples define density.", "Noise is a result, not a failure.", "Reduce dimension first."],
   use_cases="Spatial data, anomaly review, arbitrary shaped clusters.", limitations="One global density, high dimension sensitivity.", next="HDBSCAN.")

un("04", CL, id="hdbscan", title="HDBSCAN: Clusters of Varying Density", level="Advanced",
   tagline="Build a hierarchy of density clusters and keep the stable ones.", name="HDBSCAN", dataset="digits",
   objectives=["Mutual reachability distance", "How stability selects clusters", "Why min_cluster_size replaces eps", "How soft probabilities describe membership"],
   problem=dict(context="Clusters in real data rarely share one density, so a single eps fails.",
                problem="Cluster digit images in PCA space without choosing an eps.",
                matters="Fewer tuning decisions and a per-row confidence.",
                why_ml="Density structure is not visible in raw pixels.",
                why_algo="HDBSCAN extends DBSCAN across all density levels."),
   papers=["campello2013", "schubert2017"],
   paper_relation="Campello et al. introduce HDBSCAN. We use the scikit-learn implementation, which may differ in details from the original code.",
   math=r"""**Core distance.** $d_{core}(x)$ is the distance to the minPts-th nearest neighbour.

**Mutual reachability.** $d_{mreach}(a,b) = \max\{d_{core}(a), d_{core}(b), d(a,b)\}$.

**Hierarchy.** Build the minimum spanning tree of the mutual reachability graph and remove edges from the largest to the smallest, giving a tree of clusters over density levels.

**Stability.** A cluster's stability is $\sum_{x \in C} (\lambda_{x} - \lambda_{birth})$ with $\lambda = 1/\text{distance}$. Clusters with higher stability than the sum of their children are kept.

**Limits.** Parameters min_cluster_size and min_samples still matter.""",
   dataset_md=DS_DIGITS, prep_md="Same 10 component PCA projection as the DBSCAN notebook.",
   prep='''X = PCA(10, random_state=SEED).fit_transform(X_raw.values)
print(X.shape)''',
   concept="DBSCAN with every possible eps at once, then keep clusters that stay stable over a wide range of densities.",
   scratch_md="We compute core distances, the mutual reachability matrix and a single linkage tree on it, then cut at one level. This shows the first half of the algorithm. The stability based cluster selection is not implemented here.",
   scratch='''from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
sub = X[:400]
dist = np.sqrt(((sub[:, None] - sub[None]) ** 2).sum(axis=2))
core = np.sort(dist, axis=1)[:, 10]
mreach = np.maximum(np.maximum(core[:, None], core[None]), dist)
np.fill_diagonal(mreach, 0)
Z = linkage(squareform(mreach, checks=False), method="single")
cut = np.percentile(Z[:, 2], 90)
lab = fcluster(Z, cut, "distance")
sizes = np.bincount(lab)[1:]
print("clusters at one cut:", len(sizes), "| sizes >= 10:", int((sizes >= 10).sum()))
print("mean core distance:", round(float(core.mean()), 2), "| mean pairwise distance:", round(float(dist.mean()), 2))''',
   params="| Parameter | Meaning |\n|---|---|\n| `min_cluster_size` | Smallest group accepted as a cluster. |\n| `min_samples` | Neighbours for the core distance. |\n| `cluster_selection_method` | `eom` or `leaf`. |",
   train_md="We vary min_cluster_size and pick the setting with the best silhouette on non noise rows among settings with at most 40 percent noise.",
   train='''from sklearn.cluster import HDBSCAN
rows = []
for mcs in [10, 15, 20, 30, 50]:
    lab = HDBSCAN(min_cluster_size=mcs).fit_predict(X)
    s = internal_scores(X, lab)
    rows.append({"min_cluster_size": mcs, "clusters": s["clusters"], "noise": s["noise_fraction"], "silhouette": s.get("Silhouette", np.nan)})
sweep = pd.DataFrame(rows)
display(sweep.round(3))
ok = sweep[(sweep.noise <= 0.40) & sweep.silhouette.notna()]
BEST = int(ok.loc[ok.silhouette.idxmax(), "min_cluster_size"]) if len(ok) else 15
model = HDBSCAN(min_cluster_size=BEST).fit(X)
labels = model.labels_
print("chosen min_cluster_size:", BEST)''',
   eval='''METRICS = {**internal_scores(X, labels)}
mask = labels != -1
if mask.sum() > 1 and len(np.unique(labels[mask])) > 1:
    METRICS.update({k + "_nonnoise": v for k, v in external_scores(y_true[mask], labels[mask]).items()})
CONFIG = {"min_cluster_size": BEST, "pca_components": 10}
EXTRA = {"sweep": sweep.round(4).to_dict("records"), "mean_probability": float(model.probabilities_[mask].mean()) if mask.any() else None}''',
   viz='''fig, axes = plt.subplots(1, 2, figsize=(11, 4))
Z2 = PCA(2, random_state=SEED).fit_transform(X)
axes[0].scatter(Z2[labels == -1, 0], Z2[labels == -1, 1], c="lightgray", s=8)
axes[0].scatter(Z2[mask, 0], Z2[mask, 1], c=labels[mask], cmap="tab10", s=10); axes[0].set_title("HDBSCAN clusters")
sc = axes[1].scatter(Z2[:, 0], Z2[:, 1], c=model.probabilities_, cmap="viridis", s=10); axes[1].set_title("Membership probability")
plt.colorbar(sc, ax=axes[1]); plt.tight_layout(); save_figure(fig, NOTEBOOK_ID, "overview"); plt.show()''',
   interpret_md="Low probability rows sit at the edge of a cluster. They are the ones to review first.",
   interpret='''display(pd.crosstab(pd.Series(labels, name="cluster"), pd.Series(y_true, name="reference digit")))
print("rows with probability below 0.5:", int((model.probabilities_ < 0.5).sum()))''',
   experiments=[
       dict(title="HDBSCAN versus DBSCAN", md="Compare noise and cluster count with the DBSCAN from the previous notebook.",
            code='''from sklearn.cluster import DBSCAN
for name, lab in [("HDBSCAN", labels), ("DBSCAN eps=16", DBSCAN(eps=16, min_samples=10).fit_predict(X))]:
    s = internal_scores(X, lab)
    print(f"{name}: clusters {s['clusters']}, noise {s['noise_fraction']:.2f}")'''),
       dict(title="min_samples effect", md="Vary min_samples with min_cluster_size fixed.",
            code='''for ms in [3, 5, 10, 20]:
    s = internal_scores(X, HDBSCAN(min_cluster_size=BEST, min_samples=ms).fit_predict(X))
    print(f"min_samples={ms}: clusters {s['clusters']}, noise {s['noise_fraction']:.2f}")'''),
       dict(title="Selection method", md="Excess of mass (`eom`) versus leaf clusters.",
            code='''for method in ["eom", "leaf"]:
    s = internal_scores(X, HDBSCAN(min_cluster_size=BEST, cluster_selection_method=method).fit_predict(X))
    print(f"{method}: clusters {s['clusters']}, noise {s['noise_fraction']:.2f}")''')],
   discussion="Campello et al. motivate HDBSCAN by the weakness of a global eps. Our comparison with DBSCAN checks whether the hierarchy gives a different cluster count and noise level on this data.",
   improvements="Use cluster persistence plots and test on data with known varying density.",
   takeaways=["No eps, but still two parameters.", "Probabilities give per-row confidence.", "Noise remains possible."],
   use_cases="Exploratory clustering, embeddings, mixed density data.", limitations="Parameters still matter.", next="Gaussian mixture models.")

un("05", CL, id="gaussian_mixture", title="Gaussian Mixture Models: Soft Clusters of Cars", level="Intermediate",
   tagline="Fit a mixture of Gaussians with EM and choose the model with BIC.", name="GaussianMixture", dataset="auto_mpg",
   objectives=["How EM alternates responsibilities and parameter updates", "How covariance type changes the shape of clusters", "How BIC balances fit and complexity", "How soft assignments differ from hard ones"],
   problem=dict(context="Car designs from the 1970s and 1980s came from different regions with different engineering choices.",
                problem="Find groups of cars from engine and weight measurements, without using the origin label or mpg.",
                matters="Soft membership tells how typical each car is of its group.",
                why_ml="Several correlated measurements define the groups.",
                why_algo="A Gaussian mixture gives elliptical clusters and probabilities."),
   papers=["dempster1977"],
   paper_relation="Dempster, Laird and Rubin present the EM algorithm in general form. Gaussian mixtures are one application.",
   math=r"""**Model.** $p(x) = \sum_{k=1}^K \pi_k\, \mathcal{N}(x \mid \mu_k, \Sigma_k)$.

**E step.** Responsibility $r_{ik} = \dfrac{\pi_k \mathcal{N}(x_i\mid\mu_k,\Sigma_k)}{\sum_j \pi_j \mathcal{N}(x_i\mid\mu_j,\Sigma_j)}$.

**M step.** $N_k = \sum_i r_{ik}$, $\pi_k = N_k/n$, $\mu_k = \frac1{N_k}\sum_i r_{ik}x_i$, $\Sigma_k = \frac1{N_k}\sum_i r_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top$.

**Likelihood.** Never decreases over EM iterations, but may reach a local maximum.

**BIC.** $\text{BIC} = p\ln n - 2\ln \hat L$ with $p$ free parameters. Lower is better.

**Limits.** Local optima, singular covariances, assumes Gaussian shapes.""",
   dataset_md=DS_MPG,
   eda="""print("Rows and columns:", X_raw.shape)
print("Missing values:", int(X_raw.isna().sum().sum()))
display(X_raw.describe().T.round(3))
fig, ax = plt.subplots(figsize=(5, 3))
X_raw["origin"].value_counts().sort_index().plot.bar(ax=ax)
ax.set_xlabel("Origin (reference only)"); ax.set_ylabel("Cars"); plt.show()""",
   prep_md="The columns `cylinders`, `displacement`, `horsepower`, `weight`, `acceleration` and `model_year` are standardised. `origin` is kept aside as a reference label. `mpg` is the target of another notebook and is not used here.",
   prep='''cols = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]
X = StandardScaler().fit_transform(X_raw[cols])
y_true = X_raw["origin"].values      # reference only
print(X.shape, "| origin counts:", pd.Series(y_true).value_counts().sort_index().to_dict())''',
   concept="Every car belongs to every group with some probability. EM raises the likelihood step by step.",
   scratch_md="We write EM for a diagonal covariance mixture and compare the log likelihood with scikit-learn.",
   scratch='''from sklearn.mixture import GaussianMixture
def gmm_diag(X, k, iters=200, seed=0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    mu = X[rng.choice(n, k, replace=False)]
    var = np.ones((k, d)); pi = np.ones(k) / k
    for _ in range(iters):
        logp = -0.5 * (((X[:, None] - mu[None]) ** 2) / var[None] + np.log(2 * np.pi * var[None])).sum(axis=2) + np.log(pi)
        m = logp.max(axis=1, keepdims=True)
        r = np.exp(logp - m); total = r.sum(axis=1, keepdims=True); r /= total
        ll = float((m + np.log(total)).sum())
        nk = r.sum(axis=0) + 1e-9
        pi = nk / n
        mu = (r.T @ X) / nk[:, None]
        var = (r.T @ (X ** 2)) / nk[:, None] - mu ** 2 + 1e-4
    return ll
ll_mine = max(gmm_diag(X, 3, seed=s) for s in range(5))
ref = GaussianMixture(3, covariance_type="diag", n_init=5, random_state=SEED).fit(X)
print("total log likelihood scratch:", round(ll_mine, 1), "| sklearn:", round(float(ref.score(X) * len(X)), 1))''',
   params="| Parameter | Meaning |\n|---|---|\n| `n_components` | Number of Gaussians. |\n| `covariance_type` | `full`, `tied`, `diag`, `spherical`. |\n| `n_init` | Restarts. |\n| `reg_covar` | Added to the diagonal for stability. |",
   train_md="We fit k from 1 to 8 for four covariance types and select the model with the lowest BIC. The origin label is not used.",
   train='''rows = []
for cov in ["full", "tied", "diag", "spherical"]:
    for k in range(1, 9):
        g = GaussianMixture(k, covariance_type=cov, n_init=3, random_state=SEED).fit(X)
        rows.append({"covariance": cov, "k": k, "BIC": g.bic(X), "AIC": g.aic(X)})
grid = pd.DataFrame(rows)
best = grid.loc[grid.BIC.idxmin()]
print("lowest BIC:", best.to_dict())
model = GaussianMixture(int(best.k), covariance_type=best.covariance, n_init=5, random_state=SEED).fit(X)
labels = model.predict(X)
proba = model.predict_proba(X)''',
   eval='''METRICS = {**internal_scores(X, labels), **external_scores(y_true, labels), "BIC": float(best.BIC), "mean_max_responsibility": float(proba.max(axis=1).mean())}
CONFIG = {"k": int(best.k), "covariance_type": str(best.covariance), "n_init": 5}
EXTRA = {"bic_grid": grid.round(2).to_dict("records")}''',
   viz='''fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for cov, g in grid.groupby("covariance"):
    axes[0].plot(g.k, g.BIC, marker="o", label=cov)
axes[0].set_xlabel("Components k"); axes[0].set_ylabel("BIC (lower is better)"); axes[0].legend(); axes[0].set_title("Model selection")
Z = PCA(2, random_state=SEED).fit_transform(X)
sc = axes[1].scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", s=18, alpha=proba.max(axis=1)); axes[1].set_title("Components, opacity = confidence")
plt.tight_layout(); save_figure(fig, NOTEBOOK_ID, "overview"); plt.show()''',
   interpret_md="Component means in original units describe the car groups. Compare with the origin label only as a check.",
   interpret='''display(X_raw[cols].assign(component=labels).groupby("component").mean().round(1))
display(pd.crosstab(pd.Series(labels, name="component"), pd.Series(y_true, name="origin (1 US, 2 Europe, 3 Japan)")))''',
   experiments=[
       dict(title="Hard versus soft assignment", md="List the cars with the most uncertain membership.",
            code='''unc = 1 - proba.max(axis=1)
idx = np.argsort(unc)[-5:]
display(X_raw.iloc[idx][cols].assign(max_prob=proba.max(axis=1)[idx].round(2)))'''),
       dict(title="Initialisation", md="Single start fits with different seeds can land in different local optima.",
            code='''scores = [GaussianMixture(int(best.k), covariance_type=best.covariance, n_init=1, init_params="random", random_state=s).fit(X).score(X) for s in range(20)]
print("mean log likelihood per row: min", round(min(scores), 3), "max", round(max(scores), 3))'''),
       dict(title="K-means versus GMM", md="Compare with k-means at the same k using the origin label as reference.",
            code='''from sklearn.cluster import KMeans
km = KMeans(int(best.k), n_init=10, random_state=SEED).fit_predict(X)
print("ARI k-means:", round(external_scores(y_true, km)["ARI"], 3), "| ARI GMM:", round(METRICS["ARI"], 3))''')],
   discussion="Dempster et al. show that EM never lowers the likelihood. Our scratch implementation and the restart experiment show the practical side: the final value depends on the start.",
   improvements="Use Bayesian mixtures to infer k, and check stability of the chosen BIC model across resamples.",
   takeaways=["EM alternates soft assignment and parameter updates.", "BIC penalises complexity.", "Report soft membership."],
   use_cases="Soft segmentation, density estimation, anomaly scores.", limitations="Gaussian shapes, local optima.", next="Spectral clustering.")
