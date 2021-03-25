import json
import h5py
import configargparse
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from src.utils.domain_translation_metrics import apply_gaussian_kernel


def main():
    p = configargparse.ArgParser()
    p.add_argument('--embeddings-path', required=True)
    p.add_argument('--split-path', required=False, default="./mnist/split/split_mnist_3_8.json")
    p.add_argument('--indexes-name', required=False, default="test_indexes")
    args = p.parse_args()

    with open(args.split_path, "r") as f:
        split = json.load(f)
        reference_list = split[args.indexes_name]

    results = h5py.File(args.embeddings_path, "r")
    x_real, x_ours_a, x_ours_st, x_mg_a, x_sa_a, x_sa_st, x_cyc_a,  = [], [], [], [], [], [], []
    label, prediction, prediction_t_duo, prediction_st_duo, \
    prediction_t_mg, prediction_t_sa, prediction_st_sa = [], [], [], [], [], [], []
    for ref in reference_list:
        group = f"data/{ref[0]}/{ref[1]}"

        x_real.append(results[group + "/x_real/mu"][()])
        x_ours_a.append(results[group + "/x_ours_a/mu"][()])
        x_cyc_a.append(results[group + "/x_cyc_a/mu"][()])
        x_ours_st.append(results[group + "/x_ours_st/mu"][()])
        x_mg_a.append(results[group + "/x_mg_a/mu"][()])
        x_sa_a.append(results[group + "/x_sa_a/mu"][()])
        x_sa_st.append(results[group + "/x_sa_st/mu"][()])

        label.append(results[group + "/x_real/label"][()])
        prediction.append(results[group + "/x_real/prediction"][()])

    x_real = np.array(x_real)
    x_ours_a = np.array(x_ours_a)
    x_cyc_a = np.array(x_cyc_a)
    x_ours_st = np.array(x_ours_st)
    x_mg_a = np.array(x_mg_a)
    x_sa_a = np.array(x_sa_a)
    x_sa_st = np.array(x_sa_st)
    prediction = np.array(prediction)
    results.close()

    y_real = np.ravel(np.where(prediction < 0.5, 0, 1))
    yt_duo = np.ravel(np.where(prediction < 0.5, 2, 3))
    yst_duo = np.ravel(np.where(prediction < 0.5, 4, 5))
    yt_mg = np.ravel(np.where(prediction < 0.5, 6, 7))
    yt_sa = np.ravel(np.where(prediction < 0.5, 8, 9))
    yst_sa = np.ravel(np.where(prediction < 0.5, 10, 11))
    yt_cycg = np.ravel(np.where(prediction < 0.5, 12, 13))

    x_list = [x_ours_a, x_ours_st, x_mg_a, x_sa_a, x_sa_st, x_cyc_a]
    y_list = [yt_duo, yst_duo, yt_mg, yt_sa, yst_sa, yt_cycg]
    X = x_real
    y = y_real
    for i, j in zip(x_list, y_list):
        X = np.concatenate((X, i), axis=0)
        y = np.concatenate((y, j), axis=0)


    pca = PCA(n_components=2)
    print("Fit PCA ...")
    pca.fit(X)
    print(f"PCA variance ratio = {pca.explained_variance_ratio_}")
    print("Transform X with PCA ...")
    X_embedded = pca.transform(X)

    xi, yi = np.mgrid[X_embedded[:, 0].min():X_embedded[:, 0].max():100 * 1j,
             X_embedded[:, 1].min():X_embedded[:, 1].max():100 * 1j]

    zi0 = apply_gaussian_kernel(X_embedded[y == 0], xi, yi)
    zi1 = apply_gaussian_kernel(X_embedded[y == 1], xi, yi)
    zi2 = apply_gaussian_kernel(X_embedded[y == 2], xi, yi)
    zi4 = apply_gaussian_kernel(X_embedded[y == 4], xi, yi)
    zi6 = apply_gaussian_kernel(X_embedded[y == 6], xi, yi)
    zi8 = apply_gaussian_kernel(X_embedded[y == 8], xi, yi)
    zi10 = apply_gaussian_kernel(X_embedded[y == 10], xi, yi)
    zi3 = apply_gaussian_kernel(X_embedded[y == 3], xi, yi)
    zi5 = apply_gaussian_kernel(X_embedded[y == 5], xi, yi)
    zi7 = apply_gaussian_kernel(X_embedded[y == 7], xi, yi)
    zi9 = apply_gaussian_kernel(X_embedded[y == 9], xi, yi)
    zi11 = apply_gaussian_kernel(X_embedded[y == 11], xi, yi)
    zi12 = apply_gaussian_kernel(X_embedded[y == 12], xi, yi)
    zi13 = apply_gaussian_kernel(X_embedded[y == 13], xi, yi)

    plt.figure(figsize=[8, 6])
    plt.scatter(X_embedded[:, 0][y == 0], X_embedded[:, 1][y == 0], s=20, c="red",
                label="Real - cfc(x) = 0")
    plt.scatter(X_embedded[:, 0][y == 1], X_embedded[:, 1][y == 1], s=20, c="blue",
                label="Real - cfc(x) = 1")
    plt.scatter(X_embedded[:, 0][y == 2], X_embedded[:, 1][y == 2], s=20, c="darkgreen",
                label="Ours adv - cfc(x) = 0")
    plt.scatter(X_embedded[:, 0][y == 4], X_embedded[:, 1][y == 4], s=20, c="limegreen",
                label="Ours st - cfc(x) = 0")
    plt.scatter(X_embedded[:, 0][y == 6], X_embedded[:, 1][y == 6], s=20, c="purple",
                label="mg adv - cfc(x) = 0")
    plt.scatter(X_embedded[:, 0][y == 8], X_embedded[:, 1][y == 8], s=20, c="dimgray",
                label="sa adv - cfc(x) = 0")
    plt.scatter(X_embedded[:, 0][y == 10], X_embedded[:, 1][y == 10], s=20, c="lightgray",
                label="sa st - cfc(x) = 0")
    plt.scatter(X_embedded[:, 0][y == 12], X_embedded[:, 1][y == 12], s=20, c="magenta",
                label="Cyc t - cfc(x) = 0")
    plt.legend()
    plt.xlabel("Axis 1")
    plt.ylabel("Axis 2")
    plt.show()

    plt.figure(figsize=[8, 6])
    plt.scatter(X_embedded[:, 0][y == 0], X_embedded[:, 1][y == 0], s=20, c="red",
                label="real - cfc(x) = 0")
    plt.scatter(X_embedded[:, 0][y == 1], X_embedded[:, 1][y == 1], s=20, c="blue",
                label="real - cfc(x) = 1")
    plt.scatter(X_embedded[:, 0][y == 3], X_embedded[:, 1][y == 3], s=20, c="darkgreen",
                label="Ours adv - cfc(x) = 1")
    plt.scatter(X_embedded[:, 0][y == 5], X_embedded[:, 1][y == 5], s=20, c="limegreen",
                label="Ours st - cfc(x) = 1")
    plt.scatter(X_embedded[:, 0][y == 7], X_embedded[:, 1][y == 7], s=20, c="purple",
                label="mg adv - cfc(x) = 1")
    plt.scatter(X_embedded[:, 0][y == 9], X_embedded[:, 1][y == 9], s=20, c="dimgray",
                label="sa adv - cfc(x) = 1")
    plt.scatter(X_embedded[:, 0][y == 11], X_embedded[:, 1][y == 11], s=20, c="lightgray",
                label="sa st - cfc(x) = 1")
    plt.scatter(X_embedded[:, 0][y == 13], X_embedded[:, 1][y == 13], s=20, c="magenta",
                label="Cyc t - cfc(x) = 1")
    plt.legend()
    plt.xlabel("Axis 1")
    plt.ylabel("Axis 2")
    plt.show()

    fig = plt.figure(figsize=(8, 8))
    plt.pcolormesh(xi, yi, zi0.reshape(xi.shape), cmap="hot")
    plt.contour(xi, yi, zi0.reshape(xi.shape), colors="red", linewidths=3, levels=3)
    plt.contour(xi, yi, zi1.reshape(xi.shape), colors="blue", linewidths=3, levels=3)
    plt.contour(xi, yi, zi2.reshape(xi.shape), colors="darkgreen", linewidths=3, levels=3)
    plt.contour(xi, yi, zi4.reshape(xi.shape), colors="limegreen", linewidths=3, levels=3)
    plt.contour(xi, yi, zi6.reshape(xi.shape), colors="purple", linewidths=3, levels=3)
    plt.contour(xi, yi, zi8.reshape(xi.shape), colors="dimgray", linewidths=3, levels=3)
    plt.contour(xi, yi, zi10.reshape(xi.shape), colors="lightgray", linewidths=3, levels=3)
    plt.contour(xi, yi, zi12.reshape(xi.shape), colors="magenta", linewidths=3, levels=3)
    plt.xlabel("Axis 1")
    plt.ylabel("Axis 2")
    plt.show()
    #
    fig = plt.figure(figsize=(8, 8))
    plt.pcolormesh(xi, yi, zi1.reshape(xi.shape), cmap="hot")
    plt.contour(xi, yi, zi0.reshape(xi.shape), colors="red", linewidths=3, levels=3)
    plt.contour(xi, yi, zi1.reshape(xi.shape), colors="blue", linewidths=3, levels=3)
    plt.contour(xi, yi, zi3.reshape(xi.shape), colors="darkgreen", linewidths=3, levels=3)
    plt.contour(xi, yi, zi5.reshape(xi.shape), colors="limegreen", linewidths=3, levels=3)
    plt.contour(xi, yi, zi7.reshape(xi.shape), colors="purple", linewidths=3, levels=3)
    plt.contour(xi, yi, zi9.reshape(xi.shape), colors="dimgray", linewidths=3, levels=3)
    plt.contour(xi, yi, zi11.reshape(xi.shape), colors="lightgray", linewidths=3, levels=3)
    plt.contour(xi, yi, zi13.reshape(xi.shape), colors="magenta", linewidths=3, levels=3)
    plt.xlabel("Axis 1")
    plt.ylabel("Axis 2")
    plt.show()




if __name__ == '__main__':
    main()
