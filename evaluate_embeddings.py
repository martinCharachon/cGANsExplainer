import json
import h5py
import configargparse
import numpy as np
from sklearn.decomposition import PCA
from src.utils.domain_translation_metrics import apply_gaussian_kernel, js, calculate_fid


def main():
    p = configargparse.ArgParser()
    p.add_argument('--embeddings-path', required=True)
    p.add_argument('--results-path', required=True)
    p.add_argument('--evaluation-metrics', required=False, default="JS")
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

    if args.evaluation_metrics == "FD":
        res = {
            "FD": {
                "Real 0 - Real 1": calculate_fid(X[y == 0], X[y == 1]),
                "Real 0 - Ours Adv 0": calculate_fid(X[y == 0], X[y == 2]),
                "Real 0 - Ours St 0": calculate_fid(X[y == 0], X[y == 4]),
                "Real 0 - MG Adv 0": calculate_fid(X[y == 0], X[y == 6]),
                "Real 0 - SAG Adv 0": calculate_fid(X[y == 0], X[y == 8]),
                "Real 0 - SAG St 0": calculate_fid(X[y == 0], X[y == 10]),
                "Real 0 - Cyc Adv 0": calculate_fid(X[y == 0], X[y == 12]),
                "Real 1 - Ours Adv 0": calculate_fid(X[y == 1], X[y == 2]),
                "Real 1 - Ours St 0": calculate_fid(X[y == 1], X[y == 4]),
                "Real 1 - MG Adv 0": calculate_fid(X[y == 1], X[y == 6]),
                "Real 1 - SAG Adv 0": calculate_fid(X[y == 1], X[y == 8]),
                "Real 1 - SAG St 0": calculate_fid(X[y == 1], X[y == 10]),
                "Real 1 - Cyc Adv 0": calculate_fid(X[y == 1], X[y == 12]),
                "Real 1 - Real 0": calculate_fid(X[y == 1], X[y == 0]),
                "Real 0 - Ours Adv 1": calculate_fid(X[y == 0], X[y == 3]),
                "Real 0 - Ours St 1": calculate_fid(X[y == 0], X[y == 5]),
                "Real 0 - MG Adv 1": calculate_fid(X[y == 0], X[y == 7]),
                "Real 0 - SAG Adv 1": calculate_fid(X[y == 0], X[y == 9]),
                "Real 0 - SAG St 1": calculate_fid(X[y == 0], X[y == 11]),
                "Real 0 - Cyc Adv 1": calculate_fid(X[y == 0], X[y == 13]),
                "Real 1 - Ours Adv 1": calculate_fid(X[y == 1], X[y == 3]),
                "Real 1 - Ours St 1": calculate_fid(X[y == 1], X[y == 5]),
                "Real 1 - MG Adv 1": calculate_fid(X[y == 1], X[y == 7]),
                "Real 1 - SAG Adv 1": calculate_fid(X[y == 1], X[y == 9]),
                "Real 1 - SAG St 1": calculate_fid(X[y == 1], X[y == 11]),
                "Real 1 - Cyc Adv 1": calculate_fid(X[y == 1], X[y == 13])
            }
        }

    elif args.evaluation_metrics == "JS":
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

        eps = [0, 1e-100, 1e-50, 1e-30, 1e-20, 1e-10, 1e-5]
        js_00, js_01, js_02, js_03, js_04, js_05, js_06, js_07, \
        js_08, js_09, js_010, js_011, js_012, js_013 = \
            [], [], [], [], [], [], [], [], [], [], [], [], [], []
        js_10, js_11, js_12, js_13, js_14, js_15, js_16, js_17, \
        js_18, js_19, js_110, js_111, js_112, js_113 = \
            [], [], [], [], [], [], [], [], [], [], [], [], [], []
        for e in eps:
            js_00.append(np.sqrt(js(zi0, zi0, e)))
            js_01.append(np.sqrt(js(zi0, zi1, e)))
            js_02.append(np.sqrt(js(zi0, zi2, e)))
            js_03.append(np.sqrt(js(zi0, zi3, e)))
            js_04.append(np.sqrt(js(zi0, zi4, e)))
            js_05.append(np.sqrt(js(zi0, zi5, e)))
            js_06.append(np.sqrt(js(zi0, zi6, e)))
            js_07.append(np.sqrt(js(zi0, zi7, e)))
            js_08.append(np.sqrt(js(zi0, zi8, e)))
            js_09.append(np.sqrt(js(zi0, zi9, e)))
            js_010.append(np.sqrt(js(zi0, zi10, e)))
            js_011.append(np.sqrt(js(zi0, zi11, e)))
            js_012.append(np.sqrt(js(zi0, zi12, e)))
            js_013.append(np.sqrt(js(zi0, zi13, e)))
            js_10.append(np.sqrt(js(zi1, zi0, e)))
            js_11.append(np.sqrt(js(zi1, zi1, e)))
            js_12.append(np.sqrt(js(zi1, zi2, e)))
            js_13.append(np.sqrt(js(zi1, zi3, e)))
            js_14.append(np.sqrt(js(zi1, zi4, e)))
            js_15.append(np.sqrt(js(zi1, zi5, e)))
            js_16.append(np.sqrt(js(zi1, zi6, e)))
            js_17.append(np.sqrt(js(zi1, zi7, e)))
            js_18.append(np.sqrt(js(zi1, zi8, e)))
            js_19.append(np.sqrt(js(zi1, zi9, e)))
            js_110.append(np.sqrt(js(zi1, zi10, e)))
            js_111.append(np.sqrt(js(zi1, zi11, e)))
            js_112.append(np.sqrt(js(zi1, zi12, e)))
            js_113.append(np.sqrt(js(zi1, zi13, e)))

        js_00 = np.array(js_00).astype(float).tolist()
        js_01 = np.array(js_01).astype(float).tolist()
        js_02 = np.array(js_02).astype(float).tolist()
        js_03 = np.array(js_03).astype(float).tolist()
        js_04 = np.array(js_04).astype(float).tolist()
        js_05 = np.array(js_05).astype(float).tolist()
        js_06 = np.array(js_06).astype(float).tolist()
        js_07 = np.array(js_07).astype(float).tolist()
        js_08 = np.array(js_08).astype(float).tolist()
        js_09 = np.array(js_09).astype(float).tolist()
        js_010 = np.array(js_010).astype(float).tolist()
        js_011 = np.array(js_011).astype(float).tolist()
        js_012 = np.array(js_012).astype(float).tolist()
        js_013 = np.array(js_013).astype(float).tolist()
        js_10 = np.array(js_10).astype(float).tolist()
        js_11 = np.array(js_11).astype(float).tolist()
        js_12 = np.array(js_12).astype(float).tolist()
        js_13 = np.array(js_13).astype(float).tolist()
        js_14 = np.array(js_14).astype(float).tolist()
        js_15 = np.array(js_15).astype(float).tolist()
        js_16 = np.array(js_16).astype(float).tolist()
        js_17 = np.array(js_17).astype(float).tolist()
        js_18 = np.array(js_18).astype(float).tolist()
        js_19 = np.array(js_19).astype(float).tolist()
        js_110 = np.array(js_110).astype(float).tolist()
        js_111 = np.array(js_111).astype(float).tolist()
        js_112 = np.array(js_112).astype(float).tolist()
        js_113 = np.array(js_113).astype(float).tolist()
        res = {
            "JS": {
                "Real 0 - Real 0": js_00,
                "Real 0 - Real 1": js_01,
                "Real 0 - Ours Adv 0": js_02,
                "Real 0 - Ours St 0": js_04,
                "Real 0 - MG Adv 0": js_06,
                "Real 0 - SAG Adv 0": js_08,
                "Real 0 - SAG St 0": js_010,
                "Real 0 - Cyc Adv 0": js_012,
                "Real 1 - Ours Adv 0": js_12,
                "Real 1 - Ours St 0": js_14,
                "Real 1 - MG Adv 0": js_16,
                "Real 1 - SAG Adv 0": js_18,
                "Real 1 - SAG St 0": js_110,
                "Real 1 - Cyc Adv 0": js_112,
                "Real 1 - Real 0": js_10,
                "Real 0 - Ours Adv 1": js_03,
                "Real 0 - Ours St 1": js_05,
                "Real 0 - MG Adv 1": js_07,
                "Real 0 - SAG Adv 1": js_09,
                "Real 0 - SAG St 1": js_011,
                "Real 0 - Cyc Adv 1": js_013,
                "Real 1 - Ours Adv 1": js_13,
                "Real 1 - Ours St 1": js_15,
                "Real 1 - MG Adv 1": js_17,
                "Real 1 - SAG Adv 1": js_19,
                "Real 1 - SAG St 1": js_111,
                "Real 1 - Cyc Adv 1": js_113
            }
        }
    else:
        raise NotImplementedError

    with open(args.results_path, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == '__main__':
    main()
