import h5py
from tqdm import tqdm

from .dataset import create_radius_graph


def build_oracle_purified_split(
    h5_path,
    train_target=40000,
    test_bg_target=5000,
    test_sig_target=5000,
    scan_limit=200000,
    radius=0.4,
):
    train_graphs = []
    test_graphs = []
    test_labels = []

    with h5py.File(h5_path, 'r') as h5_file:
        dset = h5_file['df']['block0_values']
        upper = min(scan_limit, dset.shape[0])

        for index in tqdm(range(upper), desc='Oracle split scan'):
            row = dset[index, :]
            label = int(row[2100])

            graph = create_radius_graph(row, radius=radius)
            if graph is None:
                continue

            if label == 0:
                if len(train_graphs) < train_target:
                    train_graphs.append(graph)
                elif test_labels.count(0) < test_bg_target:
                    test_graphs.append(graph)
                    test_labels.append(0)
            elif label == 1 and test_labels.count(1) < test_sig_target:
                test_graphs.append(graph)
                test_labels.append(1)

            enough_train = len(train_graphs) >= train_target
            enough_signal = test_labels.count(1) >= test_sig_target
            enough_bg = test_labels.count(0) >= test_bg_target
            if enough_train and enough_signal and enough_bg:
                break

    return train_graphs, test_graphs, test_labels
