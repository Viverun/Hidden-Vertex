import warnings

import numpy as np
from tqdm import tqdm

from .dataset import LABEL_COLUMN, create_radius_graph, open_events


def build_oracle_purified_split(
    h5_path,
    train_target=40000,
    test_bg_target=5000,
    test_sig_target=5000,
    scan_limit=200000,
    radius=0.4,
    wrap_phi=False,
    read_block=10000,
    return_indices=False,
):
    """
    Scans events in file order and routes them with the truth label (column 2100):
    the first `train_target` background events go to training, the next `test_bg_target`
    background events and the first `test_sig_target` signal events go to the test set.

    Rows are read from HDF5 in blocks of `read_block`; routing is identical to the
    row-by-row notebook loop. Blocks in which no event can be routed any more are skipped
    without per-row work.

    scan_limit=None scans the whole file. This is required for
    `events_anomalydetection_v2.h5`, which is sorted (rows 0-999,999 background,
    1,000,000+ signal); the original shuffled `events_anomalydetection.h5` used by the
    notebook fills all quotas within ~56k rows.

    return_indices=True also returns (train_indices, test_indices): the H5 row of each graph.
    """
    train_graphs, test_graphs, test_labels = [], [], []
    train_indices, test_indices = [], []
    n_test_bg = 0
    n_test_sig = 0

    h5_file, dset = open_events(h5_path)
    with h5_file:
        upper = dset.shape[0] if scan_limit is None else min(scan_limit, dset.shape[0])
        progress = tqdm(total=upper, desc='Oracle split scan')
        done = False

        for block_start in range(0, upper, read_block):
            block = dset[block_start:min(block_start + read_block, upper)]
            progress.update(len(block))

            need_bg = len(train_graphs) < train_target or n_test_bg < test_bg_target
            need_sig = n_test_sig < test_sig_target
            labels = block[:, LABEL_COLUMN]
            if not ((need_bg and (labels == 0).any()) or (need_sig and (labels == 1).any())):
                continue

            for offset, row in enumerate(block):
                label = int(row[LABEL_COLUMN])

                if label == 0:
                    if len(train_graphs) < train_target:
                        bucket = 'train'
                    elif n_test_bg < test_bg_target:
                        bucket = 'test_bg'
                    else:
                        bucket = None
                elif label == 1 and n_test_sig < test_sig_target:
                    bucket = 'test_sig'
                else:
                    bucket = None

                # Only build graphs for events that will be kept
                graph = create_radius_graph(row, radius=radius, wrap_phi=wrap_phi) if bucket else None
                if graph is not None:
                    if bucket == 'train':
                        train_graphs.append(graph)
                        train_indices.append(block_start + offset)
                    else:
                        test_graphs.append(graph)
                        test_indices.append(block_start + offset)
                        if bucket == 'test_bg':
                            test_labels.append(0)
                            n_test_bg += 1
                        else:
                            test_labels.append(1)
                            n_test_sig += 1

                if (
                    len(train_graphs) >= train_target
                    and n_test_sig >= test_sig_target
                    and n_test_bg >= test_bg_target
                ):
                    done = True
                    break

            if done:
                break

        progress.close()

    shortfalls = {
        name: (got, want)
        for name, got, want in [
            ('train', len(train_graphs), train_target),
            ('test_bg', n_test_bg, test_bg_target),
            ('test_sig', n_test_sig, test_sig_target),
        ]
        if got < want
    }
    if shortfalls:
        warnings.warn(
            f"Oracle split quotas not met after scanning {upper:,} rows (got, wanted): {shortfalls}. "
            "If the file is sorted by label (e.g. events_anomalydetection_v2.h5), set scan_limit to "
            "None to scan the whole file.",
            stacklevel=2,
        )

    if return_indices:
        return train_graphs, test_graphs, test_labels, np.array(train_indices), np.array(test_indices)
    return train_graphs, test_graphs, test_labels
