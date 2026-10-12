import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import LABEL_COLUMN, create_radius_graph, open_events
from torch_geometric.data import Data


def process_h5_to_graphs(h5_path, output_dir, chunk_size=10000, radius=0.4, wrap_phi=False):
    """
    Converts LHCO 2020 H5 events into PyTorch Geometric radius graphs (ΔR <= radius),
    streaming `chunk_size` events at a time so the full file never sits in memory.

    Writes `data_chunk_{c}.pt` (one graph per event, in file order, so global index
    i lives in chunk i // chunk_size) and `labels.npy` (truth label per event,
    0: QCD, 1: W'), which training uses to select background-only events.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"📖 Reading {h5_path}...")
    h5_file, dset = open_events(h5_path)
    with h5_file:
        num_events = dset.shape[0]
        labels = np.empty(num_events, dtype=np.int8)

        for start in tqdm(range(0, num_events, chunk_size), desc='Chunks'):
            block = dset[start:min(start + chunk_size, num_events)]
            chunk_graphs = []

            for offset, event in enumerate(block):
                label = int(event[LABEL_COLUMN])
                labels[start + offset] = label

                graph = create_radius_graph(event, radius=radius, wrap_phi=wrap_phi)
                if graph is None:
                    # Keep a placeholder so indices stay aligned with the H5 rows
                    graph = Data(
                        x=torch.zeros((1, 3), dtype=torch.float),
                        edge_index=torch.empty((2, 0), dtype=torch.long),
                    )
                graph.y = torch.tensor([label], dtype=torch.float)
                chunk_graphs.append(graph)

            torch.save(chunk_graphs, os.path.join(output_dir, f"data_chunk_{start // chunk_size}.pt"))

    np.save(os.path.join(output_dir, 'labels.npy'), labels)
    print(f"✅ Saved {num_events:,} graphs and labels.npy to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert LHCO H5 events into radius-graph chunks.')
    parser.add_argument('--h5', default="dataset/raw/events_anomalydetection_v2.h5")
    parser.add_argument('--out', default="dataset/processed")
    parser.add_argument('--chunk-size', type=int, default=10000)
    parser.add_argument('--radius', type=float, default=0.4)
    parser.add_argument('--wrap-phi', action='store_true')
    args = parser.parse_args()
    process_h5_to_graphs(args.h5, args.out, args.chunk_size, args.radius, args.wrap_phi)
