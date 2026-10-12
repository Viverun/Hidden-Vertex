import math
from collections import OrderedDict

import h5py
import numpy as np
import torch
from torch.utils.data import Sampler
from torch_geometric.data import Data
from torch_geometric.data import Dataset

try:
    # Registers the compression filters used by the LHCO HDF5 files (needed on Kaggle).
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

NUM_PARTICLE_COLUMNS = 2100  # 700 particles x (pT, eta, phi)
LABEL_COLUMN = 2100


def open_events(h5_path):
    """Opens the LHCO events file and returns (file handle, event dataset)."""
    h5_file = h5py.File(h5_path, 'r')
    return h5_file, h5_file['df']['block0_values']


def create_radius_graph(event_row, radius=0.4, wrap_phi=False):
    """
    Builds a radius graph in (eta, phi) space connecting particles with ΔR <= radius.

    wrap_phi=False reproduces the Part 1 notebook (plain Euclidean distance, so
    particles on either side of phi = ±π are never connected). wrap_phi=True uses
    the physically correct periodic Δφ.
    """
    particles = np.asarray(event_row[:NUM_PARTICLE_COLUMNS], dtype=np.float32).reshape(-1, 3)
    particles = particles[particles[:, 0] > 0]

    if len(particles) < 2:
        return None

    x = torch.from_numpy(particles)
    eta = x[:, 1]
    phi = x[:, 2]
    d_eta = eta[:, None] - eta[None, :]
    d_phi = phi[:, None] - phi[None, :]
    if wrap_phi:
        d_phi = torch.remainder(d_phi + math.pi, 2 * math.pi) - math.pi

    # Compare squared distances: same edges as torch.cdist(...) <= radius, without the sqrt
    adjacency = d_eta.square_() + d_phi.square_() <= radius * radius
    adjacency.fill_diagonal_(False)
    edge_index = adjacency.nonzero().t().contiguous()
    return Data(x=x, edge_index=edge_index)


class RawChunkDataset(Dataset):
    """
    Loads graphs from pre-processed chunk files (`data_chunk_{i}.pt`, 10k graphs each).
    Keeps the `max_cache` most recently used chunks in memory.

    Pair with `ChunkGroupedSampler` when shuffling: a plain random shuffle touches a
    different chunk on almost every item and reloads chunks from disk constantly.
    """
    def __init__(self, chunk_dir, indices, max_cache=20, chunk_size=10000):
        super().__init__(None, None, None)
        self.chunk_dir = chunk_dir
        self.data_indices = list(indices)
        self.chunk_size = chunk_size
        self.cache = OrderedDict()
        self.max_cache = max_cache

    def len(self):
        return len(self.data_indices)

    def get(self, idx):
        real_idx = self.data_indices[idx]
        chunk_idx, in_chunk_idx = divmod(real_idx, self.chunk_size)

        if chunk_idx in self.cache:
            self.cache.move_to_end(chunk_idx)
        else:
            if len(self.cache) >= self.max_cache:
                self.cache.popitem(last=False)  # evict least recently used
            chunk_path = f"{self.chunk_dir}/data_chunk_{chunk_idx}.pt"
            # weights_only=False is required for complex Graph objects
            self.cache[chunk_idx] = torch.load(chunk_path, map_location='cpu', weights_only=False)

        return self.cache[chunk_idx][in_chunk_idx]


class ChunkGroupedSampler(Sampler):
    """
    Shuffles chunk order and the items within each chunk, but yields all items of one
    chunk before moving to the next, so each chunk file is loaded once per epoch.
    """
    def __init__(self, indices, chunk_size=10000, generator=None):
        self.chunk_size = chunk_size
        self.generator = generator
        groups = {}
        for position, real_idx in enumerate(indices):
            groups.setdefault(real_idx // chunk_size, []).append(position)
        self.groups = [torch.tensor(g) for g in groups.values()]
        self.num_items = len(indices)

    def __iter__(self):
        for g in torch.randperm(len(self.groups), generator=self.generator).tolist():
            group = self.groups[g]
            yield from group[torch.randperm(len(group), generator=self.generator)].tolist()

    def __len__(self):
        return self.num_items


class LHCH5Dataset(Dataset):
    """Streams events straight from the HDF5 file. The file is opened once per worker."""
    def __init__(self, file_path, start_idx=0, num_events=100000, radius=0.4, wrap_phi=False):
        super().__init__(None, None, None)
        self.file_path = file_path
        self.start_idx = start_idx
        self.num_events = num_events
        self.radius = radius
        self.wrap_phi = wrap_phi
        self._h5_file = None
        self._events = None

    def __getstate__(self):
        # h5py handles cannot be pickled into DataLoader workers; each worker reopens.
        state = self.__dict__.copy()
        state['_h5_file'] = None
        state['_events'] = None
        return state

    def len(self):
        return self.num_events

    def get(self, idx):
        if self._events is None:
            self._h5_file, self._events = open_events(self.file_path)
        row = self._events[self.start_idx + idx, :]

        label = torch.tensor([row[LABEL_COLUMN]], dtype=torch.float)
        graph = create_radius_graph(row, radius=self.radius, wrap_phi=self.wrap_phi)
        if graph is not None:
            graph.y = label
            return graph

        return Data(
            x=torch.zeros((1, 3), dtype=torch.float),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            y=label,
        )
