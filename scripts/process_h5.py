import h5py
import torch
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm
import os

def process_h5_to_graphs(h5_path, output_dir, chunk_size=10000):
    """
    Converts LHCO 2020 H5 events into PyTorch Geometric graphs.
    Each event becomes a graph where nodes are particles.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"📖 Reading {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        # Accessing the event matrix
        data = f['df']['block0_values'][:]
        
        # Total events (usually 1.1 million in R&D set)
        num_events = data.shape[0]
        num_chunks = int(np.ceil(num_events / chunk_size))

        for c in range(num_chunks):
            chunk_graphs = []
            start, end = c * chunk_size, min((c + 1) * chunk_size, num_events)
            
            for i in range(start, end):
                event = data[i]
                # Last element is the Truth Label (0: QCD, 1: W')
                label = event[-1]
                
                # Reshape particles: (Up to 700 particles, 3 features each)
                particles = event[:-1].reshape(-1, 3)
                # Remove zero-padded particles
                mask = particles[:, 0] > 0
                nodes = torch.tensor(particles[mask], dtype=torch.float)
                
                # Create a fully connected graph (simplest for LHC anomalies)
                # For high-speed training, we can also use Radius Graphs
                num_nodes = nodes.shape[0]
                edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
                
                graph = Data(x=nodes, edge_index=edge_index, y=torch.tensor([label]))
                chunk_graphs.append(graph)
            
            # Save the chunk to the directory specified in your .gitignore
            torch.save(chunk_graphs, f"{output_dir}/data_chunk_{c}.pt")
            print(f"✅ Chunk {c} saved ({len(chunk_graphs)} graphs)")

if __name__ == "__main__":
    h5_file = "dataset/raw/events_anomalydetection_v2.h5"
    out_dir = "dataset/processed"
    process_h5_to_graphs(h5_file, out_dir)