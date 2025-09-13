'''
Example of using the ESM Forge SDK to compute and visualize protein embeddings
This will use credits from your ESM account
Code adapted from the ESM tutorial:
https://github.com/evolutionaryscale/esm/blob/main/cookbook/tutorials/2_embed.ipynb
'''

import os
from concurrent.futures import CancelledError, TimeoutError, ThreadPoolExecutor
from typing import Sequence
from esm.sdk import client

from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    ESMProteinError,
    LogitsConfig,
    LogitsOutput,
    ProteinType,
)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import torch

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

N_KMEANS_CLUSTERS = 3

EMBEDDING_CONFIG = LogitsConfig(
    sequence=True, return_embeddings=True, return_hidden_states=True
)

def embed_sequence(model: ESM3InferenceClient, sequence: str) -> LogitsOutput:
    protein = ESMProtein(sequence=sequence)
    protein_tensor = model.encode(protein)
    output = model.logits(protein_tensor, EMBEDDING_CONFIG)
    return output

def batch_embed(
    model: ESM3InferenceClient, inputs: Sequence[ProteinType]
) -> Sequence[LogitsOutput]:
    """Forge supports auto-batching. So batch_embed() is as simple as running a collection
    of embed calls in parallel using asyncio.
    """
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(embed_sequence, model, str(protein)) for protein in inputs
        ]
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except (CancelledError, TimeoutError) as e:
                results.append(ESMProteinError(500, str(e)))
                print(f"Error embedding sequence: {e}")
    return results

def plot_embeddings_at_layer(all_mean_embeddings: list[torch.Tensor], layer_idx: int,
                             df: pd.DataFrame):
    stacked_mean_embeddings = torch.stack(
        [embedding[layer_idx, :] for embedding in all_mean_embeddings]
    ).detach().cpu().to(torch.float32).numpy()

    # project all the embeddings to 2D using PCA
    pca = PCA(n_components=2, random_state=0)
    pca.fit(stacked_mean_embeddings)
    projected_mean_embeddings = pca.transform(stacked_mean_embeddings)

    # compute kmeans purity as a measure of how good the clustering is
    kmeans = KMeans(n_clusters=N_KMEANS_CLUSTERS, random_state=0).fit(
        projected_mean_embeddings
    )
    rand_index = adjusted_rand_score(df["lid_type"], kmeans.labels_)

    # plot the clusters
    plt.figure(figsize=(4, 4))
    sns.scatterplot(
        x=projected_mean_embeddings[:, 0],
        y=projected_mean_embeddings[:, 1],
        hue=df["lid_type"],
    )
    plt.title(
        f"PCA of mean embeddings at layer {layer_idx}.\nRand index: {rand_index:.2f}"
    )
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.show()

def main():
    print("Verifying model access...")
    token=os.getenv("ESM_KEY")
    if token is None:
        raise ValueError("Environment variable 'ESM_KEY' is not set.")
    model = client(
        model="esmc-6b-2024-12", url="https://forge.evolutionaryscale.ai", token=token
    )
    print("Model loaded.")
    adk_path = "adk.csv"
    print(f"Reading data from {adk_path}...")
    df = pd.read_csv(adk_path)
    print("Data loaded.")
    df = df[["org_name", "sequence", "lid_type", "temperature"]]
    df = df[df["lid_type"] != "other"]  # drop one structural class for simplicity
    # shuffle the data
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    # take a subset of the data for faster processing
    df = df.iloc[:50]
    print(f"Computing embeddings for {len(df)} sequences...")
    sequence_list: list[ProteinType] = df["sequence"].tolist()
    outputs = batch_embed(model, sequence_list)
    good_outputs = [o.hidden_states for o in outputs if isinstance(o.hidden_states, torch.Tensor)]

    # we'll summarize the embeddings using their mean across the sequence dimension
    # which allows us to compare embeddings for sequences of different lengths
    all_mean_embeddings = [
        torch.mean(output, dim=-2).squeeze() for output in good_outputs
    ]

    # now we have a list of tensors of [num_layers, hidden_size]
    print("embedding shape [num_layers, hidden_size]:", all_mean_embeddings[0].shape)

    plot_embeddings_at_layer(all_mean_embeddings, layer_idx=30, df=df)
    plot_embeddings_at_layer(all_mean_embeddings, layer_idx=12, df=df)

if __name__ == "__main__":
    main()
