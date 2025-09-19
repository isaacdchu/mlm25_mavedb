'''
Example of using the ESM Forge SDK to compute and visualize protein embeddings
This will use credits from your ESM account
Code adapted from the ESM tutorial:
https://github.com/evolutionaryscale/esm/blob/main/cookbook/tutorials/2_embed.ipynb
"ask.csv" taken from https://www.biorxiv.org/content/10.1101/2024.10.23.619915v1
'''

import os
from concurrent.futures import ThreadPoolExecutor, CancelledError
from typing import Sequence

from dotenv import load_dotenv
from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    ESMProteinError,
    LogitsConfig,
    LogitsOutput,
    ProteinType,
)
from esm.sdk import client
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

N_KMEANS_CLUSTERS: int = 3
EMBEDDING_CONFIG: LogitsConfig = LogitsConfig(
    sequence=True, return_embeddings=True, return_hidden_states=True
)

def embed_sequence(model: ESM3InferenceClient, sequence: ProteinType) -> LogitsOutput:
    """
    Embed a single protein sequence using the provided ESM model.
    Args:
        model (ESM3InferenceClient): An instantiated ESM Forge client
        sequence (ProteinType): A protein sequence string or ESMProtein object
    Returns:
        LogitsOutput: The output containing logits and embeddings
    """
    protein = ESMProtein(sequence=str(sequence))
    protein_tensor = model.encode(protein)
    output = model.logits(protein_tensor, EMBEDDING_CONFIG)
    return output


def batch_embed(
    model: ESM3InferenceClient,
    inputs: Sequence[ProteinType]
) -> Sequence[LogitsOutput]:
    """
    Forge supports auto-batching. So batch_embed() is as simple as running a collection
    of embed calls in parallel using asyncio.
    Args:
        model (ESM3InferenceClient): An instantiated ESM Forge client
        inputs (Sequence[ProteinType]): A sequence of protein sequences or ESMProtein objects
    Returns:
        Sequence[LogitsOutput]: A sequence of outputs containing logits and embeddings
    """
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(embed_sequence, model, protein) for protein in inputs
        ]
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except (CancelledError, TimeoutError) as e:
                results.append(ESMProteinError(500, str(e)))
    return results

def plot_embeddings_at_layer(
    all_mean_embeddings: list[torch.Tensor],
    layer_idx: int,
    df: pd.DataFrame
) -> None:
    """
    Plot the mean embeddings at a specific layer using PCA and KMeans clustering.
    Args:
        all_mean_embeddings (list[torch.Tensor]): List of mean embeddings for each sequence
        layer_idx (int): The layer index to visualize
        df (pd.DataFrame): DataFrame containing metadata for coloring the plot
    """
    stacked_mean_embeddings = torch.stack(
        [embedding[layer_idx, :] for embedding in all_mean_embeddings]
    ).to(torch.float32).numpy()

    # project all the embeddings to 2D using PCA
    pca = PCA(n_components=2)
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

def get_model(model_type: str) -> ESM3InferenceClient:
    """
    Initialize and return an ESM3InferenceClient model using the provided model type.
    Requires the ESM_API_KEY environment variable to be set.
    Args:
        model_type (str): The type of ESM model to use (e.g., "esmc-300m-2024-12")
    Returns:
        ESM3InferenceClient: An instantiated ESM Forge client
    Raises:
        ValueError: If the ESM_API_KEY environment variable is not set
    """
    load_dotenv()  # take environment variables from .env file
    token: str | None = os.environ.get("ESM_API_KEY", None)
    if token is None:
        raise ValueError("ESM_API_KEY environment variable not set")
    model: ESM3InferenceClient = client(
        model=model_type,
        url="https://forge.evolutionaryscale.ai",
        token=token,
    )
    del token
    return model

def main() -> None:
    """
    Main function to load data, compute embeddings, and visualize them.
    1. Load the dataset from "adk.csv".
    2. Filter out rows with "lid_type" as "other".
    3. Initialize the ESM model using an API key from environment variables.
    4. Compute embeddings for the protein sequences in the dataset.
    5. Summarize embeddings by taking the mean across the sequence dimension.
    6. Visualize the embeddings at specified layers using PCA and KMeans clustering.
    """
    # Initialize the model and data
    model: ESM3InferenceClient = get_model("esmc-600m-2024-12")
    adk_path = "adk.csv"
    df = pd.read_csv(adk_path)
    df = df[["org_name", "sequence", "lid_type", "temperature"]]
    df = df[df["lid_type"] != "other"]  # drop one structural class for simplicity

    # You may see some error messages due to rate limits on each Forge account,
    # but this will retry until the embedding job is complete
    # This may take a few minutes to run
    outputs: Sequence[LogitsOutput] = batch_embed(model, df["sequence"].tolist())
    all_hidden_states: list[torch.Tensor] = []
    for output in outputs:
        if isinstance(output, ESMProteinError):
            print(f"Error embedding sequence: {output}")
            print(f"{output}")
            continue
        if not isinstance(output.hidden_states, torch.Tensor):
            print(f"Error: hidden_states of {output} \
                  is not a tensor: {type(output.hidden_states)}")
            continue
        all_hidden_states.append(output.hidden_states)

    # we'll summarize the embeddings using their mean across the sequence dimension
    # which allows us to compare embeddings for sequences of different lengths
    all_mean_embeddings: list[torch.Tensor] = [
        torch.mean(hidden_states, dim=-2).squeeze() for hidden_states in all_hidden_states
    ]

    # now we have a list of tensors of [num_layers, hidden_size]
    print("embedding shape [num_layers, hidden_size]:", all_mean_embeddings[0].shape)
    plot_embeddings_at_layer(all_mean_embeddings, layer_idx=30, df=df)
    plot_embeddings_at_layer(all_mean_embeddings, layer_idx=12, df=df)
    plot_embeddings_at_layer(all_mean_embeddings, layer_idx=0, df=df)
    plt.show()

if __name__ == "__main__":
    main()
