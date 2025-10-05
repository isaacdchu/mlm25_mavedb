'''
Example of using the ESM Forge SDK to compute and visualize protein embeddings
This will use credits from your ESM account
Code adapted from the ESM tutorial:
https://github.com/evolutionaryscale/esm/blob/main/cookbook/tutorials/2_embed.ipynb
"ask.csv" taken from https://www.biorxiv.org/content/10.1101/2024.10.23.619915v1
'''

from typing import Sequence

from esm.models.esmc import ESMC
from esm.sdk.api import (
    ESMProtein,
    ESMProteinTensor,
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

N_KMEANS_CLUSTERS: int = 3
EMBEDDING_CONFIG: LogitsConfig = LogitsConfig(
    sequence=True, return_embeddings=True
)

def batch_embed(
    model: ESMC,
    inputs: Sequence[ProteinType],
    df: pd.DataFrame
) -> Sequence[LogitsOutput]:
    """
    Forge supports auto-batching. So batch_embed() is as simple as running a collection
    of embed calls in parallel using asyncio.
    Args:
        model (ESMC): A pretrained ESM C model
        inputs (Sequence[ProteinType]): A sequence of protein sequences or ESMProtein objects
    Returns:
        Sequence[LogitsOutput]: A sequence of outputs containing logits and embeddings
    """
    results: list[LogitsOutput] = []
    row_num: int = 0
    for protein_sequence in inputs:
        protein: ESMProtein = ESMProtein(sequence=str(protein_sequence))
        protein_tensor: ESMProteinTensor = model.encode(protein)
        try:
            output = model.logits(protein_tensor, EMBEDDING_CONFIG)
            results.append(output)
        except Exception as e:
            print(f"Error embedding sequence: {e}")
            # Remove the corresponding row from df
            df.drop(index=row_num, inplace=True)
        finally:
            row_num += 1
    return results

def get_rand_indices(all_embeddings: list[torch.Tensor], df: pd.DataFrame) -> list[float]:
    """
    Compute the Rand indices for clustering at each layer of the model.
    Args:
        all_embeddings (list[torch.Tensor]): List of embeddings for each sequence
        df (pd.DataFrame): DataFrame containing metadata for coloring the plot
    Returns:
        list[float]: List of Rand indices for each layer
    """
    mean_embeddings = torch.stack([embedding.mean(dim=0) for embedding in all_embeddings])
    rand_indices: list[float] = []
    # mean_embeddings: [num_samples, num_layers, hidden_size]
    # We want to compute clustering for each layer across samples
    num_layers = mean_embeddings.shape[1]
    for layer in range(num_layers):
        layer_embeddings = mean_embeddings[:, layer].numpy().reshape(-1, 1)
        pca = PCA(n_components=1)
        projected_layer_embeddings = pca.fit_transform(layer_embeddings)
        kmeans = KMeans(n_clusters=N_KMEANS_CLUSTERS, random_state=0).fit(
            projected_layer_embeddings
        )
        rand_index = adjusted_rand_score(df["lid_type"], kmeans.labels_)
        rand_indices.append(rand_index)
    return rand_indices

def plot_embeddings(
    all_mean_embeddings: list[torch.Tensor],
    df: pd.DataFrame
) -> None:
    """
    Plot the embeddings
    """
    # Each embedding is of shape [1152]
    # Project each sequence's embedding to 2D using PCA
    # Color each point by its lid_type
    pca = PCA(n_components=2)
    all_mean_embeddings_stack = torch.stack(all_mean_embeddings).numpy()
    projected_embeddings = pca.fit_transform(all_mean_embeddings_stack)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=projected_embeddings[:, 0],
        y=projected_embeddings[:, 1],
        hue=df["lid_type"],
        palette="Set1",
        s=100,
    )
    plt.title("PCA of Protein Embeddings Colored by Lid Type")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Lid Type")
    plt.grid(True)
    plt.tight_layout()

def get_model(model_type: str, device: str) -> ESMC:
    """
    Initialize and return a pretrained ESM C model.
    Args:
        model_type (str): The type of ESM model to use (e.g., "esmc_600m")
    Returns:
        ESMC: An instantiated pretrained ESM C model
    """
    client: ESMC = ESMC.from_pretrained(model_type).to(device) # cuda or cpu
    return client

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
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model: ESMC = get_model("esmc_600m", device)
    adk_path = "adk.csv"
    df = pd.read_csv(adk_path)
    df = df[["org_name", "sequence", "lid_type", "temperature"]]
    df = df[df["lid_type"] != "other"]  # drop one structural class for simplicity

    # You may see some error messages due to rate limits on each Forge account,
    # but this will retry until the embedding job is completed.
    # This may take a few minutes to run
    outputs: Sequence[LogitsOutput] = batch_embed(model, df["sequence"].tolist(), df)
    all_embeddings: list[torch.Tensor] = []
    for output in outputs:
        if isinstance(output, ESMProteinError):
            print(f"Error embedding sequence: {output}")
            print(f"{output}")
            continue
        if not isinstance(output.embeddings, torch.Tensor):
            print(f"Error: embeddings of {output} \
                  is not a tensor: {type(output.embeddings)}")
            continue
        all_embeddings.append(output.embeddings[0].mean(dim=0))

    print("embedding shape:", all_embeddings[0].shape)
    plot_embeddings(all_embeddings, df)
    plt.show()

if __name__ == "__main__":
    main()
