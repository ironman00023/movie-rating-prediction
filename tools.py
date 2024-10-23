import os

import torch
import torch_geometric
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

EDGE_SIZE = 2
LOSSES_FOLDER = './losses'


def string_to_array(input_string):
    elements = input_string.strip('[]').split(',')
    elements = [element.strip("\' ") for element in elements]
    if elements == ['']:
        return []
    return elements


def extract_person_edge_index_as_tensor(edges_df, role):
    person_from = []
    person_to = []
    for idx, row in edges_df.iterrows():
        if row['role'] != role:
            continue
        person_from.append(row['movie_idx'])
        person_to.append(row['person_idx'])
    if not person_from:
        return torch.tensor([]), torch.tensor([])
    person_from = torch.tensor(person_from)
    person_to = torch.tensor(person_to)

    person_edge_index = torch.concat((person_from, person_to)).reshape(-1, len(person_to)).long()

    return person_edge_index


def save_model_losses(epochs, model_name):
    if not os.path.exists(LOSSES_FOLDER):
        os.mkdir(LOSSES_FOLDER)
    with open(f'{LOSSES_FOLDER}/{model_name}.txt', 'w') as file:
        for item in epochs:
            file.write(f"{item[0].item()}\n")


def plot_train_validation_loss_graph():
    if not os.path.exists(LOSSES_FOLDER):
        return

    plt.figure(figsize=(10, 6))
    for filename in os.listdir(LOSSES_FOLDER):
        if filename.endswith(".txt"):
            file_path = os.path.join(LOSSES_FOLDER, filename)

            with open(file_path, 'r') as file:
                data = [float(line.strip()) for line in file]

            epoch_numbers = list(range(0, len(data) * 5, 5))
            plt.plot(epoch_numbers, data, marker='o', label=filename)

    plt.title('Training losses for models')
    plt.xlabel('Epochs')
    plt.ylim(0.01, 1)
    plt.ylabel('Loss (log)')
    plt.yscale('log')
    plt.legend()
    plt.show()



def draw_graph(data_unidirectional, name, title=None, predictions_df=None):
    graph = torch_geometric.utils.to_networkx(data_unidirectional, to_undirected=False)
    labels_dict = {}
    i = 0
    movies_count = len(data_unidirectional['movie'].label)
    person_count = len(data_unidirectional['person'].label)
    center_index = None
    if predictions_df is not None:
        for _, row in predictions_df.iterrows():
            if row['title'].lower() == name.lower():
                center_index = i
            labels_dict[i] = f"{row['title']}\n{row['imdb_score']:.2f}->{row['predicted_score']:.2f}"
            i += 1
    else:
        for _, node_label in enumerate(data_unidirectional['movie']['label'].tolist()):
            if node_label.lower() == name.lower():
                center_index = i
            labels_dict[i] = node_label
            i += 1
    for _, node_label in enumerate(data_unidirectional['person'].label.tolist()):
        if node_label.lower() == name.lower():
            center_index = i
        labels_dict[i] = node_label
        i += 1
    if 'country' in data_unidirectional.x_dict:
        for _, node_label in enumerate(data_unidirectional['country'].label.tolist()):
            labels_dict[i] = node_label
            i += 1
    if not center_index:
        raise ValueError('not found')
    for edge_type in [('movie', 'acted_by', 'person'), ('movie', 'directed_by', 'person')]:
        edge_index = data_unidirectional[edge_type].edge_index
        edge_weights = data_unidirectional[edge_type].edge_weight

        # Add edge weights to the graph
        for i in range(edge_index.size(1)):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()

            if graph.has_edge(u, movies_count + v):
                graph[u][movies_count + v]['weight'] = edge_weights[i].item()
            else:
                print(f"Warning: Edge ({u}, {v}) not found in the NetworkX graph.")

    if 'country' in data_unidirectional.x_dict:
        edge_type = ("movie", "produced_in", "country")
        edge_index = data_unidirectional[edge_type].edge_index
        edge_weights = data_unidirectional[edge_type].edge_weight

        # Add edge weights to the graph
        for i in range(edge_index.size(1)):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()

            if graph.has_edge(u, movies_count + person_count + v):
                graph[u][movies_count + person_count + v]['weight'] = edge_weights[i].item()
            else:
                print(f"Warning: Edge ({u}, {v}) not found in the NetworkX graph.")

    node_type_colors = {
        "movie": "#4599C3",
        "person": "#ED8546",
        "country": "#FF85FF",
    }
    edge_type_colors = {
        ("movie", "directed_by", "person"): "#8B000E",
        ("movie", "acted_by", "person"): "#8B4D9E",
        ("movie", "produced_in", "country"): "#FF4DFF",
    }
    edge_type_labels = {
        ("movie", "directed_by", "person"): "directed by",
        ("movie", "acted_by", "person"): "acted by",
        ("movie", "produced_in", "country"): "produced in",
    }
    subgraph_nodes = nx.ego_graph(graph, center_index, undirected=True, radius=1)
    subgraph = graph.subgraph(subgraph_nodes)

    node_colors = [node_type_colors[graph.nodes[n]['type']] for n in subgraph.nodes()]
    edge_colors = [edge_type_colors[graph.edges[e]['type']] for e in subgraph.edges()]
    pos = nx.spring_layout(subgraph)
    _, ax = plt.subplots(figsize=(16, 8))
    if title is not None:
        title = name
    if predictions_df is not None and title is not '':
        title += " [actual->prediction]"
    # ax.set_title(title, fontsize=16)

    node_legend_handles = [
        Line2D([0], [0], marker='o', color='w', label='Person Node',
               markerfacecolor=node_type_colors['person'], markersize=15),
        Line2D([0], [0], marker='o', color='w', label='Movie Node [actual->prediction]',
               markerfacecolor=node_type_colors['movie'], markersize=15)
    ]

    edge_legend_handles = [
        Line2D([0], [0], color=edge_type_colors[("movie", "acted_by", "person")], lw=2, label='acted by'),
        Line2D([0], [0], color=edge_type_colors[("movie", "directed_by", "person")], lw=2, label='directed by')
    ]

    legend_handles = node_legend_handles + edge_legend_handles

    ax.legend(handles=legend_handles, loc='lower left')
    ax.axis('off')

    nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=2000, ax=ax)

    edge_weights = [(graph.edges[e].get('weight') + 1) * EDGE_SIZE for e in subgraph.edges()]
    nx.draw_networkx_edges(subgraph, pos, edge_color=edge_colors, width=edge_weights, ax=ax)
    nx.draw_networkx_labels(subgraph, pos, {k: v for k, v in labels_dict.items() if k in set(subgraph)}, ax=ax)
    nx.draw_networkx_edge_labels(
        subgraph,
        pos,
        edge_labels={
            (e[0], e[1]): f"{edge_type_labels[e[2]['type']]} [{e[2]['weight']:.2f}]"
            for e in subgraph.edges(data=True)
        },
        ax=ax
    )


def draw_graph_with_2_radius_excluding_countries(data_unidirectional, name, title, predictions_df):
    graph = torch_geometric.utils.to_networkx(data_unidirectional, to_undirected=False)
    labels_dict = {}
    i = 0
    movies_count = len(data_unidirectional['movie'].label)
    person_count = len(data_unidirectional['person'].label)
    center_index = None

    for movie, node_label in enumerate(data_unidirectional['movie']['label'].tolist()):
        score = predictions_df[predictions_df['title'] == node_label]['imdb_score'].values[0]
        if node_label.lower() == name.lower():
            center_index = i
            prediction = predictions_df[predictions_df['title'] == node_label]['predicted_score'].values[0]
            labels_dict[i] = f"{node_label}\n{score:.2f}->{prediction:.2f}"
        else:
            labels_dict[i] = f"{score:.2f}"
        i += 1
    for _, node_label in enumerate(data_unidirectional['person'].label.tolist()):
        if node_label.lower() == name.lower():
            center_index = i
        labels_dict[i] = node_label
        i += 1
    if not center_index:
        raise ValueError('not found')
    for edge_type in [('movie', 'acted_by', 'person'), ('movie', 'directed_by', 'person')]:
        edge_index = data_unidirectional[edge_type].edge_index
        edge_weights = data_unidirectional[edge_type].edge_weight

        for i in range(edge_index.size(1)):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()

            if graph.has_edge(u, movies_count + v):
                graph[u][movies_count + v]['weight'] = edge_weights[i].item()
            else:
                print(f"Warning: Edge ({u}, {v}) not found in the NetworkX graph.")

    node_type_colors = {
        "movie": "#4599C3",
        "person": "#ED8546",
    }
    edge_type_colors = {
        ("movie", "directed_by", "person"): "#8B000E",
        ("movie", "acted_by", "person"): "#8B4D9E",
    }
    node_types = {}
    for idx in graph.nodes():
        if idx < movies_count:
            node_types[idx] = 'movie'
        elif idx < movies_count + person_count:
            node_types[idx] = 'person'
        else:
            node_types[idx] = 'country'

    country_nodes = {node for node, ntype in node_types.items() if ntype == 'country'}

    graph_no_country = graph.copy()
    graph_no_country.remove_nodes_from(country_nodes)
    subgraph_nodes = set()
    ego_graph = nx.ego_graph(graph_no_country, center_index, undirected=True, radius=2)
    subgraph_nodes.update(ego_graph.nodes)
    subgraph = graph.subgraph(subgraph_nodes)

    node_colors = [node_type_colors[graph.nodes[n]['type']] for n in subgraph.nodes()]
    edge_colors = [edge_type_colors[graph.edges[e]['type']] for e in subgraph.edges()]
    neighbors = list(subgraph.neighbors(center_index))
    remaining_nodes = set(subgraph.nodes()) - {center_index} - set(neighbors)
    shells = [[center_index], neighbors, list(remaining_nodes)]
    pos = nx.shell_layout(subgraph, nlist=shells)
    _, ax = plt.subplots(figsize=(14, 8))
    ax.set_title(title, fontsize=16)

    node_legend_handles = [
        Line2D([0], [0], marker='o', color='w', label='Person Node',
               markerfacecolor=node_type_colors['person'], markersize=15),
        Line2D([0], [0], marker='o', color='w', label='Movie Node [actual->prediction]',
               markerfacecolor=node_type_colors['movie'], markersize=15)
    ]

    edge_legend_handles = [
        Line2D([0], [0], color=edge_type_colors[("movie", "acted_by", "person")], lw=2, label='acted by'),
        Line2D([0], [0], color=edge_type_colors[("movie", "directed_by", "person")], lw=2, label='directed by')
    ]

    legend_handles = node_legend_handles + edge_legend_handles

    ax.legend(handles=legend_handles, loc='lower left')
    ax.axis('off')

    nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=4200, ax=ax)
    edge_weights = [(graph.edges[e].get('weight') + 1) * EDGE_SIZE for e in subgraph.edges()]
    nx.draw_networkx_edges(subgraph, pos, edge_color=edge_colors, width=edge_weights, ax=ax)
    nx.draw_networkx_labels(subgraph, pos, {k: v for k, v in labels_dict.items() if k in set(subgraph)}, ax=ax)
