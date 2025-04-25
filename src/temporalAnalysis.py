import json
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd

from .data_processing import DataProcessing


def create_digraph_for_year(year: int) -> nx.DiGraph:
    """
    Create a directed graph for the given year.
    """
    df = pd.read_csv("../data/transfers.csv")
    dp = DataProcessing(df)
    df = dp.process_data()
    df_season = df[df["season"] == year]

    di_graph = nx.from_pandas_edgelist(
        df_season,
        source="team_name",
        target="counter_team_name",
        edge_attr=["total_fee", "is_loan", "same_country"],
        create_using=nx.DiGraph,
    )
    for _, row in df_season.iterrows():
        di_graph.nodes[row["team_name"]]["country"] = row["team_country"]
        di_graph.nodes[row["counter_team_name"]]["country"] = row[
            "counter_team_country"
        ]

    return di_graph


def analyze_transfer_network(G: nx.DiGraph, year: int) -> dict:
    """
    Analyze a directed graph representing football transfer market.

    Parameters:
    -----------
    G : networkx.DiGraph
        A directed graph where nodes are teams and edges represent transfers.
        Edge attributes should include 'total_fee', 'is_loan', and 'same_country'.
        Node attributes should include 'country'.

    Returns:
    --------
    dict
        A dictionary containing various network metrics and statistics.
    """
    metrics = {}

    # Basic graph statistics
    metrics["basic"] = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "is_strongly_connected": nx.is_strongly_connected(G),
        "number_strongly_connected_components": nx.number_strongly_connected_components(
            G
        ),
        "is_weakly_connected": nx.is_weakly_connected(G),
        "number_weakly_connected_components": nx.number_weakly_connected_components(G),
    }

    # Largest connected components
    largest_wcc = max(nx.weakly_connected_components(G), key=len)
    largest_scc = max(nx.strongly_connected_components(G), key=len)
    metrics["connectivity"] = {
        "largest_wcc_size": len(largest_wcc),
        "largest_wcc_percentage": len(largest_wcc) / G.number_of_nodes() * 100,
        "largest_scc_size": len(largest_scc),
        "largest_scc_percentage": len(largest_scc) / G.number_of_nodes() * 100,
    }

    # Degree statistics
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]
    metrics["degree"] = {
        "max_in_degree": max(in_degrees) if in_degrees else 0,
        "min_in_degree": min(in_degrees) if in_degrees else 0,
        "avg_in_degree": np.mean(in_degrees) if in_degrees else 0,
        "median_in_degree": np.median(in_degrees) if in_degrees else 0,
        "max_out_degree": max(out_degrees) if out_degrees else 0,
        "min_out_degree": min(out_degrees) if out_degrees else 0,
        "avg_out_degree": np.mean(out_degrees) if out_degrees else 0,
        "median_out_degree": np.median(out_degrees) if out_degrees else 0,
    }

    # Top teams by in/out degree (buyers and sellers)
    top_buyers = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)[:10]
    top_sellers = sorted(G.out_degree(), key=lambda x: x[1], reverse=True)[:10]
    metrics["top_teams"] = {
        "top_buyers": top_buyers,
        "top_sellers": top_sellers,
    }

    # Centrality measures (computed on largest weakly connected component to be more efficient)
    G_wcc = G.subgraph(largest_wcc).copy()

    # Betweenness centrality (teams that act as bridges in the transfer market)
    betweenness = nx.betweenness_centrality(G_wcc)
    top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]

    # Eigenvector centrality (influential teams)
    try:
        eigenvector = nx.eigenvector_centrality(G_wcc, max_iter=1000)
        top_eigenvector = sorted(eigenvector.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]
    except:
        top_eigenvector = (
            "Could not compute eigenvector centrality (convergence issues)"
        )

    # PageRank (important teams in network)
    pagerank = nx.pagerank(G_wcc)
    top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]

    metrics["centrality"] = {
        "top_betweenness": top_betweenness,
        "top_eigenvector": top_eigenvector,
        "top_pagerank": top_pagerank,
    }

    # Financial analysis
    if "total_fee" in G.edges[next(iter(G.edges()))]:
        total_transfer_volume = sum(
            data["total_fee"]
            for _, _, data in G.edges(data=True)
            if data["total_fee"] is not None
        )
        avg_transfer_fee = (
            total_transfer_volume / G.number_of_edges()
            if G.number_of_edges() > 0
            else 0
        )

        # Net spending for each team
        net_spending = {}
        for node in G.nodes():
            incoming = sum(
                data["total_fee"]
                for u, v, data in G.in_edges(node, data=True)
                if data["total_fee"] is not None
            )
            outgoing = sum(
                data["total_fee"]
                for u, v, data in G.out_edges(node, data=True)
                if data["total_fee"] is not None
            )
            net_spending[node] = incoming - outgoing

        # Top spenders and sellers by volume
        top_spenders = sorted(net_spending.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]
        top_income = sorted(net_spending.items(), key=lambda x: x[1])[:10]

        metrics["financial"] = {
            "total_transfer_volume": total_transfer_volume,
            "avg_transfer_fee": avg_transfer_fee,
            "top_spenders": top_spenders,
            "top_income_generators": top_income,
        }

    # Geographical analysis
    if all("country" in G.nodes[n] for n in G.nodes()):
        countries = [G.nodes[n]["country"] for n in G.nodes()]
        country_count = Counter(countries)

        # Count domestic vs international transfers
        domestic_transfers = sum(
            1 for _, _, data in G.edges(data=True) if data.get("same_country") is True
        )
        international_transfers = sum(
            1 for _, _, data in G.edges(data=True) if data.get("same_country") is False
        )

        # Transfer relationships between countries
        country_relationships = {}
        for u, v, data in G.edges(data=True):
            source_country = G.nodes[u]["country"]
            target_country = G.nodes[v]["country"]

            if source_country != target_country:
                country_pair = (source_country, target_country)
                if country_pair not in country_relationships:
                    country_relationships[country_pair] = 0
                country_relationships[country_pair] += 1

        top_country_relationships = sorted(
            country_relationships.items(), key=lambda x: x[1], reverse=True
        )[:10]

        metrics["geographical"] = {
            "country_distribution": dict(country_count),
            "domestic_transfers": domestic_transfers,
            "international_transfers": international_transfers,
            "domestic_percentage": domestic_transfers / G.number_of_edges() * 100
            if G.number_of_edges() > 0
            else 0,
            "top_country_relationships": top_country_relationships,
        }

    # Loan analysis
    if "is_loan" in G.edges[next(iter(G.edges()))]:
        loan_count = sum(
            1 for _, _, data in G.edges(data=True) if data["is_loan"] is True
        )
        permanent_count = sum(
            1 for _, _, data in G.edges(data=True) if data["is_loan"] is False
        )

        metrics["loan_analysis"] = {
            "loan_transfers": loan_count,
            "permanent_transfers": permanent_count,
            "loan_percentage": loan_count / G.number_of_edges() * 100
            if G.number_of_edges() > 0
            else 0,
        }

    # Community detection
    try:
        communities = nx.community.greedy_modularity_communities(G.to_undirected())
        metrics["communities"] = {
            "number_of_communities": len(communities),
            "largest_community_size": len(communities[0]) if communities else 0,
            "modularity_score": nx.community.modularity(G.to_undirected(), communities),
        }
    except:
        metrics["communities"] = (
            "Could not compute communities (likely due to graph structure)"
        )

    # Path analysis
    if nx.is_strongly_connected(G_wcc):
        metrics["paths"] = {
            "average_shortest_path_length": nx.average_shortest_path_length(G_wcc),
            "diameter": nx.diameter(G_wcc),
        }
    else:
        metrics["paths"] = "Graph is not connected, path metrics are undefined"

    # Reciprocity (teams that transfer players back and forth)
    metrics["reciprocity"] = {
        "overall_reciprocity": nx.overall_reciprocity(G),
        "edge_reciprocity": nx.reciprocity(G),
    }

    # Advanced structural metrics
    metrics["structural"] = {
        "transitivity": nx.transitivity(G),
        "average_clustering": nx.average_clustering(G),
    }

    with open(f"../market_analysis/{year}.json", "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


def print_network_report(metrics: dict) -> None:
    """
    Print a human-readable report of the network metrics.

    Parameters:
    -----------
    metrics : dict
        Output from analyze_transfer_network function.
    """
    print("=== FOOTBALL TRANSFER MARKET NETWORK ANALYSIS ===\n")

    print("BASIC STATISTICS:")
    print(f"Number of teams: {metrics['basic']['nodes']}")
    print(f"Number of transfers: {metrics['basic']['edges']}")
    print(f"Network density: {metrics['basic']['density']:.4f}")
    print(f"Is strongly connected: {metrics['basic']['is_strongly_connected']}")
    print(
        f"Number of strongly connected components: {metrics['basic']['number_strongly_connected_components']}"
    )
    print(f"Is weakly connected: {metrics['basic']['is_weakly_connected']}")
    print(
        f"Number of weakly connected components: {metrics['basic']['number_weakly_connected_components']}"
    )
    print()

    print("CONNECTIVITY:")
    print(
        f"Largest weakly connected component: {metrics['connectivity']['largest_wcc_size']} teams "
        f"({metrics['connectivity']['largest_wcc_percentage']:.1f}% of network)"
    )
    print(
        f"Largest strongly connected component: {metrics['connectivity']['largest_scc_size']} teams "
        f"({metrics['connectivity']['largest_scc_percentage']:.1f}% of network)"
    )
    print()

    print("DEGREE DISTRIBUTION:")
    print(
        f"In-degree (incoming transfers): max={metrics['degree']['max_in_degree']}, "
        f"min={metrics['degree']['min_in_degree']}, avg={metrics['degree']['avg_in_degree']:.2f}, "
        f"median={metrics['degree']['median_in_degree']}"
    )
    print(
        f"Out-degree (outgoing transfers): max={metrics['degree']['max_out_degree']}, "
        f"min={metrics['degree']['min_out_degree']}, avg={metrics['degree']['avg_out_degree']:.2f}, "
        f"median={metrics['degree']['median_out_degree']}"
    )
    print()

    print("TOP TEAMS:")
    print("Top 10 buyers (highest in-degree):")
    for i, (team, degree) in enumerate(metrics["top_teams"]["top_buyers"], 1):
        print(f"  {i}. {team}: {degree} incoming transfers")

    print("\nTop 10 sellers (highest out-degree):")
    for i, (team, degree) in enumerate(metrics["top_teams"]["top_sellers"], 1):
        print(f"  {i}. {team}: {degree} outgoing transfers")
    print()

    print("CENTRALITY MEASURES:")
    print("Top 10 teams by betweenness centrality (market intermediaries):")
    for i, (team, score) in enumerate(metrics["centrality"]["top_betweenness"], 1):
        print(f"  {i}. {team}: {score:.4f}")

    if isinstance(metrics["centrality"]["top_eigenvector"], list):
        print(
            "\nTop 10 teams by eigenvector centrality (connected to important teams):"
        )
        for i, (team, score) in enumerate(metrics["centrality"]["top_eigenvector"], 1):
            print(f"  {i}. {team}: {score:.4f}")

    print("\nTop 10 teams by PageRank (overall market importance):")
    for i, (team, score) in enumerate(metrics["centrality"]["top_pagerank"], 1):
        print(f"  {i}. {team}: {score:.4f}")
    print()

    if "financial" in metrics:
        print("FINANCIAL ANALYSIS:")
        print(
            f"Total transfer market volume: {metrics['financial']['total_transfer_volume']:,.2f}"
        )
        print(f"Average transfer fee: {metrics['financial']['avg_transfer_fee']:,.2f}")

        print("\nTop 10 net spenders:")
        for i, (team, amount) in enumerate(metrics["financial"]["top_spenders"], 1):
            print(f"  {i}. {team}: {amount:,.2f}")

        print("\nTop 10 net income generators:")
        for i, (team, amount) in enumerate(
            metrics["financial"]["top_income_generators"], 1
        ):
            print(f"  {i}. {team}: {-amount:,.2f}")  # Negate to show as positive income
        print()

    if "geographical" in metrics:
        print("GEOGRAPHICAL ANALYSIS:")
        print("Top countries by number of teams:")
        for i, (country, count) in enumerate(
            sorted(
                metrics["geographical"]["country_distribution"].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10],
            1,
        ):
            print(f"  {i}. {country}: {count} teams")

        print(
            f"\nDomestic transfers: {metrics['geographical']['domestic_transfers']} "
            f"({metrics['geographical']['domestic_percentage']:.1f}%)"
        )
        print(
            f"International transfers: {metrics['geographical']['international_transfers']} "
            f"({100 - metrics['geographical']['domestic_percentage']:.1f}%)"
        )

        print("\nTop transfer relationships between countries:")
        for i, ((source, target), count) in enumerate(
            metrics["geographical"]["top_country_relationships"], 1
        ):
            print(f"  {i}. {source} → {target}: {count} transfers")
        print()

    if "loan_analysis" in metrics:
        print("LOAN ANALYSIS:")
        print(
            f"Loan transfers: {metrics['loan_analysis']['loan_transfers']} "
            f"({metrics['loan_analysis']['loan_percentage']:.1f}%)"
        )
        print(
            f"Permanent transfers: {metrics['loan_analysis']['permanent_transfers']} "
            f"({100 - metrics['loan_analysis']['loan_percentage']:.1f}%)"
        )
        print()

    if "communities" in metrics and isinstance(metrics["communities"], dict):
        print("COMMUNITY STRUCTURE:")
        print(
            f"Number of communities: {metrics['communities']['number_of_communities']}"
        )
        print(
            f"Largest community size: {metrics['communities']['largest_community_size']} teams"
        )
        print(f"Modularity score: {metrics['communities']['modularity_score']:.4f}")
        print()

    if "paths" in metrics and isinstance(metrics["paths"], dict):
        print("PATH ANALYSIS:")
        print(
            f"Average shortest path length: {metrics['paths']['average_shortest_path_length']:.4f}"
        )
        print(f"Network diameter: {metrics['paths']['diameter']}")
        print()

    print("RECIPROCITY:")
    print(f"Overall reciprocity: {metrics['reciprocity']['overall_reciprocity']:.4f}")
    print(f"Edge reciprocity: {metrics['reciprocity']['edge_reciprocity']:.4f}")
    print()

    print("STRUCTURAL METRICS:")
    print(f"Transitivity: {metrics['structural']['transitivity']:.4f}")
    print(
        f"Average clustering coefficient: {metrics['structural']['average_clustering']:.4f}"
    )
    print()
