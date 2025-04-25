import glob
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Set page config
st.set_page_config(
    page_title="Football Transfer Market Analysis", page_icon="⚽", layout="wide"
)

# Dashboard title
st.title("⚽ Football Transfer Market Analysis (2009-2021)")
st.markdown(
    "Analyzing trends and patterns in the global football transfer market network"
)


# Function to load all JSON files from the specified directory
@st.cache_data
def load_all_data(directory="market_analysis"):
    data_by_year = {}

    # Get all JSON files in the directory
    json_files = glob.glob(f"{directory}/*.json")

    for file_path in json_files:
        try:
            # Extract the year from the filename (assuming filename is YYYY.json)
            year = int(Path(file_path).stem)

            # Load the JSON data
            with open(file_path, "r") as f:
                data = json.load(f)

            data_by_year[year] = data
        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")

    return dict(sorted(data_by_year.items()))  # Sort by year


# Load data
data_by_year = load_all_data()
years = list(data_by_year.keys())

if not data_by_year:
    st.error("No data files found. Please check the 'market_analysis' directory.")
    st.stop()


# Extract data for all years to create trend DataFrames
def extract_trend_data():
    # Financial trends
    financial_data = {
        "Year": [],
        "Total Transfer Volume": [],
        "Average Transfer Fee": [],
    }

    # Network dynamics
    network_data = {
        "Year": [],
        "Nodes": [],
        "Edges": [],
        "Density": [],
        "Avg In-Degree": [],
        "Avg Out-Degree": [],
    }

    # Geographical data
    geo_data = {
        "Year": [],
        "Domestic Transfers": [],
        "International Transfers": [],
        "Domestic Percentage": [],
    }

    # Community and reciprocity data
    community_data = {
        "Year": [],
        "Number of Communities": [],
        "Largest Community Size": [],
        "Modularity Score": [],
        "Reciprocity": [],
    }

    for year, data in data_by_year.items():
        # Financial data
        financial_data["Year"].append(year)
        financial_data["Total Transfer Volume"].append(
            data["financial"]["total_transfer_volume"]
        )
        financial_data["Average Transfer Fee"].append(
            data["financial"]["avg_transfer_fee"]
        )

        # Network data
        network_data["Year"].append(year)
        network_data["Nodes"].append(data["basic"]["nodes"])
        network_data["Edges"].append(data["basic"]["edges"])
        network_data["Density"].append(data["basic"]["density"])
        network_data["Avg In-Degree"].append(data["degree"]["avg_in_degree"])
        network_data["Avg Out-Degree"].append(data["degree"]["avg_out_degree"])

        # Geographical data
        geo_data["Year"].append(year)
        geo_data["Domestic Transfers"].append(
            data["geographical"]["domestic_transfers"]
        )
        geo_data["International Transfers"].append(
            data["geographical"]["international_transfers"]
        )
        geo_data["Domestic Percentage"].append(
            data["geographical"]["domestic_percentage"]
        )

        # Community and reciprocity data
        community_data["Year"].append(year)
        community_data["Number of Communities"].append(
            data["communities"]["number_of_communities"]
        )
        community_data["Largest Community Size"].append(
            data["communities"]["largest_community_size"]
        )
        community_data["Modularity Score"].append(
            data["communities"]["modularity_score"]
        )
        community_data["Reciprocity"].append(data["reciprocity"]["overall_reciprocity"])

    return {
        "financial": pd.DataFrame(financial_data),
        "network": pd.DataFrame(network_data),
        "geographical": pd.DataFrame(geo_data),
        "community": pd.DataFrame(community_data),
    }


# Create DataFrame for trends
trend_data = extract_trend_data()

# Sidebar for year selection
st.sidebar.header("Filters")
selected_year = st.sidebar.selectbox(
    "Select Year for Detailed View", years, index=len(years) - 1
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Data Range:** 2009-2021")
st.sidebar.markdown("**Total Years:** " + str(len(years)))

# Main dashboard content
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Financial Trends",
        "Network Dynamics",
        "Geographical Analysis",
        "Community & Power",
    ]
)

# Tab 1: Financial Trends
with tab1:
    st.header("📊 Market Volume & Financial Trends")

    # Row 1: Financial trend charts
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            trend_data["financial"]["Year"],
            trend_data["financial"]["Total Transfer Volume"] / 1e9,
            marker="o",
            linewidth=2,
        )
        ax.set_title("Total Transfer Volume Over Time")
        ax.set_xlabel("Year")
        ax.set_ylabel("Volume (Billion €)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            trend_data["financial"]["Year"],
            trend_data["financial"]["Average Transfer Fee"] / 1e6,
            marker="o",
            linewidth=2,
            color="green",
        )
        ax.set_title("Average Transfer Fee Over Time")
        ax.set_xlabel("Year")
        ax.set_ylabel("Average Fee (Million €)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    st.markdown("---")

    # Row 2: Top spenders for selected year
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Top Spenders ({selected_year})")
        top_spenders = data_by_year[selected_year]["financial"]["top_spenders"]

        # Convert to DataFrame for better display
        df_top_spenders = pd.DataFrame(top_spenders, columns=["Club", "Amount (€)"])
        # Convert to millions
        df_top_spenders["Amount (Million €)"] = df_top_spenders["Amount (€)"] / 1e6
        df_top_spenders = df_top_spenders.drop("Amount (€)", axis=1)

        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(
            df_top_spenders["Club"][::-1],
            df_top_spenders["Amount (Million €)"][::-1],
            color="skyblue",
        )
        ax.set_xlabel("Spending (Million €)")
        ax.set_title(f"Top Spenders in {selected_year}")
        ax.grid(True, alpha=0.3, axis="x")
        st.pyplot(fig)

    with col2:
        st.subheader(f"Top Income Generators ({selected_year})")
        top_income = data_by_year[selected_year]["financial"]["top_income_generators"]

        # Convert to DataFrame for better display
        df_top_income = pd.DataFrame(top_income, columns=["Club", "Amount (€)"])
        # Convert to millions and make absolute value
        df_top_income["Amount (Million €)"] = abs(df_top_income["Amount (€)"]) / 1e6
        df_top_income = df_top_income.drop("Amount (€)", axis=1)

        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(
            df_top_income["Club"][::-1],
            df_top_income["Amount (Million €)"][::-1],
            color="coral",
        )
        ax.set_xlabel("Income (Million €)")
        ax.set_title(f"Top Income Generators in {selected_year}")
        ax.grid(True, alpha=0.3, axis="x")
        st.pyplot(fig)

# Tab 2: Network Dynamics
with tab2:
    st.header("🔄 Transfer Network Dynamics")

    # Row 1: Network metrics over time
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            trend_data["network"]["Year"],
            trend_data["network"]["Nodes"],
            marker="o",
            label="Nodes (Teams)",
        )
        ax.plot(
            trend_data["network"]["Year"],
            trend_data["network"]["Edges"],
            marker="s",
            label="Edges (Transfers)",
        )
        ax.set_title("Network Size Over Time")
        ax.set_xlabel("Year")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            trend_data["network"]["Year"],
            trend_data["network"]["Density"],
            marker="o",
            color="purple",
        )
        ax.set_title("Network Density Over Time")
        ax.set_xlabel("Year")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    st.markdown("---")

    # Row 2: Top buyers/sellers for selected year
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Top Buyers ({selected_year})")
        top_buyers = data_by_year[selected_year]["top_teams"]["top_buyers"]

        # Convert to DataFrame for better display
        df_top_buyers = pd.DataFrame(top_buyers, columns=["Club", "In-Degree"])

        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(
            df_top_buyers["Club"][::-1], df_top_buyers["In-Degree"][::-1], color="green"
        )
        ax.set_xlabel("Number of Incoming Transfers")
        ax.set_title(f"Top Buyers in {selected_year}")
        ax.grid(True, alpha=0.3, axis="x")
        st.pyplot(fig)

    with col2:
        st.subheader(f"Top Sellers ({selected_year})")
        top_sellers = data_by_year[selected_year]["top_teams"]["top_sellers"]

        # Convert to DataFrame for better display
        df_top_sellers = pd.DataFrame(top_sellers, columns=["Club", "Out-Degree"])

        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(
            df_top_sellers["Club"][::-1],
            df_top_sellers["Out-Degree"][::-1],
            color="red",
        )
        ax.set_xlabel("Number of Outgoing Transfers")
        ax.set_title(f"Top Sellers in {selected_year}")
        ax.grid(True, alpha=0.3, axis="x")
        st.pyplot(fig)

    # Row 3: Connected components
    st.markdown("---")
    st.subheader("Connected Components")

    col1, col2 = st.columns(2)

    with col1:
        # Extract data for strongly connected components
        years_list = list(data_by_year.keys())
        scc_counts = [
            data_by_year[year]["basic"]["number_strongly_connected_components"]
            for year in years_list
        ]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(years_list, scc_counts, marker="o", color="blue")
        ax.set_title("Number of Strongly Connected Components")
        ax.set_xlabel("Year")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with col2:
        # Extract data for largest strongly connected component
        scc_sizes = [
            data_by_year[year]["connectivity"]["largest_scc_percentage"]
            for year in years_list
        ]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(years_list, scc_sizes, marker="o", color="orange")
        ax.set_title("Largest Strongly Connected Component Size (%)")
        ax.set_xlabel("Year")
        ax.set_ylabel("% of Network")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# Tab 3: Geographical Analysis
with tab3:
    st.header("🌍 Geopolitical Transfer Patterns")

    # Row 1: International vs Domestic transfers
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(
            trend_data["geographical"]["Year"],
            trend_data["geographical"]["Domestic Transfers"],
            label="Domestic Transfers",
            color="blue",
            alpha=0.7,
        )
        ax.bar(
            trend_data["geographical"]["Year"],
            trend_data["geographical"]["International Transfers"],
            bottom=trend_data["geographical"]["Domestic Transfers"],
            label="International Transfers",
            color="red",
            alpha=0.7,
        )
        ax.set_title("Domestic vs International Transfers")
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Transfers")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            trend_data["geographical"]["Year"],
            trend_data["geographical"]["Domestic Percentage"],
            marker="o",
            color="green",
        )
        ax.set_title("Domestic Transfers Percentage")
        ax.set_xlabel("Year")
        ax.set_ylabel("Percentage (%)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)  # Set y-axis from 0 to 100%
        st.pyplot(fig)

    st.markdown("---")

    # Row 2: Country distribution for selected year
    st.subheader(f"Country Distribution ({selected_year})")

    country_data = data_by_year[selected_year]["geographical"]["country_distribution"]
    # Convert to DataFrame and sort by number of teams
    df_countries = pd.DataFrame(
        list(country_data.items()), columns=["Country", "Number of Teams"]
    )
    df_countries = df_countries.sort_values("Number of Teams", ascending=False).head(
        20
    )  # Top 20 countries

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(df_countries["Country"], df_countries["Number of Teams"], color="skyblue")
    ax.set_title(f"Top 20 Countries by Number of Teams in {selected_year}")
    ax.set_xlabel("Country")
    ax.set_ylabel("Number of Teams")
    plt.xticks(rotation=90)
    ax.grid(True, alpha=0.3, axis="y")
    st.pyplot(fig)

    st.markdown("---")

    # Row 3: Top bilateral relationships for selected year
    st.subheader(f"Top Bilateral Relationships ({selected_year})")

    bilateral_data = data_by_year[selected_year]["geographical"][
        "top_country_relationships"
    ]
    # Convert to more friendly format
    df_bilateral = pd.DataFrame(
        [
            {"Source": item[0][0], "Target": item[0][1], "Transfers": item[1]}
            for item in bilateral_data
        ]
    )

    # Display as table
    st.dataframe(df_bilateral, use_container_width=True)

    # Visualize top 10 relationships as a horizontal bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    relationship_labels = [
        f"{row['Source']} → {row['Target']}" for _, row in df_bilateral.iterrows()
    ]
    ax.barh(relationship_labels[::-1], df_bilateral["Transfers"][::-1], color="orange")
    ax.set_title(f"Top Bilateral Transfer Relationships in {selected_year}")
    ax.set_xlabel("Number of Transfers")
    ax.grid(True, alpha=0.3, axis="x")
    st.pyplot(fig)

# Tab 4: Community & Power
with tab4:
    st.header("⚖️ Centrality & Community Structure")

    # Row 1: Community metrics over time
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            trend_data["community"]["Year"],
            trend_data["community"]["Number of Communities"],
            marker="o",
            color="purple",
        )
        ax.set_title("Number of Communities Over Time")
        ax.set_xlabel("Year")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            trend_data["community"]["Year"],
            trend_data["community"]["Modularity Score"],
            marker="o",
            color="green",
        )
        ax.set_title("Modularity Score Over Time")
        ax.set_xlabel("Year")
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    st.markdown("---")

    # Row 2: Reciprocity over time
    st.subheader("Reciprocity Over Time")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        trend_data["community"]["Year"],
        trend_data["community"]["Reciprocity"],
        marker="o",
        color="blue",
    )
    ax.set_title("Network Reciprocity Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Reciprocity")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.markdown("---")

    # Row 3: Centrality for selected year
    st.subheader(f"Centrality Metrics ({selected_year})")

    tab_between, tab_eigen, tab_pagerank = st.tabs(
        ["Betweenness Centrality", "Eigenvector Centrality", "PageRank"]
    )

    with tab_between:
        betweenness_data = data_by_year[selected_year]["centrality"]["top_betweenness"]
        df_between = pd.DataFrame(betweenness_data, columns=["Club", "Betweenness"])

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(df_between["Club"][::-1], df_between["Betweenness"][::-1], color="teal")
        ax.set_title(f"Top Clubs by Betweenness Centrality in {selected_year}")
        ax.set_xlabel("Betweenness Centrality Score")
        ax.grid(True, alpha=0.3, axis="x")
        st.pyplot(fig)

        st.markdown(
            "Teams with high betweenness centrality act as important intermediaries in the transfer market, connecting different parts of the network."
        )

    with tab_eigen:
        eigenvector_data = data_by_year[selected_year]["centrality"]["top_eigenvector"]
        df_eigen = pd.DataFrame(eigenvector_data, columns=["Club", "Eigenvector"])

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(df_eigen["Club"][::-1], df_eigen["Eigenvector"][::-1], color="purple")
        ax.set_title(f"Top Clubs by Eigenvector Centrality in {selected_year}")
        ax.set_xlabel("Eigenvector Centrality Score")
        ax.grid(True, alpha=0.3, axis="x")
        st.pyplot(fig)

        st.markdown(
            "Teams with high eigenvector centrality are connected to other highly central clubs, indicating influence in the transfer network."
        )

    with tab_pagerank:
        pagerank_data = data_by_year[selected_year]["centrality"]["top_pagerank"]
        df_pagerank = pd.DataFrame(pagerank_data, columns=["Club", "PageRank"])

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(
            df_pagerank["Club"][::-1], df_pagerank["PageRank"][::-1], color="orange"
        )
        ax.set_title(f"Top Clubs by PageRank in {selected_year}")
        ax.set_xlabel("PageRank Score")
        ax.grid(True, alpha=0.3, axis="x")
        st.pyplot(fig)

        st.markdown(
            "PageRank measures a club's importance in the transfer network based on the quantity and quality of incoming connections."
        )

# Footer
st.markdown("---")
st.markdown("### Key Insights")

# Create two columns for insights
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Market Volume and Financial Trends")
    st.markdown("""
    - The total transfer volume shows significant changes over the analyzed period.
    - Average transfer fees have generally increased, indicating market inflation.
    - Top spending clubs have remained relatively consistent, with Premier League teams often leading.
    """)

    st.markdown("#### Network Dynamics")
    st.markdown("""
    - The transfer network has grown in terms of both nodes (teams) and edges (transfers).
    - Network density fluctuates, suggesting periods of consolidation and expansion.
    - The number of strongly connected components indicates the level of market fragmentation.
    """)

with col2:
    st.markdown("#### Geographical Patterns")
    st.markdown("""
    - There's a noticeable trend in the balance between domestic and international transfers.
    - Certain countries maintain dominant positions in terms of team count.
    - Bilateral relationships reveal consistent transfer corridors between specific countries.
    """)

    st.markdown("#### Community Structure and Power")
    st.markdown("""
    - Centrality metrics identify the most influential clubs in the transfer network.
    - Teams with high betweenness centrality often act as market intermediaries.
    - Reciprocity levels indicate the tendency for teams to form mutual transfer relationships.
    """)

st.sidebar.markdown("---")
st.sidebar.info("""
**About this dashboard:**
This dashboard analyzes football transfer market data from 2009 to 2021, 
visualizing network metrics, financial trends, geographical patterns, and 
market power dynamics.
""")
