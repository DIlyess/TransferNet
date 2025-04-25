import glob
import json
import os
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Football Transfer Market Analysis",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Define utility functions
def load_json_data(file_path):
    """Load and return JSON data from a file"""
    with open(file_path, "r") as file:
        return json.load(file)


def load_all_years_data(data_directory):
    """Load data for all available years from the given directory"""
    # Get all JSON files in the directory
    json_files = glob.glob(os.path.join(data_directory, "*.json"))
    data = {}

    for file_path in json_files:
        try:
            year = int(Path(file_path).stem)  # Extract year from filename
            data[year] = load_json_data(file_path)
        except ValueError:
            # Skip files that don't have a year as filename
            continue

    return data


# Define visualization components
class MarketVolumeVisualizer:
    """Component for visualizing market volume and financial trends"""

    @staticmethod
    def plot_yearly_transfer_volume(yearly_data):
        """Plot total transfer volume per year"""
        years = sorted(yearly_data.keys())
        volumes = [
            yearly_data[year]["financial"]["total_transfer_volume"] for year in years
        ]

        fig = px.line(
            x=years,
            y=volumes,
            labels={"x": "Year", "y": "Total Transfer Volume (€)"},
            title="Yearly Transfer Volume",
        )
        fig.update_layout(xaxis=dict(tickmode="linear", dtick=1))
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_avg_transfer_fee_trend(yearly_data):
        """Plot average transfer fee trends over years"""
        years = sorted(yearly_data.keys())
        avg_fees = [
            yearly_data[year]["financial"]["avg_transfer_fee"] for year in years
        ]

        fig = px.line(
            x=years,
            y=avg_fees,
            labels={"x": "Year", "y": "Average Transfer Fee (€)"},
            title="Average Transfer Fee Trends",
        )
        fig.update_layout(xaxis=dict(tickmode="linear", dtick=1))
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_top_spenders(yearly_data, selected_year, top_n=10):
        """Plot top spenders for a selected year"""
        if selected_year not in yearly_data:
            st.error(f"Data for year {selected_year} not found.")
            return

        top_spenders = yearly_data[selected_year]["financial"]["top_spenders"]
        teams = [item[0] for item in top_spenders[:top_n]]
        amounts = [item[1] for item in top_spenders[:top_n]]

        fig = px.bar(
            x=teams,
            y=amounts,
            labels={"x": "Team", "y": "Amount Spent (€)"},
            title=f"Top {top_n} Spenders in {selected_year}",
        )
        fig.update_layout(xaxis_title="Team", yaxis_title="Amount Spent (€)")
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_top_income_generators(yearly_data, selected_year, top_n=10):
        """Plot top income generators for a selected year"""
        if selected_year not in yearly_data:
            st.error(f"Data for year {selected_year} not found.")
            return

        top_income = yearly_data[selected_year]["financial"]["top_income_generators"]
        teams = [item[0] for item in top_income[:top_n]]
        # Convert negative values to positive for better visualization
        amounts = [-item[1] for item in top_income[:top_n]]

        fig = px.bar(
            x=teams,
            y=amounts,
            labels={"x": "Team", "y": "Income Generated (€)"},
            title=f"Top {top_n} Income Generators in {selected_year}",
        )
        fig.update_layout(xaxis_title="Team", yaxis_title="Income Generated (€)")
        st.plotly_chart(fig, use_container_width=True)


class NetworkDynamicsVisualizer:
    """Component for visualizing transfer network dynamics"""

    @staticmethod
    def plot_network_density_over_time(yearly_data):
        """Plot network density over time"""
        years = sorted(yearly_data.keys())
        densities = [yearly_data[year]["basic"]["density"] for year in years]

        fig = px.line(
            x=years,
            y=densities,
            labels={"x": "Year", "y": "Network Density"},
            title="Transfer Network Density Over Time",
        )
        fig.update_layout(xaxis=dict(tickmode="linear", dtick=1))
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_degree_distribution(yearly_data, selected_year):
        """Plot in-degree vs out-degree distribution for a selected year"""
        if selected_year not in yearly_data:
            st.error(f"Data for year {selected_year} not found.")
            return

        degree_data = yearly_data[selected_year]["degree"]

        # Create a figure with subplots
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("In-Degree Distribution", "Out-Degree Distribution"),
        )

        # Add in-degree data
        fig.add_trace(
            go.Bar(
                x=["Max", "Min", "Avg", "Median"],
                y=[
                    degree_data["max_in_degree"],
                    degree_data["min_in_degree"],
                    degree_data["avg_in_degree"],
                    degree_data["median_in_degree"],
                ],
                name="In-Degree",
            ),
            row=1,
            col=1,
        )

        # Add out-degree data
        fig.add_trace(
            go.Bar(
                x=["Max", "Min", "Avg", "Median"],
                y=[
                    degree_data["max_out_degree"],
                    degree_data["min_out_degree"],
                    degree_data["avg_out_degree"],
                    degree_data["median_out_degree"],
                ],
                name="Out-Degree",
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            title_text=f"Degree Distribution in {selected_year}", height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_top_buyers_sellers(yearly_data, selected_year, top_n=10):
        """Plot top buyers vs sellers for a selected year"""
        if selected_year not in yearly_data:
            st.error(f"Data for year {selected_year} not found.")
            return

        top_buyers = yearly_data[selected_year]["top_teams"]["top_buyers"][:top_n]
        top_sellers = yearly_data[selected_year]["top_teams"]["top_sellers"][:top_n]

        buyer_teams = [item[0] for item in top_buyers]
        buyer_counts = [item[1] for item in top_buyers]

        seller_teams = [item[0] for item in top_sellers]
        seller_counts = [item[1] for item in top_sellers]

        # Create a figure with subplots
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(f"Top {top_n} Buyers", f"Top {top_n} Sellers"),
        )

        # Add buyers data
        fig.add_trace(
            go.Bar(x=buyer_teams, y=buyer_counts, name="Buyers"), row=1, col=1
        )

        # Add sellers data
        fig.add_trace(
            go.Bar(x=seller_teams, y=seller_counts, name="Sellers"), row=1, col=2
        )

        fig.update_layout(
            title_text=f"Top Buyers vs Sellers in {selected_year}", height=600
        )

        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_connected_components(yearly_data):
        """Plot the number and size of strongly and weakly connected components over time"""
        years = sorted(yearly_data.keys())

        # Extract the number of strongly and weakly connected components
        num_scc = [
            yearly_data[year]["basic"]["number_strongly_connected_components"]
            for year in years
        ]
        num_wcc = [
            yearly_data[year]["basic"]["number_weakly_connected_components"]
            for year in years
        ]

        # Extract the size of the largest strongly and weakly connected components
        largest_scc_size = [
            yearly_data[year]["connectivity"]["largest_scc_size"] for year in years
        ]
        largest_wcc_size = [
            yearly_data[year]["connectivity"]["largest_wcc_size"] for year in years
        ]

        # Create a figure with subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Number of Strongly Connected Components",
                "Number of Weakly Connected Components",
                "Size of Largest Strongly Connected Component",
                "Size of Largest Weakly Connected Component",
            ),
        )

        # Add traces for number of components
        fig.add_trace(
            go.Scatter(x=years, y=num_scc, mode="lines+markers", name="SCC"),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(x=years, y=num_wcc, mode="lines+markers", name="WCC"),
            row=1,
            col=2,
        )

        # Add traces for largest component sizes
        fig.add_trace(
            go.Scatter(
                x=years, y=largest_scc_size, mode="lines+markers", name="Largest SCC"
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=years, y=largest_wcc_size, mode="lines+markers", name="Largest WCC"
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title_text="Connected Components Analysis Over Time",
            height=800,
            xaxis=dict(tickmode="linear", dtick=1),
            xaxis2=dict(tickmode="linear", dtick=1),
            xaxis3=dict(tickmode="linear", dtick=1),
            xaxis4=dict(tickmode="linear", dtick=1),
        )

        st.plotly_chart(fig, use_container_width=True)


# Main application function
def main():
    st.title("⚽ Football Transfer Market Analysis Dashboard")
    st.markdown("""
    This dashboard provides insights into football transfer market trends from 2009 to 2021.
    Use the sidebar to navigate through different analyses.
    """)

    # Sidebar for navigation
    st.sidebar.title("Navigation")

    # Data directory input
    data_dir = st.sidebar.text_input(
        "Enter the directory path containing JSON files:",
        value="../market_analysis",  # Default directory
    )

    # Try to load data
    try:
        yearly_data = load_all_years_data(data_dir)
        if not yearly_data:
            st.error(f"No valid JSON files found in {data_dir}")
            st.stop()

        available_years = sorted(yearly_data.keys())
        selected_year = st.sidebar.selectbox(
            "Select Year for Detailed Analysis",
            available_years,
            index=len(available_years) - 1,
        )

        analysis_type = st.sidebar.radio(
            "Select Analysis Type",
            ["Market Volume & Financial Trends", "Transfer Network Dynamics"],
        )

        # Display analysis based on selection
        if analysis_type == "Market Volume & Financial Trends":
            st.header("📊 Market Volume & Financial Trends")

            col1, col2 = st.columns(2)

            with col1:
                MarketVolumeVisualizer.plot_yearly_transfer_volume(yearly_data)
            with col2:
                MarketVolumeVisualizer.plot_avg_transfer_fee_trend(yearly_data)

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                MarketVolumeVisualizer.plot_top_spenders(yearly_data, selected_year)
            with col2:
                MarketVolumeVisualizer.plot_top_income_generators(
                    yearly_data, selected_year
                )

        else:  # Transfer Network Dynamics
            st.header("🔄 Transfer Network Dynamics")

            NetworkDynamicsVisualizer.plot_network_density_over_time(yearly_data)

            st.markdown("---")

            NetworkDynamicsVisualizer.plot_degree_distribution(
                yearly_data, selected_year
            )

            st.markdown("---")

            NetworkDynamicsVisualizer.plot_top_buyers_sellers(
                yearly_data, selected_year
            )

            st.markdown("---")

            NetworkDynamicsVisualizer.plot_connected_components(yearly_data)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        if st.checkbox("Show traceback"):
            import traceback

            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
