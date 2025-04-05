import pandas as pd


class DataProcessing:

    def __init__(self, df):
        """
        Initialize the DataProcessing class with a DataFrame.
        Args:
            df (pd.DataFrame): DataFrame to be processed.
        """
        self.df = df

    def process_data(self):
        """
        Process the DataFrame by removing duplicate transfers and converting 'in-transfers' to 'out-transfers'.
        Returns:
            pd.DataFrame: Processed DataFrame.
        """
        self.df = self.remove_duplicate_transfer(self.df)
        self.df = self.in2out_transfer(self.df)
        self.df = self.aggregate_club_level(self.df)
        return self.df

    @staticmethod
    def aggregate_club_level(df):
        grouping_cols = [
            "season",
            "team_id",
            "counter_team_id",
            "team_name",
            "counter_team_name",
            "team_country",
            "counter_team_country",
        ]
        agg_dict = {
            "transfer_fee_amnt": "sum",
            "is_loan": "mean",
        }
        rename_dict = {
            "transfer_fee_amnt": "total_fee",
        }
        df_club = df.groupby(grouping_cols).agg(agg_dict).reset_index()
        df_club = df_club.rename(columns=rename_dict)

        return df_club

    @staticmethod
    def remove_duplicate_transfer(df):
        """
        Remove duplicate transfers from the DataFrame by droping only the 'in-transfers'.
        Args:
            df (pd.DataFrame): DataFrame containing transfer data.
        Returns:
            pd.DataFrame: DataFrame with duplicate transfers removed and 'in-transfers' turned into 'out-transfers'.
        """
        # Drop the duplicated 'in-transfers'
        transfer_id_count = df.groupby(
            "transfer_id"
        ).size()  # Count the number of occurrences of each transfer_id
        to_drop = transfer_id_count[transfer_id_count > 1].index
        df_in = df[df["dir"] == "in"]
        to_drop = df_in[df_in["transfer_id"].isin(to_drop)].index
        df = df.drop(to_drop)

        return df

    @staticmethod
    def in2out_transfer(df):
        """
        Convert 'in-transfers' to 'out-transfers' in the DataFrame.
        Args:
            df (pd.DataFrame): DataFrame containing transfer data.
        Returns:
            pd.DataFrame: DataFrame with 'in-transfers' converted to 'out-transfers'.
        """
        # Change the 'in-transfers' to 'out-transfers'
        df_out = df[df["dir"] == "left"]
        df_in = df[df["dir"] == "in"]
        swap_map = {
            "team_id": "counter_team_id",
            "team_name": "counter_team_name",
            "team_country": "counter_team_country",
        }
        swap_map.update({v: k for k, v in swap_map.items()})
        df_in = df_in.rename(columns=swap_map)
        df_in = df_in[df_out.columns]  # Keep the same col order
        df = pd.concat([df_out, df_in], ignore_index=True).drop(columns=["dir"])

        return df
