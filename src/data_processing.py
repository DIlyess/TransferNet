import pandas as pd


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
