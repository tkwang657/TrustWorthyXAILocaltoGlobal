def process_numeric_with_exempt(df, colname, create_status_column=True):
    """Preprocess a column that contains numeric values as strings, 'Exempt', or missing values.

    Parameters:
    -----------
    create_status_column : bool
        Whether to create a categorical status column ('Reported' / 'Exempt'/'Missing').

    Returns:
    --------
    pd.DataFrame
        DataFrame with:
        - numeric column (original name) with NaNs for exempt/missing,
        - optional categorical status column: <colname>_status
    """

    numeric_col = pd.to_numeric(df[colname], errors='coerce')
    
    # Identify 'Exempt' entries
    exempt_mask = df[colname].lower() == 'exempt'
    numeric_col[exempt_mask] = np.nan
    if create_status_column:
        status_col = f"{colname}_status"
        status_col_data = pd.Series(np.where(
            exempt_mask, 'Exempt',
            np.where(numeric_col.notna(), 'Reported', 'Missing')
        ), index=df.index)
        df[status_col] = status_col_data
    df[colname] = numeric_col
    return df
def yes_no_to_bool(df, colname):
    df[colname]=df[colname].astype(str).lower().map({'yes': 1, "no": 0}).where(df[colname].notna())
    return df
