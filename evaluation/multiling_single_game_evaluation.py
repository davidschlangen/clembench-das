import os
import sys

sys.path.append('..')
import glob
from pathlib import Path
import pandas as pd


def save_overview_table(df: pd.DataFrame, categories: list, path: str) -> None:
    """Create multilingual results as a table."""
    scored_df = df[(df.game == 'referencegame') & (df["metric"].isin(["Played", "Success", "Aborted at Player 1"]))]

    # compute mean metrics
    df_means = (scored_df.groupby(categories)
                .mean(numeric_only=True)
                .reset_index())
    # convert to percentages
    aux_ab_p1 = df_means.loc[df_means.metric == "Aborted at Player 1", 'value']
    aux_played = df_means.loc[df_means.metric == "Played", 'value']
    aux_aborted = (1-aux_played).to_list()
    df_means.loc[df_means.metric == "Aborted at Player 1", 'value'] = (aux_ab_p1/aux_aborted) * 100
    df_means.loc[df_means.metric == "Played", 'value'] *= 100
    df_means.loc[df_means.metric == "Success", 'value'] *= 100

    df_means = df_means.round(2)
    df_means['metric'].replace(
        {"Played": '% Played', "Success": '% Success (of Played)', "Aborted at Player 1": 'Aborted at Player 1 (of Aborted)'},
        inplace=True)

    # make metrics separate columns
    df_means = df_means.pivot(columns=categories[-1], index=categories[:-1])
    df_means = df_means.droplevel(0, axis=1)

    # compute clemscores and add to df
    clemscore = (df_means['% Played'] / 100) * df_means['% Success (of Played)']
    clemscore = clemscore.round(2).to_frame(name=('clemscore (Played * Success)'))
    df_results = pd.concat([clemscore, df_means], axis=1)

    # sort models within language by clemscore
    reset = df_results.reset_index()
    sorted = reset.sort_values(['lang','clemscore (Played * Success)'],ascending=[True,False])

    # TODO: figure out how to display this nicely (use jupyter notebook?)
    # save table
    sorted.to_csv(Path(path) / 'results_multiling.csv')
    sorted.to_html(Path(path) / 'results_multiling.html', index=False)
    print(f'\n Saved results into {path}/results_multiling.html and .csv')


if __name__ == '__main__':
    # TODO: make options command line arguments
    # collect all language specific results in one dataframe
    path = '../results/v1.5_multiling'
    df = None
    result_dir = Path(path)
    lang_dirs = glob.glob(f"{result_dir}/*/") # the trailing / ensures that only directories are found
    for lang_dir in lang_dirs:
        lang = lang_dir.split("/")[1]
        raw_file = os.path.join(lang_dir, 'raw.csv')
        assert Path(raw_file).is_file()
        lang_result = pd.read_csv(raw_file, index_col=0)
        lang_result.insert(0, 'lang', lang)
        df = pd.concat([df, lang_result], ignore_index=True)

    #categories = ['lang', 'model', 'experiment', 'metric'] #detailed by experiment
    categories = ['lang', 'model', 'metric']
    stats_df = save_overview_table(df, categories, path)