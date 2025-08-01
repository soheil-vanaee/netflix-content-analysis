import pandas as pd
from collections import Counter

def load_and_clean_data(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df.drop_duplicates(inplace=True)
    df.dropna(how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    if 'Release_Date' in df.columns:
        df['Release_Date'] = pd.to_datetime(df['Release_Date'], format='%B %d, %Y', errors='coerce')
        df = df.dropna(subset=['Release_Date'])
        df['Year'] = df['Release_Date'].dt.year
    if 'Category' in df.columns:
        df['Category'] = df['Category'].fillna('Unknown')
    if 'Description' in df.columns:
        df['Description'] = df['Description'].fillna('')
    if 'Duration' in df.columns:
        df['Duration'] = df['Duration'].fillna('Unknown')
    if 'Rating' in df.columns:
        df['Rating'] = df['Rating'].fillna('Not Rated')
    if 'Type' in df.columns:
        df['Type'] = df['Type'].fillna('Unknown')
    df.to_csv(output_path, index=False)
    return df



def analyze_split_genres(df: pd.DataFrame, genre_column: str) -> pd.DataFrame:
    genre_lists = df[genre_column].dropna().apply(lambda x: [g.strip() for g in x.split(',')])
    all_genres = [genre for sublist in genre_lists for genre in sublist]
    counts = Counter(all_genres)
    total = sum(counts.values())
    data = {'Count': dict(counts), 'Percent': {k: round(v / total * 100, 2) for k, v in counts.items()}}
    result = pd.DataFrame(data)
    return result.sort_values(by='Count', ascending=False)


def analyze_production_by_country(df: pd.DataFrame, country_col: str, genre_col: str) -> pd.DataFrame:
    df = df.dropna(subset=[country_col, genre_col])
    country_counts = df[country_col].value_counts()
    country_genres = {}
    for country, group in df.groupby(country_col):
        genres_series = group[genre_col].dropna().apply(lambda x: [g.strip() for g in x.split(',')])
        all_genres = [genre for sublist in genres_series for genre in sublist]
        counts = Counter(all_genres)
        top_3 = counts.most_common(3)
        country_genres[country] = [genre for genre, count in top_3]
    result = pd.DataFrame({
        'Total_Productions': country_counts,
        'Top_3_Genres': pd.Series(country_genres)
    })
    result = result.sort_values(by='Total_Productions', ascending=False)
    return result


def analyze_directors_actors(df: pd.DataFrame, director_col: str, cast_col: str, genre_col: str):
    df = df.dropna(subset=[director_col, cast_col, genre_col])
    directors = df[director_col].dropna().apply(lambda x: [d.strip() for d in x.split(',')])
    all_directors = [d for sublist in directors for d in sublist]
    director_counts = Counter(all_directors)

    casts = df[cast_col].dropna().apply(lambda x: [c.strip() for c in x.split(',')])
    all_casts = [c for sublist in casts for c in sublist]
    cast_counts = Counter(all_casts)

    director_genres = {}
    for director in director_counts.keys():
        director_rows = df[df[director_col].str.contains(director, na=False)]
        genres_series = director_rows[genre_col].dropna().apply(lambda x: [g.strip() for g in x.split(',')])
        all_genres = [genre for sublist in genres_series for genre in sublist]
        counts = Counter(all_genres)
        top_genres = counts.most_common()
        director_genres[director] = top_genres

    directors_df = pd.DataFrame(director_counts.most_common(), columns=['Director', 'Total_Productions'])
    casts_df = pd.DataFrame(cast_counts.most_common(), columns=['Actor', 'Total_Productions'])
    return directors_df, casts_df, director_genres

import pandas as pd

def genre_trend_over_time(df: pd.DataFrame, date_col: str, genre_col: str):
    df = df.dropna(subset=[date_col, genre_col])
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df['Year'] = df[date_col].dt.year
    df = df.dropna(subset=['Year'])
    df[genre_col] = df[genre_col].fillna('')
    df_expanded = df[[genre_col, 'Year']].copy()
    df_expanded[genre_col] = df_expanded[genre_col].apply(lambda x: [g.strip() for g in x.split(',')])
    df_expanded = df_expanded.explode(genre_col)
    trend = df_expanded.groupby(['Year', genre_col]).size().unstack(fill_value=0)
    return trend.sort_index()
