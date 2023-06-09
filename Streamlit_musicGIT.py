import streamlit as st
import pandas as pd 
import streamlit.components.v1 as components

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

st.sidebar.image("new_logo_datascientest.png", width=100)
# st.sidebar.title("Recommandation system")
pages = ["Recommandation system"]
page = st.sidebar.radio("Pick a page",options = pages)

df = pd.read_csv("final.csv")
dataset = df    

if page == pages[0]: 
    st.header("Recommandation system")
    st.markdown('##')
    st.markdown("##### Type in name of artist and track : ")    
    option = st.selectbox("", df['artist_track'].unique(), 
                          index = 25369,
                          label_visibility = "collapsed")
    
    def reco(track):
        # Create a new dataframe with all info on selected track:
        df_track = dataset.loc[(dataset['artist_track'] == track)][['artists', 'track_name', 'artist_track', 'track_genre', 'cluster_genres', 'tempo', 'energy', 'loudness', 'popularity']]
        artist = df_track['artists']
  
        # Create a new dataframe with tracks that have similar caracteristics:
        track_infos = pd.DataFrame()
        for i, j, k, l, m, n, o in zip(dataset['artist_track'], dataset['cluster_attributs'], dataset['tempo'], dataset['loudness'], dataset['energy'], dataset['acousticness'], dataset['danceability']):
            if i == track:
                track_infos = dataset.loc[(dataset['cluster_attributs'] == j) 
                & ((dataset['tempo'] > (k - 5)) & (dataset['tempo'] < (k + 5))) 
                | ((dataset['tempo'] > ((k/2) - 5)) & (dataset['tempo'] < ((k/2) + 5)))
                | ((dataset['tempo'] > ((k*2) - 5)) & (dataset['tempo'] < ((k*2) + 5)))
                & (dataset['loudness'] > (l - 1)) & (dataset['loudness'] < (l + 1)) 
                & (dataset['energy'] > (m - 0.2)) & (dataset['energy'] < (m + 0.2))
                & (dataset['acousticness'] > (n - 0.3)) & (dataset['acousticness'] < (n + 0.3))
                & (dataset['danceability'] > (o - 0.3)) & (dataset['danceability'] < (o + 0.3))]
                [['artists', 'track_name', 'artist_track', 'track_genre', 'cluster_genres', 'tempo', 'energy', 'loudness', 'popularity']]
  
        track_infos = track_infos.drop_duplicates(subset = ['artist_track'])
        track_infos = track_infos[track_infos['artist_track'] != track]
        track_infos = track_infos[~track_infos['artist_track'].str.contains(track[:20])]
        track_infos = track_infos.sort_values('popularity', ascending = False)
  
        # Create a list and a dataframe with tracks that have same genres:
        list_genres = list(df_track['track_genre'])
        df_genre = dataset[dataset['track_genre'].isin(list_genres)]

        # Filter tracks with sames genres based on similar caracteristics:
        for i, j, k, l, m, n, o in zip(df_genre['artist_track'], df_genre['cluster_genres'], df_genre['tempo'], df_genre['loudness'], df_genre['energy'], dataset['acousticness'], dataset['danceability']):
            if i == track:
                df_genre = df_genre.loc[(df_genre['cluster_genres'] == j)
                & (df_genre['tempo'] > (k - 7)) & (df_genre['tempo'] < (k + 7))
                | ((df_genre['tempo'] > ((k/2) - 7)) & (df_genre['tempo'] < ((k/2) + 7)))
                | ((df_genre['tempo'] > ((k*2) - 7)) & (df_genre['tempo'] < ((k*2) + 7)))
                & (df_genre['loudness'] > (l - 2.5)) & (df_genre['loudness'] < (l + 2.5)) 
                & (df_genre['energy'] > (m - 0.2)) & (df_genre['energy'] < (m + 0.2))
                & (dataset['acousticness'] > (n - 0.2)) & (dataset['acousticness'] < (n + 0.2))
                & (dataset['danceability'] > (o - 0.2)) & (dataset['danceability'] < (o + 0.2))]
                [['artists', 'track_name', 'artist_track', 'track_genre', 'cluster_genres', 'tempo', 'energy', 'loudness', 'popularity']]
  
        df_genre = df_genre[df_genre['artist_track'] != track]
        df_genre = df_genre.drop_duplicates(subset = ['artist_track'])
        df_genre = df_genre[~df_genre['artist_track'].str.contains(track[:20])]
        df_genre = df_genre.sort_values('popularity', ascending = False)
 
        # Create a new dataframe with all tracks from same artist or band:
        df_artist = dataset.loc[(dataset['artists'].isin(artist)) 
        | (dataset['artists'].str.contains(track[:9]))]
        [['artists', 'track_name', 'artist_track', 'track_genre', 'cluster_genres', 'tempo', 'energy', 'loudness', 'popularity']]   
  
        df_artist = df_artist.drop_duplicates(subset = ['artist_track'])
        df_artist = df_artist[df_artist['artist_track'] != track] 
        df_artist = df_artist[~df_artist['artist_track'].str.contains(track[:20])]
        df_artist = df_artist.sort_values('popularity', ascending = False)

        # Filter all created dataframes to prioritize result:
        if not df_artist.empty:
            return print(df_artist['artist_track'].values[:1])
        elif not df_genre.empty:  
            return print(df_genre['artist_track'].values[:1])
        else:
            return print(track_infos['artist_track'].values[:1])

    OG_track = "Taylor Swift - Blank Space"
    if option != OG_track:
        st.write("You picked the track :", option)
        track = str(option)
        reco1 = reco(track)
        st.write(track)

    