import streamlit as st
import pandas as pd 
import numpy as np
from PIL import Image, ImageOps
import streamlit.components.v1 as components
import requests
import json
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()
from scipy.spatial.distance import pdist, squareform
import graphviz 
import plotly.express as px

# Masquage du footer 
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

# Gestion des images et des datasets
image = Image.open('target.png')
flimage = ImageOps.mirror(image)
JLR = Image.open("JLR_circle.png")
fliJLR = JLR.rotate(270)
dataset = pd.read_csv("final_red.csv", index_col = 0)
dfm = pd.read_csv("df_matrix.csv", index_col = 'artist_track')
dfm2 = pd.read_csv("df_matrix.csv")
dataset_sample = pd.read_csv("dataset.csv", index_col = 0)
pop_genre = pd.read_csv("pop_genre.csv", index_col = 0)
numgenres = pd.read_csv('numgenres.csv', index_col = 0)
top10artists = pd.read_csv('top10artists.csv', index_col = 0)
top15artists = pd.read_csv('top15artists.csv', index_col = 0)
top15titres = pd.read_csv('top15titres.csv', index_col = 0)
top15genres = pd.read_csv('top15genres.csv', index_col = 0)

# Menu latéral 
st.sidebar.image("logo_mines.png", output_format = "PNG", width = 200)
st.sidebar.header("[NAME OF CERTIFICATE] - Research project")
st.sidebar.header("Menu")
pages = ["Home Page", "Introduction", "Datasets", "Data Visualizations", "Data Modeling", "Conclusion", "Recommendation System"]
page = st.sidebar.radio("Select a page", options = pages)
st.sidebar.header("Author")
st.sidebar.markdown("""
Julie Le Rudulier - [LinkedIn](https://www.linkedin.com/in/julielerudulier/)  
""") 
st.sidebar.image(fliJLR, width = 100, output_format = "PNG")

# Page d'accueil 
if page == pages[0]: 
    st.markdown("### Music Recommendations: Recommending Songs Through Two Different Machine Learning Algorithms")
    st.markdown("##")
    col1, col2, col3 = st.columns([1,5,1])
    with col1:
        st.write("")
    with col2:
        st.markdown("![Alt Text](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbmR5MWJrZjZsbTh3aHNtNDdxMzJubHdnMGMzNTZ2ZHUyNjF4bXI1NiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/tUxON6fVGX2LD87p57/giphy.gif)")
    with col3:
        st.write("")

    st.markdown("##")
    col11, col22, col33 = st.columns([1,5,1])
    with col11:
        st.write("")
    with col22:
        st.write("Research project conducted as part of the [NOM DE LA FORMATION], June 2023.")
    with col33:
        st.write("")

# Page 1 - Introduction
if page == pages[1]: 
    st.header("Introduction")
    tab1, tab2, tab3 = st.tabs(['Project Context', 'Approach', 'Research Objectives'])
    with tab1:
        st.markdown("##### Project Context")
        st.write("The aim of this project is to recommend songs accurately to users, based on data retrieved from Spotify and Twitter.")
        st.write("This project started in January 2023. A paper on data cleansing and exploration was handed in to Faculty members in March. This paper also included data visualizations and a primary analysis of trends in data. A second paper on data modeling was handed in in May.")
        st.write("The final essay was handed in to Faculty members in June, while the outcomes of this project as well as the recommendation system were presented to a jury on June 26th, 2023.")
    with tab2:    
        st.markdown("##### Approach")
        st.write("There are mainly four recommendation methods that are commonly used in music recommender systems: \n- Content-based filtering; \n- Context-based filtering; \n- Collaborative filtering; \n- And hybrid methods, which combine the other filtering methods and minimize the issues a single method can have.")
        st.write("The former method was preferred to conduct this project. Indeed, most streaming platforms generally use collaborative filtering systems, sometimes combined with other algorithms. It thus seemed more interesting to design a system that is not based on ratings and reactions by similar users, but rather that is based on the similarity of the songs' technical features.")
        st.write("Also, recommender systems do not always provide the most accurate recommendations. To help improve users' satisfaction we chose to build our system based on two different algorithms : \n - A clustering algorithm; \n - A similarity matrix.")
        st.write("The two algorithms should return songs based on similar attributes that they share with a seed track, through two different methods and distance metrics. Both systems will operate simultaneously and each will recommend one song so that users can pick the song they like the most.")
    with tab3:   
        st.markdown("##### Research Objectives") 
        st.write("In the music domain, content-based filtering ranks songs based on how similar they are to a seed song according to some similarity measure, which focuses on an objective distance between items and does not include any subjective factors. This makes it possible to recommend new items that do not have any user ratings associated with them.")    
        st.write("As a result one of the initial objectives of this project will be to determine which attributes are essential to provide effective similarity between tracks, through the degree of linearity or correlation of the songs' features for instance, or through automatic feature selection.")
        st.write("Once key features will be identified, the main objective will be to design a system with very specific instructions so that it provides users with the most accurate recommendations possible, considering the number of songs contained in the datasets of this project is rather small.") 
        st.write("It should be noted that allowing users to rate the recommendations and save their ratings in a database was not in the scope of this project.")
            
# Page 2 - Datasets
if page == pages[2]: 
    st.header("Datasets")
    st.write("Five datasets were made available to carry out this project with data such as: \n- Information and attributes of songs played on Spotify; \n- Hashtags published on Twitter by users who were listening to music; \n- The hashtags' aggregate sentiment values, assessed through multiple sentiment dictionaries.") 
    st.write("After several unsuccessful attempts at creating relevant algorithms with the multiple datasets available, we established that our key variables were all contained in one dataset and we decided to focus our work on this one dataset only: the dataset #1")
    st.markdown("###")

    tabDS1, tabDS2, tabDS3, tabDS4, tabDS5 = st.tabs(['Dataset #1', 'Dataset #2', 'Dataset #3', 'Dataset #4', 'Dataset #5'])
    with tabDS1:   
        st.markdown("##### Dataset #1: 'Dataset'")
        st.write("The first dataset is titled 'Dataset'. It is a dataset of Spotify tracks over a range of 125 different music genres, associated with multiple audio features such as the tracks' tempo, their mode, valence, danceability, liveness... While we have no information regarding the release date of the songs in this dataset, it seems to contain tracks from past and present time.") 
        st.write("There are 20 original columns in this dataset - 14 numeric variables, 5 categorial variables and 1 Boolean variable - and 114 000 rows. Its size is 20.12Mo. This dataset is available on [Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset).")
        st.write("First 5 rows of the dataset:")
        dataset_5 = pd.read_csv("dataset_5.csv", index_col = 0)
        st.dataframe(dataset_5)

    with tabDS2:
        st.markdown("##### Dataset #2: 'User_Track_Hashtag_Timestamp'")
        st.write("The second dataset is titled 'User Track Hashtag Timestamp'. It contains basic information on 11.6 million music listening events of 139K users and 346K tracks collected from Twitter.") 
        st.write("There are 4 original columns and 17,560,113 rows in this dataset. Its size is 1.31Go. This dataset is available on [Kaggle](https://www.kaggle.com/datasets/chelseapower/nowplayingrs?select=user_track_hashtag_timestamp.csv).")
        st.write("First 5 rows of the dataset:")
        utht_5 = pd.read_csv("user_track_hashtag_timestamp_5.csv", index_col = 0)
        st.dataframe(utht_5)
        
    with tabDS3:
        st.markdown("##### Dataset #3: 'Sentiment_Values'")
        st.write("The third dataset is titled 'Sentiment Values'. It is linked to the 11.6 million music listening events listed in dataset #2 and contains hashtags and their associated sentiment values gathered via four different sentiment dictionaries: AFINN, Opinion Lexicon, Sentistrength Lexicon and Vader. For each of these dictionaries, the minimum, maximum, sum and average of all sentiments of the tokens of the hashtag were listed if available.") 
        st.write("There are 21 original columns and 5,290 rows in this dataset. Its size is 382Ko. This dataset is available on [Kaggle](https://www.kaggle.com/datasets/chelseapower/nowplayingrs?select=user_track_hashtag_timestamp.csv).")
        st.write("First 5 rows of the dataset:")
        sentiment_5 = pd.read_csv("sentiment_values_5.csv", index_col = 0)
        st.dataframe(sentiment_5)

    with tabDS4:
        st.markdown("##### Dataset #4: 'Context_Content_Features'")
        st.write("The fourth dataset is titled 'Context Content Features'. It contains all context and content features of the 11.6 million Twitter music listening events. For each event, content features regarding the track mentioned in the tweet were listed, such as instrumentalness, liveness, speechiness... Context features regarding the listening event such as the language of the tweet were also listed in this dataset.") 
        st.write("There are 22 original columns and 11,614,671 rows in this dataset. Its size is 2.21Go. This dataset is available on [Kaggle](https://www.kaggle.com/datasets/chelseapower/nowplayingrs?select=user_track_hashtag_timestamp.csv).")
        st.write("First 5 rows of the dataset:")
        ccf_5 = pd.read_csv("ccf_reduit_5.csv", index_col = 0)
        st.dataframe(ccf_5)

    with tabDS5:
        st.markdown("##### Dataset #5: 'Module4 Cleaned'")
        st.write("The fifth and last dataset is titled 'Module4 Cleaned'. It is a combination of the following datasets: 'Sentiment Values', 'User Track Hashtag Timestamp' and 'Context Content Features'. Data for each of these datasets were previously cleansed.") 
        st.write("During the data exploration phase of this project, we were not entirely sure how we were going to process data in order to design our recommendation system. For that reason we decided to let this dataset aside and do the data cleansing process ourselves. As a result, this dataset was not used at all in this project.")
        st.write("This dataset is available on [Kaggle](https://www.kaggle.com/code/chelseapower/module4-project/output).")
        st.write("First 5 rows of the dataset:")
        module4_5 = pd.read_csv("module4_cleaned_5.csv", index_col = 0)
        st.dataframe(module4_5)
       
# Page 3 - Data Visualizations
if page == pages[3]:
    st.header("Data Visualizations")
    st.write("Using visual elements like charts and graphs will provide us with an accessible way to see and understand trends, outliers, and patterns in our data. Furthermore, capturing the distribution of key variables will allow us to understand critical statistical properties of the data we will work with, and it will help us make educated data-driven decisions on key outcomes when designing our recommendation system.")
    tab_top, tab_correlations, tab_linearite = st.tabs(["Trends in Data", "Correlation", "Linearity"])
    with tab_top:
        tab1, tab2, tab3 = st.tabs(["Artists", "Songs", "Music Genres"])
        with tab1:
            tab111, tab222 = st.tabs(["Most Popular Artists In Dataset", "Artists With Most Songs In Dataset"])
            with tab111:
                st.markdown("#### Top 15 Most Popular Artists")
                st.write("The visualization below shows that the most popular artists come from all around the world: Spain, Argentina, Scotland, England, Chile, the United States... However it appears that these artists only have two songs on average in the dataset.")
                st.markdown("**The most popular artists seem to be underrepresented in our dataset.**")
                fig_top15artists = px.area(top15artists, 
                                           x = "Popularity (mean)", 
                                           y = "Artists", 
                                           color = "Genres", 
                                           line_group = "Number of songs in dataset")
                fig_top15artists.update_yaxes(autorange = "reversed")
                fig_top15artists.update_xaxes(showgrid = False)
                fig_top15artists.update_yaxes(showgrid = False)
                st.plotly_chart(fig_top15artists, theme = "streamlit")
                st.write("Our recommendations will be based on similary between tracks, but popularity could be a secondary factor in our final ranking of similar songs. For that matter, it is important that we have a clear understanding of the observations' distribution.")
                st.write("If popular artists only have two tracks on average in our dataset, a close look at the artists with the most songs in the dataset will help confirm the idea that popular artists are underrepresent.")
                st.write("")
                with st.expander("View Source Code", expanded = False):
                    code = '''
                    fig_top15artists = px.area(top15artists, 
                        x = "Popularity (mean)", 
                        y = "Artists", 
                        color = "Genres", 
                        line_group = "Number of songs in dataset"
                    )
                    fig_top15artists.update_yaxes(autorange = "reversed")
                    fig_top15artists.update_xaxes(showgrid = False)
                    fig_top15artists.update_yaxes(showgrid = False)
                    st.plotly_chart(fig_top15artists, theme = "streamlit")'''
                    st.code(code, language = 'python')

            with tab222:
                st.markdown("#### Top 10 Artists With Most Songs In Dataset")
                st.write("The treemap below indicates that artists with the most songs in the dataset, such as The Beatles with a total of 279 tracks for instance, have very low popularity values: 0.5 on average for Ella Fitzgerald, 1 for Stevie Wonder...")
                fig_top10artists = px.treemap(top10artists,
                                            path = [px.Constant("Top 10 artists with most songs"), 'Artists', "Number of songs in dataset", "Genres"], values = 'Number of songs in dataset',
                                            color = "Popularity (mean)", hover_data = ["Popularity (mean)"], 
                                            color_continuous_scale = 'RdBu',
                                            color_continuous_midpoint = np.average(top10artists["Popularity (mean)"], 
                                                                                    weights = top10artists["Popularity (mean)"]))
                st.plotly_chart(fig_top10artists, theme = "streamlit")
                st.markdown("We do not know how this dataset was constructed but this visualization seems to support our primary analysis: **artists with the highest values of popularity are underrepresented in the dataset to the detriment of artists with low ratings.**")
                st.write("This could limit our results if our system was partly based on popularity.")
                st.write("")
                with st.expander("View Source Code", expanded = False):
                    code = '''fig_top10artists = px.treemap(top10artists,
                                            path = [px.Constant("Top 10 artists with most songs"), 'Artists', "Number of songs in dataset", "Genres"], values = 'Number of songs in dataset',
                                            color = "Popularity (mean)", hover_data = ["Popularity (mean)"], 
                                            color_continuous_scale = 'RdBu',
                                            color_continuous_midpoint = np.average(top10artists["Popularity (mean)"], 
                                                                                    weights = top10artists["Popularity (mean)"]))
                st.plotly_chart(fig_top10artists, theme = "streamlit")'''
                    st.code(code, language = 'python')

        with tab2:
            st.markdown("#### Top 15 Most Popular Songs In Dataset")
            st.write("The visualization below reveals another potential issue with the data: almost half of the most popular songs are reggaeton tracks but in the dataset these songs are listed as belonging to other music genres, such as latino, reggae and latin.")
            fig_top15titres = px.area(top15titres,
                                      x = "Popularity", 
                                      y = "Songs",
                                      color = "Genres")
            fig_top15titres.update_yaxes(autorange = "reversed")
            fig_top15titres.update_xaxes(showgrid = False)
            fig_top15titres.update_yaxes(showgrid = False)
            st.plotly_chart(fig_top15titres, theme = "streamlit")
            st.markdown("**Track genre values in our dataset could be misleading and should probably not be taken into account as a primary feature as a consequence.**")
            st.write("A closer look at the music genres listed in the dataset will help us know more about the reliability of the 'track_genre' variable.")
            st.write("")
            with st.expander("View Source Code", expanded = False):
                code = '''
                fig_top15titres = px.area(top15titres,
                    x = "Popularity", 
                    y = "Songs",
                    color = "Genres")
                fig_top15titres.update_yaxes(autorange = "reversed")
                fig_top15titres.update_xaxes(showgrid = False)
                fig_top15titres.update_yaxes(showgrid = False)
                st.plotly_chart(fig_top15titres, theme = "streamlit")'''
                st.code(code, language = 'python')

        with tab3:
            tab11, tab22, tab33, tab44 = st.tabs(["Top 15 Most Popular Genres", "Popularity of Genres", "Top 5 Songs per Genre", "Distribution of Popularity Values"])
            with tab11:
                st.markdown("#### Top 15 Most Popular Genres In Dataset")
                st.write("With the visualization below we can see that the most popular music genres in the dataset are actually subgenres: Progressive house, Deep house...") 
                st.write("More importantly, they do not correspond to the genres of the most popular songs in the dataset, except for the genre Pop.")
                fig_top15genres = px.area(top15genres,
                                      x = "Popularity (mean)", 
                                      y = "Music genres",
                                      color = "Music genres")
                fig_top15genres.update(layout_showlegend=False)
                fig_top15genres.update_xaxes(showgrid = False)
                fig_top15genres.update_yaxes(showgrid = False)
                st.plotly_chart(fig_top15genres, theme = "streamlit")
                st.write("This finding brings out a discrepancy between popularity values of songs and artists, and their associated genres.")
                st.markdown("**Consequently, our algorithm should be designed in such a manner that the songs recommended to users are the result of a balanced system that is built on content similarity but that also takes into account popularity values of songs, leaving out the notion of music genres.**")
                st.write("")
                with st.expander("View Source Code", expanded = False):
                    code = '''
                    fig_top15genres = px.area(top15genres,
                        x = "Popularity (mean)", 
                        y = "Music genres",
                        color = "Music genres")
                    fig_top15genres.update(layout_showlegend = False)
                    fig_top15genres.update_xaxes(showgrid = False)
                    fig_top15genres.update_yaxes(showgrid = False)
                    st.plotly_chart(fig_top15genres, theme = "streamlit")'''
                    st.code(code, language = 'python')
                
            with tab22:
                st.markdown("#### Popularity of Genres and Number of Songs per Genre")
                st.write("The visualization below supports our first findings: the music genres to which belong the most popular songs (Hip-Hop, Electro, Reggaeton, Dance...) tend to have fewer tracks in the dataset than most other genres, while a lot of subgenres with very low popularity values (Grindcore, Black metal, Bluegrass...) have the most songs in the dataset.")
                ratio = pd.DataFrame(dataset_sample.groupby('track_genre')['track_name'].nunique())
                ratio = ratio.reset_index()
                ratio = ratio.rename(columns = ({'track_genre' : 'Music genres', 'track_name' : 'Number of songs'}))
                ratio['Music genres'] = ratio['Music genres'].str.capitalize()
                ratio = ratio.drop(102, axis = 0)
                ratio = ratio.reset_index()
                ratio = ratio.drop(columns = 'index', axis = 1)
                ratio['Popularity (mean)'] = pop_genre['Popularity (mean)']
                fig_pop_genre = px.scatter(ratio,
                                           x = "Music genres",
                                           y = "Number of songs",
                                           size = "Popularity (mean)",
                                           color = "Popularity (mean)",
                                           hover_name = "Music genres",
                                           color_continuous_scale = "reds")
                fig_pop_genre.update_xaxes(showgrid = False)
                fig_pop_genre.update_yaxes(showgrid = False)
                st.plotly_chart(fig_pop_genre, theme = "streamlit", use_container_width = True)
                st.write("**It thus confirms the idea that popularity should be used in our system as a leverage variable to balance our results.**")
                st.write("")
                with st.expander("View Source Code", expanded = False):
                    code = '''
                    fig = px.scatter(ratio,
                        x = "Music genres",
                        y = "Number of songs",
                        size = "Popularity (mean)",
                        color = "Popularity (mean)",
                        hover_name = "Music genres",
                        color_continuous_scale = "reds"
                    )
                    fig.update_xaxes(showgrid = False)
                    fig.update_yaxes(showgrid = False)
                    st.plotly_chart(fig, theme = "streamlit", use_container_width = True)'''
                    st.code(code, language = 'python')
                
            with tab33:
                st.markdown("#### Top 5 Most Popular Songs per Music Genre")
                st.write("")
                dataset_sample['artists'] = dataset_sample['artists'].str.title()
                dataset_sample['track_name'] = dataset_sample['track_name'].str.capitalize()
                dataset_sample['track_genre'] = dataset_sample['track_genre'].str.capitalize()
                dataset_sample['titre'] = dataset_sample['track_name'] + ' - ' + dataset_sample['artists']
                genres = sorted(set(dataset_sample['track_genre'].unique()))
                choix = st.selectbox("Select a Music Genre", genres)
                st.write("")
                fig, ax = plt.subplots(figsize=(12,8))
                sns.barplot(x = 'popularity', y = 'titre', data = dataset_sample[dataset_sample['track_genre'] == choix].sort_values(by = 'popularity', ascending = False).iloc[:5], ax = ax, palette = 'coolwarm')
                for i in ax.containers:
                    ax.bar_label(i,)
                plt.title('Top 5 Most Popular Songs in Genre \''+choix+'\'', pad = 20, fontsize = 14)
                plt.xlabel('Popularity', labelpad = 20, fontsize = 12)
                plt.ylabel('Songs', labelpad = 20, fontsize = 12)
                boxplot_chart = st.pyplot(fig)    

            with tab44:
                st.markdown("#### Distribution of Popularity Values")
                st.write("If we combine similar subgenres, the distribution of popularity values for each genre seems to be much more homogeneous.")
                st.write("The top 4 most popular genres are then Pop, Dance, Hip-Hop and Reggaeton, which correlates the genres of the most popular songs in the dataset.")
                st.write("")
                st.image("repart_genresEN.png", output_format = "PNG")
                st.write("")
                with st.expander("View Source Code", expanded = False):
                    code = '''
                    sns.catplot(x = 'track_genre', y = 'popularity', kind ='boxen', data = genres, height = 15, aspect = 2, palette = 'coolwarm')
                    plt.xticks(rotation = 60)
                    plt.title('Distribution of popularity values per music genre', pad = 20, fontsize = 14)
                    plt.xlabel('Music genres', labelpad = 20, fontsize = 12)
                    plt.ylabel('Popularity', labelpad = 20, fontsize = 12);'''
                    st.code(code, language = 'python')
                st.write("")
                st.write("")
                st.write("However the visualization below reveals that despite being combined in larger categories, a lot of the most popular genres such as Pop and Hip-Hop still have less tracks in the dataset than other genres such as World music, which accounts for 13.2% of all tracks.")
                fig_num_genres = px.scatter(numgenres,
                                           x = "Genres",
                                           y = "Number of songs",
                                           size = "Popularity (mean)",
                                           color = "Popularity (mean)",
                                           hover_name = "Genres",
                                           color_continuous_scale = "reds")
                fig_num_genres.update_xaxes(showgrid = False)
                fig_num_genres.update_yaxes(showgrid = False)
                st.plotly_chart(fig_num_genres, theme = "streamlit", use_container_width = True)
                st.write("")
                with st.expander("View Source Code", expanded = False):
                    code = '''
                    fig_num_genres = px.scatter(numgenres,
                                           x = "Genres",
                                           y = "Number of songs",
                                           size = "Popularity (mean)",
                                           color = "Popularity (mean)",
                                           hover_name = "Genres",
                                           color_continuous_scale = "reds")
                    fig_num_genres.update_xaxes(showgrid = False)
                    fig_num_genres.update_yaxes(showgrid = False)
                    st.plotly_chart(fig_num_genres, theme = "streamlit", use_container_width = True)'''
                    st.code(code, language = 'python')
            
    with tab_correlations:
        st.markdown("#### Correlation of Songs' Attributes")
        st.write("The heatmap below shows that there is no particular relationship between the dataset's variables, except for the 'loudness' and 'energy' features for which the correlation value is 0.76.")
        st.write("")
        st.image("correlationEN.png", output_format = "PNG")
        st.write("")
        st.write("This indicates that **we will probably have to perform feature selection to reduce the number of input variables by eliminating all redundant or irrelevant features and narrowing down the set of features to those most relevant to our machine learning model.**")
        st.write("")
        with st.expander("View Source Code", expanded = False):
            code = '''cor = dataset.corr()
            plt.figure(figsize = (11,11))
            sns.heatmap(cor, center = 0, annot = True, cmap = 'coolwarm')
            plt.title("Corrélations entre les principales caractéristiques techniques des morceaux", pad = 20, fontsize = 14);'''
            st.code(code, language = 'python')

    with tab_linearite:
        st.markdown("#### Linearity of Songs' Attributes")
        st.write("This visualization also highlights the lack of correlation between the variables as it shows how disseminated the data are.")
        st.write("")
        st.image("linearite.png", output_format = "PNG")
        st.write("")
        st.write("**It finding confirms the need to perform feature selection to identify automatically the most relevant features.** Our model and our results may not be accurate otherwise.")
        st.write("")
        with st.expander("View Source Code", expanded = False):
            code = '''plt.figure(figsize = (12,12))
            sns.pairplot(dataset[['popularity', 'danceability', 'energy', 'loudness', 'mode', 'tempo']], diag_kind='kde', palette = 'coolwarm');'''
            st.code(code, language = 'python')         
    
# Page 4 - Data Modeling
if page == pages[4]: 
    st.header("Data Modeling")
    st.markdown("In this project we do not have a specific outcome variable that we are trying to predict. Instead we have a set of numeric features that we want to use to find collections of observations that share similar characteristics. As a result, our problem is an unsupervised learning problem, and our goal is to group automatically data points according to the similarities between them. In addition, there seems to be no particular correlation between the dataset's variables.")
    st.write("We thus chose an unsupervised learning clustering algorithm to process our data and find natural clusters if they exist in the data: **K-means clustering.** Once objects will be divided into clusters, we will sort them by popularity so that the chance that users enjoy the songs recommended to them is the highest.")
    st.markdown("###")
    
    tab1, tab2, tab3 = st.tabs(["Dimensionality Reduction", "Implementation of K-means Clustering", "Algorithm Configuration"])
    with tab1:
        st.markdown("#### Dimensionality Reduction")
        st.write("We chose a popular unsupervised learning technique for reducing the dimensionality of data: The Principal Component Analysis.") 
        st.write("PCA usually increases interpretability yet, at the same time, minimizes information loss. It also helps to find the most significant features in a dataset and it should help in finding a sequence of linear combinations of variables. This will be particularly useful in our situation since our variables do not seem to have any type of relationship.")
        st.markdown("We performed PCA to three different versions of our dataset: \n- one that includes all 125 music subgenres; \n- one that includes 24 groups of music genres; \n- one that does not contain any genre at all.")
        st.markdown("###")
        tab_complet, tab_avec, tab_no = st.tabs(["PCA to Entire Dataset", "PCA to Dataset With 24 Groups of Music Genres", "PCA to Dataset Without Any Genre"])
        with tab_complet:
            tab_var, tab_sum = st.tabs(["Explained Variance Ratio", "Cumulative Explained Variance"])
            with tab_var:
                st.markdown("#### Selecting the Number of Dimensions")
                st.write("The explained variance ratio is calculated to select the optimal number of dimensions in the PCA. The visualization below shows a PCA scree plot of the dataset, with the blue line representing the explained variance ratio. The ratio represents the variance explained by each of the principal components, starting with the first component, which is the principal component that explains most of the variance.")     
                st.write("")
                st.image("varexpfull.png", output_format = "PNG")
            with tab_sum:
                st.write("When the ratios are summed, the total value is equal to 1, indicating that the 129 components together explain 100% of the variance of the dataset. More usefully, the variance ratio explained is used as a cumulative sum, such as indicated by the blue curve in the visualization below.") 
                st.write("")
                st.image("cumsumfull.png", output_format = "PNG")
                st.write("")
                st.write("After performing ACP to the entire dataset, we found that **109 components were required to explain at least 90% of the information in the dataset from the cumulative explained variance.** Since only 15% of the components were reduced, we considered such results to be below our expectations and decided in consequence to perform ACP to another version of the dataset: a version where all subgenres were regrouped into 24 larger music genres.")
        with tab_avec:
            tab_var, tab_sum = st.tabs(["Explained Variance Ratio", "Cumulative Explained Variance"])
            with tab_var:
                st.markdown("#### Selecting the Number of Dimensions")
                st.write("On this second attempt, we replaced the 114 original subgenres with only 24 distinct values instead, and performed ACP to this new version of the dataset. The visualization below illustrates the variance of each principal component.")
                st.write("")
                st.image("varexpwith.png", output_format = "PNG")
            with tab_sum:
                st.write("After plotting the cumulative explained variance, we can see that where our line is drawn for 90%, **the total explained variance is approximately at 35 components.** We managed here to reduce 28% of the components.")
                st.write("")
                st.image("cumsumwith.png", output_format = "PNG")
                st.write("")
                st.write("Let's see if we can have an even higher reduction rate when the 'track_genre's' variable is completely removed from the dataset.")
        with tab_no:
            tab_var, tab_sum = st.tabs(["Explained Variance Ratio", "Cumulative Explained Variance"])
            with tab_var:
                st.markdown("#### Selecting the Number of Dimensions")
                st.write("On this third attempt, we deleted the 'track_genre' variable from the dataset to perform APC to the songs features only. The visualization below illustrates the variance of each principal component for this third version of the dataset.")
                st.write("")
                st.image("varexpwo.png", output_format = "PNG")
            with tab_sum:
                st.write("After performing ACP to this third version of the dataset, we found that **12 components out of 14 were required to explain at least 90% of the information in the dataset from the cumulative explained variance.** Only 14% of the components were reduced here, which we considered was far below our expectations once again.")
                st.write("")
                st.image("cumsumwo.png")
                st.write("")
                st.write("After three attemps with three different versions of the dataset, we were not able to reduce effectively the dimensionality of our dataset. We will go on and proceed with the clustering.")
                
    with tab2:
        st.markdown("#### Implementation of K-means Clustering")
        st.write("K-means clustering seemed to be the most appropriate type of algorithm for this project, for it usually helps maximise the similarity of data points within clusters. Also k-means clustering is a comparatively fast algorithm which performance, unlike most other clustering algorithms, scales linearly with the number of data points in the dataset.")
        st.write("K-means clustering has disadvantages of course, such as sensitivity to outliers or scale. But we believed that it still was the most appropriate, reliable and well-studied unsupervised clustering algorithm for our project.")
        st.write("")
        st.markdown("#### Determining the Optimal Number of Clusters")
        st.write("Like many other clustering algorithms, k-means clustering requires the number of clusters that will be created to be specified ahead of time. For that reason, determining the optimal number of clusters is a fundamental part in generating the clusters.")
        st.write("There are two main methods to find the optimal number of clusters: The elbow curve method and the Silhouette analysis. We will use the elbow method first and only refer to the Silhouette analysis if needed.")
        st.write("As we did not succeed in reducing effectively the number of features in our dataset, we will perform k-means clustering to two different versions of our dataset: the version with 24 groups of music genres and the version with no genre at all. We will then be able to compare both clusterings and evaluate their relevance according to what we are trying to achieve.")
        
        tab11, tab22 = st.tabs(["K-means on Dataset With 24 Groups of Genres", "K-means on Dataset Without Any Genre"])
        with tab11:
            st.markdown("##### The Elbow Method")
            st.write("The elbow method is probably the most well-known method for determining the optimal number of clusters. It is based on calculating the Within-Cluster-Sum of Squared Errors (WSS) for different number of clusters (k) and selecting the k for which change in WSS first starts to diminish. In the plot of WSS-versus-k, this is visible as an elbow.")
            st.write("We obtained the following plot for WSS-vs-k for our dataset with 24 groups of genres.")
            st.image("kmeansEN.png", output_format = "PNG")
            st.write("")
            st.write("**We can see here a clear elbow at k = 31.**")
            st.write("The heatmap below shows all median values for each variable per cluster, after instantiating the k-means estimator class with n_clusters = 31.")
            st.write("")
            st.image("heatmapK1EN.png", output_format = "PNG")
            st.write("")
            st.write("A closer look at this visualization let us see that each music genre was assigned to one cluster, as it seems the algorithm created clusters based on the genres more than any other variables. Such results are not what we were looking for as our goal was to find collections of observations that share similar characteristics that we were not already aware of.")
            st.write("")
            st.image("heatmapK11EN.png", output_format = "PNG")
            st.write("")
            st.write("As a consequence, we decided to run a k-means algorithm on the dataset after removing the track genre variable.")
            
        with tab22:
            tab_elbow, tab_k_elbow, tab_silhouette, tab_clusters = st.tabs(["Elbow Method", "KElbowVisualizer", "Silhouette Coefficient", "Clusters"])
            with tab_elbow:
                st.markdown("##### The Elbow Method")
                st.write("The 'track_genre' variable was deleted and k-means clustering was run on this other version of the dataset. The plot below does not allow us to determine the exact point where the rate of decrease shifts as the elbow is not as clear and sharp as previously. This indicates that data is not clearly clustered.")
                st.write("")
                st.image("elbowEN.png", output_format = "PNG")
                st.write("")
                st.write("In order to determine more precisely the value of k, we used the YellowBrick library as it can implement the elbow method with a distortion metric which computes the sum of squared distances from each point to its assigned center.")
            with tab_k_elbow:
                st.markdown("##### KElbowVisualizer")
                st.write("The KElbowVisualizer function fit the k-means model for a range of clusters values between 2 to 17. As shown in the figure below, the black vertical line indicates that the elbow point is achieved with 8 clusters.")
                st.write("")
                st.image("kelbow.png", output_format = "PNG")
                st.write("")
                st.write("In order to validate that result, we decided to use the Silhouette Method to have an additional way of determining the optimal number of clusters.")
            with tab_silhouette:
                st.markdown("##### The Silhouette Coefficient")                
                st.write("The silhouette value measures how similar a point is to its own cluster (cohesion) compared to other clusters (separation). The Silhouette Score reaches its global maximum at the optimal k. This should ideally appear as a peak in the Silhouette Value-versus-k plot.")
                st.write("Here is the plot for our own dataset (no genre):")
                st.write("")
                st.image("silhouetteEN.png", output_format = "PNG")
                st.write("")
                st.write("The range of the Silhouette value is between +1 and -1. A high value is desirable and indicates that the point is placed in the correct cluster. If many points have a negative Silhouette value, it may indicate that we have created too many or too few clusters.")
                st.write("In our case, the Silhouette scores are close to 0 which indicates that our algorithm is not very efficient.")
                st.write("The visualization also reveals that the highest Silhouette value is 7.")
                st.write("")
                st.write("To decide between the two values (8 as obtained with the KElbowVisualizer and 7 as obtained with the Silhouette score plot) we decided to display a Silhouette Plot for every value of k ranging from 6 to 10.")
                st.write("")
                tab_6, tab_7, tab_8, tab_9, tab_10 = st.tabs(["6 Clusters", "7 Clusters", "8 Clusters", "9 Clusters", "10 Clusters"])
                with tab_6:
                    st.image("6EN.png", output_format = "PNG")
                with tab_7:
                    st.image("7EN.png", output_format = "PNG")
                with tab_8:
                    st.image("8EN.png", output_format = "PNG")
                with tab_9:
                    st.image("9EN.png", output_format = "PNG")
                with tab_10:
                    st.image("10EN.png", output_format = "PNG")
                st.write("")
                st.write("We already know that the values of the silhouette coefficient are low which is an indicator of low efficiency.")
                st.write("Consequently we need to look at the distribution between clusters. Clustered areas should have similar sizes or well-distributed points. **The value of n_clusters = 7 seems to be the most optimal value even though all 7 clusters do not have similar sizes.**")
                st.write("Let's look at the median values for each variable per cluster to see if this clustering seems to be more accurate than previously.")
            with tab_clusters:
                st.markdown("##### Clusters") 
                st.write("The heatmap below shows all median values for each variable per cluster, after instantiating the k-means estimator class with n_clusters = 7.")
                st.write("")
                st.image("heatmapclustersEN.png", output_format = "PNG")   
                st.write("")
                st.write("Median values of variables such as the duration, loudness or tempo indicate that each cluster seems to contain a different type of tracks, which was our goal. We aimed at grouping songs with similar characteristics.") 
                st.write("We have to keep in mind nonetheless that our recommendation system should not be based exclusively on this clustering algorithm, as Silhouette Scores were really low and recommendations might be inaccurate.")
                st.write("Next we explain in detail how we configured our algorithm to make it relevant and to ensure it returns the best recommendations possible.")

    with tab3:
        st.markdown("#### Algorithm Configuration")
        st.write("We did not really succeed in reducing the dimensionality of our dataset and we know that our clusters do not have good Silhouette Scores. The former issue could be linked to the preprocessing of the data while the latter could be the result of the algorithm's sensitivity to the dataset's outliers, although in this case the notion of outlier is a very subjective notion as our data represent audio tracks in a lot of different and specific genres. It is also possible that we did not choose the right machine learning algorithm for our dataset.")
        st.write("As a result we decided to build the first part of our recommendation system using the clusters created with the k-means algorithm, and add manual steps in order to present users with the best recommendations possible.")
        st.write("We also decided to base our system on another approach and create a similarity matrix. The elements of the similarity matrix will measure pairwise similarities of objects and we will be able to achieve the best results possible by selecting what we consider to be the most appropriate metric value.")
        st.write("")
        
        tabAlgo1, tabAlgo2 = st.tabs(["First Algorithm", "Second Algorithm: Similarity Matrix"])
        with tabAlgo1:
            tabET1, tabET2, tabET3, tabET4, tabET5, tabET6 = st.tabs(["Step 1", "Step 2", "Step 3", "Step 4", "Step 5", "Step 6"])
            with tabET1:
                st.write(f"**Step 1 : Filtering Of Songs Based On The Seed Song's Cluster**")
                graph1 = graphviz.Digraph()
                graph1.edge('Seed Song', "Cluster", color = "red")
                st.graphviz_chart(graph1, use_container_width = True)
            
            with tabET2:
                st.write(f"**Step 2 : Additional Filtering Based On Specific Attributes Within Cluster**")
                graph2 = graphviz.Digraph()
                graph2.edge('Seed Song', "Cluster")
                graph2.edge("Cluster", 'Tempo', color = "red")
                graph2.edge("Cluster", 'Loudness', color = "red")
                graph2.edge("Cluster", 'Energy', color = "red")
                graph2.edge("Cluster", 'Acousticness', color = "red")
                graph2.edge("Cluster", 'Danceability', color = "red")
                st.graphviz_chart(graph2, use_container_width = True)
                st.write("")
                st.write("At the end of step 2, results are saved in a dataframe.")

            with tabET3:
                st.write(f"**Step 3 : Filtering Of Songs Based On Seed Song's Music Genre/Genres**")
                graph3 = graphviz.Digraph()
                graph3.edge('Seed Song', "Cluster")
                graph3.edge("Cluster", 'Tempo')
                graph3.edge("Cluster", 'Loudness')
                graph3.edge("Cluster", 'Energy')
                graph3.edge("Cluster", 'Acousticness')
                graph3.edge("Cluster", 'Danceability')
                graph3.edge('Seed Song', 'Music Genres', color = "red")
                st.graphviz_chart(graph3, use_container_width = True)

            with tabET4:
                st.write(f"**Step 4 : Additional Filtering Based On Specific Attributes**")
                graph4 = graphviz.Digraph()
                graph4.edge('Seed Song', "Cluster")
                graph4.edge("Cluster", 'Tempo')
                graph4.edge("Cluster", 'Loudness')
                graph4.edge("Cluster", 'Energy')
                graph4.edge("Cluster", 'Acousticness')
                graph4.edge("Cluster", 'Danceability')
                graph4.edge('Seed Song', 'Music Genres')
                graph4.edge('Music Genres', 'Tempo', color = 'red')
                graph4.edge('Music Genres', 'Loudness', color = 'red')
                graph4.edge('Music Genres', 'Energy', color = 'red')
                graph4.edge('Music Genres', 'Acousticness', color = 'red')
                graph4.edge('Music Genres', 'Danceability', color = 'red')
                st.graphviz_chart(graph4, use_container_width = True)
                st.write("")
                st.write("At the end of step 4, results are saved in a second dataframe.")

            with tabET5:
                st.write(f"**Step 5 : Grouping Of All Songs By Seed Artist**")
                graph5 = graphviz.Digraph()
                graph5.edge('Seed Song', "Cluster")
                graph5.edge("Cluster", 'Tempo')
                graph5.edge("Cluster", 'Loudness')
                graph5.edge("Cluster", 'Energy')
                graph5.edge("Cluster", 'Acousticness')
                graph5.edge("Cluster", 'Danceability')
                graph5.edge('Seed Song', 'Music Genres')
                graph5.edge('Music Genres', 'Tempo')
                graph5.edge('Music Genres', 'Loudness')
                graph5.edge('Music Genres', 'Energy')
                graph5.edge('Music Genres', 'Acousticness')
                graph5.edge('Music Genres', 'Danceability')
                graph5.edge('Seed Song', "Other Songs By Seed Artist", color = 'red')
                st.graphviz_chart(graph5, use_container_width = True)
                st.write("")
                st.write("If the dataset contains other songs by the seed artist, these songs will be saved at the end of step 5 in a third dataframe.")

            with tabET6:
                st.write(f"**Step 6 : Setting Priorities**")
                st.write("")
                tab61, tab62, tab63 = st.tabs(["Priority 1", "Priority 2", "Priority 3"])                    
                with tab61:
                    st.write(f"**Priority 1 : Songs By Seed Artist**")
                    graph11 = graphviz.Digraph()
                    graph11.edge("Filtering process", "Other songs by seed artist", color = "red")
                    st.graphviz_chart(graph11, use_container_width = True)
                
                with tab62:
                    st.write(f"**Priority 2 : Songs In Similar Genres As Seed Song**")
                    graph22 = graphviz.Digraph()
                    graph22.edge("Filtering process", "Other songs by seed artist")
                    graph22.edge("Other songs by seed artist", "Songs in similar genres", label = "If no result", color = "red")
                    st.graphviz_chart(graph22, use_container_width = True)
                        
                with tab63:
                    st.write(f"**Priority 3 : Songs From Seed Song's Cluster**")
                    graph33 = graphviz.Digraph()
                    graph33.edge("Filtering process", "Other songs by seed artist")
                    graph33.edge("Other songs by seed artist", "Songs in similar genres")
                    graph33.edge("Songs in similar genres", "Songs from seed song's cluster", label = "If no result", color = "red")
                    st.graphviz_chart(graph33, use_container_width = True)

        with tabAlgo2:
            st.write("**Creating the Similarity Matrix**")
            st.write("")
            st.write("We created the matrix using the scipy.spatial.distance module and 'pdist' function. This function computes pairwise distances between points using Euclidean distance as the default distance metric. The main advantage of this method is the wide range of metrics available.")
            st.write("After trying out all metrics, we found that the Mahalanobis metric was the one which returned the best results. The Mahalanobis distance is scale-invariant, yet it takes into account the correlations of the dataset. It is also widely used in cluster analysis.")
            st.write("We chose to use the matrix's results without adding any human intervention.")
            st.write("Below is a preview of the matrix:")
            st.write("")
            st.image("matrice.png",  output_format = 'PNG')

# Page 5 - Conclusion
if page == pages[5]: 
    st.header("Conclusion")

    tab1, tab2 = st.tabs(['Research Objectives', "Ways of Improvement"])

    with tab1:
        st.markdown("#### Research Objectives")
        st.write("In this project, our goal was to create a music recommendation system based on content similarity, using a dataset which contained thousands of songs in various music genres.") 
        st.write("We chose to group similar tracks automatically by running a k-means algorithm on our dataset, but we were not very successful in creating self-sufficient clusters. We were still able to use the clusters by adding manual steps and fine-tuning our function to create the first part of our recommendation system.") 
        st.write("The second part was based on a similarity matrix. Using the Mahalanobis metric allowed our function to return the best recommendations possible, given the limited number of songs avaible in our dataset.")
        st.write("We can thus say that we achieved our goal as we were able to build a music recommendation system allowing users to choose between two songs similar to a seed song of their liking. Although the relevancy of the recommendations is subjective, our system was designed and adjusted so that the songs suggested to users can be considered the most interesting suggestions possible.")
    
    with tab2:
        st.markdown("#### Ways of Improvement")
        st.markdown("With more time, we could have: \n- Designed a review system and store ratings of recommendations in a database;")
        st.markdown("- Searched for additional resources and add songs to the dataset to have a broader selection of tracks to recommend;")
        st.markdown("- Run alternative dimensionality reduction and unsupervised algorithms to get better results than the clusters created in this project.")

# Page 6 - Music recommendations
if page == pages[6]: 
    st.header("Music recommendations")
    st.markdown('##')
    st.markdown("##### Please enter the name of the artist or the song you would like to listen to: ")    
    
    # Cluster-based function: 
    def reco(track):
        df_track = dataset.loc[(dataset['artist_track'] == track)][['artists', 'track_name', 'artist_track', 'track_genre', 'cluster_genres', 'tempo', 'energy', 'loudness', 'popularity']]
        artist = df_track['artists']
  
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
                & (dataset['danceability'] > (o - 0.3)) & (dataset['danceability'] < (o + 0.3))][['artists', 'track_name', 'artist_track', 'track_genre', 'cluster_genres', 'tempo', 'energy', 'loudness', 'popularity']]
  
        track_infos = track_infos.drop_duplicates(subset = ['artist_track'])
        track_infos = track_infos[track_infos['artist_track'] != track]
        track_infos = track_infos[~track_infos['artist_track'].str.contains(track[:20])]
        track_infos = track_infos.sort_values('popularity', ascending = False)
  
        list_genres = list(df_track['track_genre'])
        df_genre = dataset[dataset['track_genre'].isin(list_genres)]

        for i, j, k, l, m, n, o in zip(df_genre['artist_track'], df_genre['cluster_genres'], df_genre['tempo'], df_genre['loudness'], df_genre['energy'], dataset['acousticness'], dataset['danceability']):
            if i == track:
                df_genre = df_genre.loc[(df_genre['cluster_genres'] == j)
                & (df_genre['tempo'] > (k - 7)) & (df_genre['tempo'] < (k + 7))
                | ((df_genre['tempo'] > ((k/2) - 7)) & (df_genre['tempo'] < ((k/2) + 7)))
                | ((df_genre['tempo'] > ((k*2) - 7)) & (df_genre['tempo'] < ((k*2) + 7)))
                & (df_genre['loudness'] > (l - 2.5)) & (df_genre['loudness'] < (l + 2.5)) 
                & (df_genre['energy'] > (m - 0.2)) & (df_genre['energy'] < (m + 0.2))
                & (dataset['acousticness'] > (n - 0.2)) & (dataset['acousticness'] < (n + 0.2))
                & (dataset['danceability'] > (o - 0.2)) & (dataset['danceability'] < (o + 0.2))][['artists', 'track_name', 'artist_track', 'track_genre', 'cluster_genres', 'tempo', 'energy', 'loudness', 'popularity']]
  
        df_genre = df_genre[df_genre['artist_track'] != track]
        df_genre = df_genre.drop_duplicates(subset = ['artist_track'])
        df_genre = df_genre[~df_genre['artist_track'].str.contains(track[:20])]
        df_genre = df_genre.sort_values('popularity', ascending = False)
 
        df_artist = dataset.loc[(dataset['artists'].isin(artist)) 
        | (dataset['artists'].str.contains(track[:9]))][['artists', 'track_name', 'artist_track', 'track_genre', 'cluster_genres', 'tempo', 'energy', 'loudness', 'popularity']]   
  
        df_artist = df_artist.drop_duplicates(subset = ['artist_track'])
        df_artist = df_artist[df_artist['artist_track'] != track] 
        df_artist = df_artist[~df_artist['artist_track'].str.contains(track[:20])]
        df_artist = df_artist.sort_values('popularity', ascending = False)
   
        if not df_artist.empty:
            return df_artist['artist_track'].values[:1]
        elif not df_genre.empty:  
            return df_genre['artist_track'].values[:1]
        elif not track_infos.empty:
            return track_infos['artist_track'].values[:1]        

    # Similarity matrix
    new_index = list(dfm.index)
    pairwise = pd.DataFrame(squareform(pdist(dfm, 'mahalanobis')))
    pairwise['artist_track'] = new_index
 
    # Matrix-based function:
    def recom(track):
        df = pairwise[pairwise['artist_track'] == track]
        df = df.reset_index(drop = True).T
        df = df[df[0] != track]
        df['artist_track'] = new_index
        df = df.rename(columns = {0 : 'distance'})
        df = df[df['distance'] != 0]
        df = df.sort_values(by = 'distance')
        if not df.empty:
            return df['artist_track'].values[:1]
        else:
            return st.write("") 

    # Deezer player:
    def player(reco):
        url = "https://api.deezer.com/search?q=" + reco
        request = requests.get(url)
        parsing = json.loads(request.text)
        reco_id = str(parsing['data'][0]['id'])
        link = "https://widget.deezer.com/widget/auto/track/" + reco_id
        return components.html(f'<iframe title="deezer-widget" src={link} width="100%" height="150" frameborder="0" allowtransparency="true" allow="encrypted-media; clipboard-write"></iframe>')

    # Seed track:
    OG = "Nirvana - Smells like teen spirit"
    OG_index = int(dfm2[dfm2['artist_track'] == OG].index.values)
    options = dfm2['artist_track']
    search_bar = st.selectbox("", options, index = OG_index, label_visibility = "collapsed", key = "search")
    player(search_bar)
    
    # Search bar update:
    def callback1():
        st.session_state.search = options[index_reco1]
        
    def callback2():
        st.session_state.search = options[index_reco2]

    # Display of recommendation:
    if search_bar != OG:
        st.markdown('#')
        st.write(f"You selected the song **{search_bar}**, here are the tracks we recommend:")
        track = str(search_bar) 
        reco1 = reco(track)[0]
        reco2 = recom(track)[0]
        index_reco1 = int(dfm2[dfm2['artist_track'] == reco1].index.values)
        index_reco2 = int(dfm2[dfm2['artist_track'] == reco2].index.values)
        col1, col2 = st.columns([1,1])
        with col1:
            st.write(f"**{reco1}**")
            player(reco1)
            bouton1 = st.button('Next recommendation', on_click = callback1, key = 1)
        with col2:
            st.write(f"**{reco2}**")
            player(reco2)
            bouton2 = st.button('Next recommendation', on_click = callback2, key = 2)  

    
