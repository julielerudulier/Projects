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

# Menu latéral 
st.sidebar.title("INTITULE DU CERTIFICAT - Research projet")
st.sidebar.header("Menu")
pages = ["Home page", "Introduction", "Datasets", "Data visualizations", "Data modeling", "Conclusion", "Recommendation systems"]
page = st.sidebar.radio("", options = pages)

st.sidebar.header("Author")
st.sidebar.markdown("""
Julie Le Rudulier - [LinkedIn](https://www.linkedin.com/in/julielerudulier/)  
""") 

# Gestion des images et des datasets
image = Image.open('target.png')
flimage = ImageOps.mirror(image)
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

# Page d'accueil 
if page == pages[0]: 
    st.title("Research project: Music recommendation systems")
    st.markdown("#### Predict songs accurately using 2 different Machine Learning algorithms")
    st.markdown("##")
    col1, col2, col3 = st.columns([1,5,1])
    with col1:
        st.write("")
    with col2:
        st.markdown("![Alt Text](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbmR5MWJrZjZsbTh3aHNtNDdxMzJubHdnMGMzNTZ2ZHUyNjF4bXI1NiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/tUxON6fVGX2LD87p57/giphy.gif)")
    with col3:
        st.write("")

    st.markdown("##")

    st.write("Projet réalisé dans le cadre de la formation **Data Analyst** (formation continue), Cohorte Novembre 2022, [DataScientest](https://datascientest.com/).", unsafe_allow_html=True)

# Page 1 - Présentation du projet
if page == pages[1]: 
    st.header("Introduction")
    st.markdown("#### Objectif et contexte")
    tab01, tab02 = st.tabs(["Cadre du sujet", "Approche choisie"])
    with tab01:
        st.write("L'objectif de notre projet est de :blue[réaliser une recommandation pertinente de musique à un utilisateur]. À partir d’un morceau de départ que l’utilisateur aura préalablement choisi ou écouté, notre algorithme proposera un ou plusieurs nouveaux morceaux basés sur leur similarité avec le morceau de départ.")
        st.write("Pour mener à bien notre projet, nous disposions de :blue[5 datasets différents] contenant notamment : \n- des titres de morceaux de musique diffusés sur la plateforme Spotify, et leurs caractéristiques techniques associées (ex : tempo, durée, caractère dansant...) ; \n- des hashtags utilisés sur Twitter lors de publications relatives à des titres de musique (ex : #NowPlaying, #LiveMusic...) ; \n- des hashtags publiés sur Twitter et leurs valeurs sentimentales associées, catégorisées à travers plusieurs dictionnaires d'analyse de sentiments.") 
    with tab02:
        st.write("Pour atteindre notre but et effectuer des recommandations pertinentes, nous avons basé notre approche sur le regroupement des morceaux selon leurs caractéristiques techniques. :blue[Notre modèle repose donc sur un filtrage basé sur le contenu.]")
        st.write("Dans un deuxième temps, :blue[nous avons également construit un autre algorithme à partir d'une matrice de similarité] pour l'ensemble de nos morceaux. Cette deuxième approche est assez similaire dans le principe à la première, mais elle est différente dans son calcul et sa mise en place. Cela nous a permis d'obtenir un deuxième modèle robuste et pertinent, qui nous permet d'effectuer des recommandations intéressantes tout en étant différentes de celles proposées par notre premier modèle.")
        st.write("Nous avons enfin testé une approche plus classique en matière de recommandations, celle du :blue[filtrage collaboratif]. Nous n'avons pas retenu cette dernière approche, car nous ne disposions pas suffisamment de données pertinentes pour pouvoir la mettre en place efficacement.")

# Page 2 - Jeux de données
if page == pages[2]: 
    st.header("Jeux de données")
    st.write("Dans cette première phase du projet, notre objectif a été de comprendre la nature de notre objectif final, tout en nous assurant de bien maîtriser le contexte de nos datasets et de nos données, ainsi que la signification de chaque variable. Cette première phase nous a permis de nous faire une première opinion sur nos différents jeux de données.")
    st.markdown("###")

    tab1, tab2 = st.tabs(['Cadre : jeux de données utilisés', 'Pertinence des données et compréhension des variables'])
    with tab1:
        st.markdown("#### Cadre : jeux de données utilisés")
        st.write("Dans le cadre de notre projet, nous disposions au départ de 5 datasets différents. Pour chacun d’entre eux, nous avons réalisé une analyse détaillée de la structure globale et de chacune des variables (étude des types de données, recherche d’éventuels doublons et de valeurs manquantes, calcul des valeurs médianes ou du mode…). Nous avons répertorié ces différentes analyses en annexes du rapport.") 
        st.write("Voici une présentation synthétique de nos jeux de données.")
        st.markdown("###")

        tabDS1, tabDS2, tabDS3, tabDS4, tabDS5 = st.tabs(['Dataset #1', 'Dataset #2', 'Dataset #3', 'Dataset #4', 'Dataset #5'])
        with tabDS1:
            st.markdown("##### Dataset #1 : Dataset")
            st.write("Notre premier dataset se nomme “Dataset”. Il provient de la plateforme de streaming Spotify et contient des informations relatives à plusieurs dizaines de milliers de morceaux de musique, répartis dans 125 genres musicaux différents. Ces informations concernent notamment la vitesse des morceaux, leur éventuel caractère acoustique, instrumental, joyeux, ou encore dansant.") 
            st.write("Ce dataset contient 20 colonnes et 114 000 lignes, pour un poids total de 20,12 Mo. Il est composé de 14 variables numériques, 5 variables catégorielles et une variable booléenne.")
            st.write("Aperçu du dataset :")
            dataset_5 = pd.read_csv("dataset_5.csv", index_col = 0)
            st.dataframe(dataset_5)

        with tabDS2:
            st.markdown("##### Dataset #2 : User Track Hashtag Timestamp")
            st.write("Notre second dataset se nomme “User Track Hashtag Timestamp”. Il contient des informations relatives à des hashtags publiés sur Twitter lors d’écoute de morceaux de musique. Pour chaque publication sont définis un numéro d’identification du morceau écouté, un numéro d’identification de la personne qui écoute le morceau, le hashtag associé au morceau et/ou au post, ainsi que le jour et l’heure à laquelle a été publié le post par l’utilisateur.") 
            st.write("Ce dataset contient 4 colonnes et 17 560 113 lignes, pour un poids total de 1,31 Go.")
            st.write("Aperçu du dataset :")
            utht_5 = pd.read_csv("user_track_hashtag_timestamp_5.csv", index_col = 0)
            st.dataframe(utht_5)
        
        with tabDS3:
            st.markdown("##### Dataset #3 : Sentiment Values")
            st.write("Notre troisième dataset se nomme “Sentiment Values”. Il contient des informations relatives aux sentiments exprimés sur Twitter par l’intermédiaire de hashtags. Les données comprennent les hashtags eux-mêmes, ainsi que les sentiments associés, définis selon 4 dictionnaires de sentiments différents. Pour chacun de ces dictionnaires ont été listés le minimum, maximum, la somme et la moyenne de tous les sentiments exprimés pour un même hashtag existant.") 
            st.write("Ce dataset contient 21 colonnes et 5 290 lignes, pour un poids total de 382 Ko.")
            st.write("Aperçu du dataset :")
            sentiment_5 = pd.read_csv("sentiment_values_5.csv", index_col = 0)
            st.dataframe(sentiment_5)

        with tabDS4:
            st.markdown("##### Dataset #4 : Context Content Features")
            st.write("Notre quatrième dataset se nomme “Context Content Features”. Chaque ligne de ce dataset correspond aux informations techniques de morceaux de musique ayant fait l'objet d'un ou plusieurs tweets. Ce jeu de données contient également les informations de localisation (si elles existent), de langue et de timing des tweets.") 
            st.write("Ce dataset contient 22 colonnes et 11 614 671 lignes, pour un poids total de 2,21 Go.")
            st.write("Aperçu du dataset :")
            ccf_5 = pd.read_csv("ccf_reduit_5.csv", index_col = 0)
            st.dataframe(ccf_5)

        with tabDS5:
            st.markdown("##### Dataset #5 : Module4 Cleaned")
            st.write("Notre cinquième et dernier dataset se nomme 'Module4 Cleaned'. Ce jeu de données regroupe en fait les datasets 'Sentiment Values', 'User Track Hashtag Timestamp' et 'Context Content Features'. Chacun de ces datasets a été préalablement nettoyé, certaines colonnes ont été supprimées ou renommées, d’autres regroupées.") 
            st.write("Nous avons décidé de ne pas utiliser ce dataset, car nous ne connaissions pas l’approche méthodologique selon laquelle il a été constitué. Dans la mesure où nous disposons des trois datasets originaux à partir desquels ce jeu de données a été conçu, nous avons préféré faire nos regroupements de données nous-mêmes, dans d'autres datasets.")
            st.write("Aperçu du dataset :")
            module4_5 = pd.read_csv("module4_cleaned_5.csv", index_col = 0)
            st.dataframe(module4_5)
            st.divider()
            st.write("Tous nos datasets sont accessibles librement sur le site [kaggle.com](https://kaggle.com/).", unsafe_allow_html=True)

    with tab2:
        st.markdown("#### Pertinence des données et compréhension des variables")

        tab21, tab22, tab23 = st.tabs(['Variables cibles', 'Variables pertinentes pour prédire la variable cible', 'Particularités et/ou limites rencontrées'])
        with tab21:
            st.markdown("##### Variables cibles")
            st.write("Notre objectif final est de prédire correctement à un utilisateur un titre de musique en fonction d’un autre morceau que cet utilisateur aura écouté ou choisi au préalable. Théoriquement, nos variables cibles sont donc le titre du morceau suggéré, ainsi que le nom du ou des interprètes de ce morceau.")
            st.markdown("##")

        with tab22:
            st.markdown("##### Variables pertinentes pour prédire la variable cible ")
            st.write("Dans la mesure où nous avons décidé de baser notre modèle sur la similarité des caractéristiques techniques, l’ensemble des données du dataset 'Dataset' nous ont semblé pertinentes, car ce jeu de données contient toutes les informations descriptives des morceaux. Il contient également les genres auxquels appartiennent les morceaux, ainsi que leur popularité ; ces deux variables sont déterminantes pour l'atteinte de notre objectif.")
            st.write("S'agissant des hashtags associés à chaque titre, ils nous ont également semblé pertinents à première vue car ils donnent dans la plupart des cas des indications relatives au genre musical et/ou à l’époque de la chanson écoutée (ex : '#90s', '#OldSchool'...).") 
            st.write("Nous pensions au départ que l'étude des sentiments codifiés par les dictionnaires d’analyse des sentiments aurait pu nous permettre d'affiner notre modèle, mais ayant choisi un système de filtrage par le contenu, ces variables se sont finalement avérées inutiles pour la mise en place de notre algorithme.")
            st.write("Enfin la variable 'track_id' (l’identifiant unique d’un morceau de musique donné) nous a été particulièrement utile pour faire le lien entre nos différents jeux de données. En effet, cette variable, couplée aux caractéristiques techniques des morceaux, nous a permis de relier entre eux nos principaux datasets, et de tester d'autres approches alternatives comme le filtrage collaboratif.")                           
            st.markdown("##")

        with tab23:
            st.markdown("##### Particularités et/ou limites rencontrées avec les jeux de données")
            st.write("Dans l’ensemble, nos jeux de données étaient relativement bien structurés et plutôt faciles à manipuler. Nous avons tout de même rencontré quelques difficultés au départ, notamment concernant les datasets 'Context Content Features' et 'Sentiment values'.")
            st.write("Pour le fichier 'Context Content Features', les difficultés ont été multiples : la taille du fichier tout d'abord rendait son utilisation difficile dans un notebook partagé sur un environnement en ligne comme Google Collaboratory. Mais surtout, il s’est avéré que le fichier avait des erreurs de colonnes sur certaines lignes : il paraissait en effet avoir 22 colonnes, tandis que certaines lignes étaient détectées comme en ayant 23. La gestion de ces lignes nous a causé énormément de difficultés au début de notre exploration.")
            st.write("Pour le fichier 'Sentiment values' en revanche, les 4 dernières colonnes du fichier n’avaient pas de libellés. La structure du fichier laissait à penser qu’il s’agissait potentiellement d’un cinquième dictionnaire de sentiment ou d’un regroupement (moyenne ou autre) des quatres autres dictionnaires. Sans certitude, nous avons préféré ignorer ces dernières colonnes.")   

# Page 3 - Visualisations
if page == pages[3]:
    st.header("Visualisations")
    tab_top, tab_correlations, tab_linearite = st.tabs(["Classements et répartitions des valeurs", "Corrélations", "Linéarité"])
    with tab_top:
        tab1, tab2, tab3 = st.tabs(["Artistes", "Morceaux", "Genres musicaux"])
        with tab1:
            tab111, tab222 = st.tabs(["Top 15 des artistes les plus populaires", "Top 10 des artistes les plus présents dans notre dataset"])
            with tab111:
                st.markdown("#### Top 15 des artistes les plus populaires")
                fig_top15artists = px.area(top15artists, 
                                           x = "Popularité moyenne", 
                                           y = "Artistes", 
                                           color = "Genres musicaux", 
                                           line_group = "Nombre total de morceaux")
                fig_top15artists.update_yaxes(autorange = "reversed")
                fig_top15artists.update_xaxes(showgrid = False)
                fig_top15artists.update_yaxes(showgrid = False)
                st.plotly_chart(fig_top15artists, theme = "streamlit")
                st.write("Ce graphique révèle une grande diversité parmi les artistes les plus populaires, s'agissant des genres musicaux qui leur correspondent, ou encore de leur pays d'origine (Espagne, Argentine, Écosse, Angleterre, Chili, États-Unis...).")
                st.write("En revanche, ces artistes ont seulement 2 morceaux en moyenne dans notre jeu de données.")
                st.markdown("**Les artistes les plus populaires ne semblent pas être beaucoup représentés dans notre dataset ; nous avons donc souhaité savoir quel était le niveau de popularité des artistes les plus présents dans notre jeu de données.**")
                st.write("Dans la mesure où nos recommandations vont s'appuyer sur la similarité des contenus et la popularité, il est important que nous ayons une bonne compréhension de la composition de notre jeu de données à cet égard.")
                st.write("")
                with st.expander("Voir le code", expanded = False):
                    code = '''
                    fig_top15artists = px.area(top15artists, 
                    x = "Popularité moyenne",
                    y = "Artistes", 
                    color = "Genre", 
                    line_group = "Nombre total de morceaux")
                    fig_top15artists.update_yaxes(autorange = "reversed")
                    fig_top15artists.update_xaxes(showgrid = False)
                    fig_top15artists.update_yaxes(showgrid = False)
                    st.plotly_chart(fig_top15artists, theme = "streamlit")'''
                    st.code(code, language = 'python')

            with tab222:
                st.markdown("#### Top 10 des artistes les plus présents dans notre dataset")
                fig_top10artists = px.treemap(top10artists,
                                            path = [px.Constant("Top 10 des artistes ayant le plus de morceaux dans le jeu de données"), 'Artistes', "Nombre total de morceaux", "Genres musicaux"], values = 'Nombre total de morceaux',
                                            color = "Popularité moyenne", hover_data = ["Popularité moyenne"], 
                                            color_continuous_scale = 'RdBu',
                                            color_continuous_midpoint = np.average(top10artists['Popularité moyenne'], 
                                                                                    weights = top10artists['Popularité moyenne']))
                st.plotly_chart(fig_top10artists, theme = "streamlit")
                st.write("Nous constatons ici une tendance complètement contraire à celle observée avec le précédent graphique : les artistes ayant le plus de morceaux dans notre dataset (279 au total pour les Beatles par exemple) ont pour la plupart des notes de popularité très basses : 0,5 en moyenne pour Ella Fitzgerald, 1 pour Stevie Wonder.")
                st.write("Les genres musicaux auxquels correspondent ces artistes semblent également être plus spécifiques et un peu moins grand public que la pop, le rock et encore le hip-hop : psych-rock, grunge, honky-tonk, indian, goth...")
                st.markdown("**Nous ne savons pas comment a été constitué notre dataset principal mais cette visualisation confirme notre hypothèse selon laquelle les artistes les plus populaires, sur lesquelles nous allons faire reposer nos recommandations en priorité, semblent être sous-représentés dans notre jeu de données.**")
                st.write("Cela risque de limiter davatange nos possibilités de recommandations.")
                st.write("")
                with st.expander("Voir le code", expanded = False):
                    code = '''fig_top10artists = px.treemap(top10artists,
                                            path = [px.Constant("Top 10 des artistes ayant le plus de morceaux dans le jeu de données"), 'Artistes', "Nombre total de morceaux", "Genres musicaux"], values = 'Nombre total de morceaux',
                                            color = "Popularité moyenne", hover_data = ["Popularité moyenne"], 
                                            color_continuous_scale = 'RdBu',
                                            color_continuous_midpoint = np.average(top10artists['Popularité moyenne'], 
                                                                                    weights = top10artists['Popularité moyenne']))
                st.plotly_chart(fig_top10artists, theme = "streamlit")'''
                    st.code(code, language = 'python')

        with tab2:
            st.markdown("#### Top 15 des morceaux les plus populaires")
            fig_top15titres = px.area(top15titres,
                                      x = "Popularité", 
                                      y = "Morceaux",
                                      color = "Genres musicaux")
            fig_top15titres.update_yaxes(autorange = "reversed")
            fig_top15titres.update_xaxes(showgrid = False)
            fig_top15titres.update_yaxes(showgrid = False)
            st.plotly_chart(fig_top15titres, theme = "streamlit")
            st.write("Près de la moitié des titres de ce classement correspondent au genre reggaeton. Or dans notre dataset, ces morceaux sont classés sous d'autres genres : latino, reggae et latin.")
            st.markdown("**Les valeurs de notre dataset liées au genre musical ne semblent donc pas être totalement fiables.**")
            st.write("")
            with st.expander("Voir le code", expanded = False):
                code = '''
                fig_top15titres = px.area(top15titres,
                x = "Popularité", 
                y = "Morceaux",
                color = "Genres musicaux")
                fig_top15titres.update_yaxes(autorange = "reversed")
                fig_top15titres.update_xaxes(showgrid = False)
                fig_top15titres.update_yaxes(showgrid = False)
                st.plotly_chart(fig_top15titres, theme = "streamlit")'''
                st.code(code, language = 'python')

        with tab3:
            tab11, tab22, tab33, tab44 = st.tabs(["Top 15", "Popularité des genres", "Top 5 des morceaux par genre", "Répartition des valeurs après regroupement"])
            with tab11:
                st.markdown("#### Top 15 des genres musicaux les plus populaires")
                fig_top15genres = px.area(top15genres,
                                      x = "Popularité moyenne", 
                                      y = "Genres musicaux",
                                      color = "Genres musicaux")
                fig_top15genres.update(layout_showlegend=False)
                fig_top15genres.update_xaxes(showgrid = False)
                fig_top15genres.update_yaxes(showgrid = False)
                st.plotly_chart(fig_top15genres, theme = "streamlit")
                st.write("Seul un genre musical parmi ceux figurant dans les autres classements de popularité figure ici : la pop.")
                st.markdown("**Il semble donc y avoir une disparité entre les genres les plus populaires et les morceaux/artistes les plus populaires.**")
                st.write("")
                with st.expander("Voir le code", expanded = False):
                    code = '''
                    fig_top15genres = px.area(top15genres,
                    x = "Popularité moyenne",
                    y = "Genres musicaux",
                    color = "Genres musicaux")
                    fig_top15genres.update(layout_showlegend=False)
                    fig_top15genres.update_xaxes(showgrid = False)
                    fig_top15genres.update_yaxes(showgrid = False)
                    st.plotly_chart(fig_top15genres, theme = "streamlit")'''
                    st.code(code, language = 'python')
                
            with tab22:
                st.markdown("#### Popularité moyenne et nombre de morceaux par genre")
                ratio = pd.DataFrame(dataset_sample.groupby('track_genre')['track_name'].nunique())
                ratio = ratio.reset_index()
                ratio = ratio.rename(columns = ({'track_genre' : 'Genres musicaux', 'track_name' : 'Nombre de morceaux'}))
                ratio['Genres musicaux'] = ratio['Genres musicaux'].str.capitalize()
                ratio = ratio.drop(102, axis = 0)
                ratio = ratio.reset_index()
                ratio = ratio.drop(columns = 'index', axis = 1)
                ratio['Popularité moyenne'] = pop_genre['Popularité moyenne']
                fig_pop_genre = px.scatter(ratio,
                                           x = "Genres musicaux",
                                           y = "Nombre de morceaux",
                                           size = "Popularité moyenne",
                                           color = "Popularité moyenne",
                                           hover_name = "Genres musicaux",
                                           color_continuous_scale = "reds")
                fig_pop_genre.update_xaxes(showgrid = False)
                fig_pop_genre.update_yaxes(showgrid = False)
                st.plotly_chart(fig_pop_genre, theme = "streamlit", use_container_width = True)
                st.write("")
                st.write("La plupart des genres musicaux ayant les popularités moyennes les plus élevées sont également les genres les plus représentés dans notre dataset.")
                st.write("")
                with st.expander("Voir le code", expanded = False):
                    code = '''
                    fig = px.scatter(
                        pop_genre,
                        x = "Genres musicaux",
                        y = "Popularité moyenne",
                        color = "Popularité moyenne",
                        color_continuous_scale = "reds",
                    )
                    fig.update_xaxes(showgrid = False)
                    fig.update_yaxes(showgrid = False)
                    st.plotly_chart(fig, theme = "streamlit", use_container_width = True)'''
                    st.code(code, language = 'python')
                
            with tab33:
                st.markdown("#### Top 5 des morceaux les plus populaires par genre")
                st.write("")
                dataset_sample['artists'] = dataset_sample['artists'].str.title()
                dataset_sample['track_name'] = dataset_sample['track_name'].str.capitalize()
                dataset_sample['titre'] = dataset_sample['track_name'] + ' - ' + dataset_sample['artists']
                genres = sorted(set(dataset_sample['track_genre'].unique()))
                choix = st.selectbox("Sélectionner un genre musical", genres)
                st.write("")
                fig, ax = plt.subplots(figsize=(12,8))
                sns.barplot(x = 'popularity', y = 'titre', data = dataset_sample[dataset_sample['track_genre'] == choix].sort_values(by = 'popularity', ascending = False).iloc[:5], ax = ax, palette = 'coolwarm')
                for i in ax.containers:
                    ax.bar_label(i,)
                plt.title('Classement des 5 titres ayant la popularité la plus élevée dans le genre \''+choix+'\'', pad = 20, fontsize = 14)
                plt.xlabel('Popularité', labelpad = 20, fontsize = 12)
                plt.ylabel('Chanson', labelpad = 20, fontsize = 12)
                boxplot_chart = st.pyplot(fig)    

            with tab44:
                st.markdown("#### Répartition des valeurs après regroupement des genres")
                st.write("")
                st.write("Si l'on regroupe les genres musicaux par grandes familles de genres, on obtient une distribution plus homogène des valeurs de popularité.")
                st.write("")
                st.image("repart_genres.png", output_format = "PNG")
                st.write("")
                with st.expander("Voir le code", expanded = False):
                    code = '''
                    sns.catplot(x = 'track_genre', y = 'popularity', kind ='boxen', data = genres, height = 15, aspect = 2, palette = 'coolwarm')
                    plt.xticks(rotation = 60)
                    plt.title('Répartition des valeurs de popularité par genre musical \n (après regroupement des genres)', pad = 20, fontsize = 14)
                    plt.xlabel('Genres musicaux', labelpad = 20, fontsize = 12)
                    plt.ylabel('Popularité globale', labelpad = 20, fontsize = 12);'''
                    st.code(code, language = 'python')
                st.write("")
                st.divider()
                st.write("")
                st.write("S'agissant du nombre de morceaux par famille de genres en revanche, le regroupement a pour effet une sur-représentation de certains genres comme les musiques du monde, ou le genre 'Divers'.")
                st.write("Le graphique ci-dessous nous permet également de constater que certains genres parmi les plus populaires, comme le hip-hop ou la pop, restent globalement sous-représentés.")
                fig_num_genres = px.scatter(numgenres,
                                           x = "Genres musicaux",
                                           y = "Nombre de morceaux",
                                           size = "Popularité moyenne",
                                           color = "Popularité moyenne",
                                           hover_name = "Genres musicaux",
                                           color_continuous_scale = "reds")
                fig_num_genres.update_xaxes(showgrid = False)
                fig_num_genres.update_yaxes(showgrid = False)
                st.plotly_chart(fig_num_genres, theme = "streamlit", use_container_width = True)
                st.write("")
                with st.expander("Voir le code", expanded = False):
                    code = '''
                    fig_num_genres = px.scatter(numgenres,
                                           x = "Genres musicaux",
                                           y = "Nombre de morceaux",
                                           size = "Popularité moyenne",
                                           color = "Popularité moyenne",
                                           hover_name = "Genres musicaux",
                                            color_continuous_scale = "reds")
                    fig_num_genres.update_xaxes(showgrid = False)
                    fig_num_genres.update_yaxes(showgrid = False)
                    st.plotly_chart(fig_num_genres, theme = "streamlit", use_container_width = True)'''
                    st.code(code, language = 'python')
            
    with tab_correlations:
        st.markdown("#### Corrélations des caractéristiques techniques")
        st.write("")
        st.image("correlation.png")
        st.write("")
        st.write("Cette data visualisation ne révèle aucune corrélation particulière entre les variables, à l’exception des variables “loudness” et “energy”, pour lesquelles le niveau de corrélation est estimé à 0.76. Ces deux variables ne semblent cependant pas avoir d’impact particulier sur la popularité.")
        st.write("")
        with st.expander("Voir le code", expanded = False):
            code = '''cor = dataset.corr()
            plt.figure(figsize = (11,11))
            sns.heatmap(cor, center = 0, annot = True, cmap = 'coolwarm')
            plt.title("Corrélations entre les principales caractéristiques techniques des morceaux", pad = 20, fontsize = 14);'''
            st.code(code, language = 'python')

    with tab_linearite:
        st.markdown("#### Linéarité entre les caractéristiques techniques")
        st.write("")
        st.image("linearite.png")
        st.write("")
        st.write("Cette data visualisation ne nous donne pas d’indications très précises, et confirme plutôt notre hypothèse évoquée juste avant. Les morceaux les plus populaires correspondent plutôt aux morceaux les plus dansants, mais la popularité, bonne ou mauvaise, est globalement diffuse à travers toutes les valeurs possibles de dansabilité.")
        st.write("Il en est de même pour l’énergie dégagée par le morceau et le tempo : quel que soit le degré d’énergie véhiculée par la musique et la vitesse du morceau, la popularité peut être aussi bonne que mauvaise. À noter : les vitesses proches de 0 correspondent à des morceaux exclusivement parlés (podcasts, récits d’histoire…).")
        st.write("S’agissant du mode, ici encore, que les morceaux soient plutôt tristes (mode mineur) ou joyeux (mode majeur), la popularité peut être aussi bonne que mauvaise.")
        st.write("Enfin seul le caractère “loudness” semble avoir un impact plus marqué sur la popularité, puisque les morceaux les plus populaires sont plutôt les morceaux les plus intenses. Nous ne savons pas néanmoins comment a été calculée cette variable, et un même niveau d’intensité (de loudness) peut correspondre à beaucoup de genres musicaux différents.")
        st.write("")
        with st.expander("Voir le code", expanded = False):
            code = '''plt.figure(figsize = (12,12))
            sns.pairplot(dataset[['popularity', 'danceability', 'energy', 'loudness', 'mode', 'tempo']], diag_kind='kde', palette = 'coolwarm');'''
            st.code(code, language = 'python')         
    
# Page 4 - Modélisation
if page == pages[4]: 
    st.header("Modélisation")
    st.markdown("Notre problème s’apparente à un problème de machine learning de type :blue[clustering]. Il nous a effectivement semblé que le regroupement de morceaux similaires en clusters serait une approche pertinente pour atteindre notre objectif. Notre objectif était ensuite de classer et trier les morceaux selon leur popularité, pour proposer aux utilisateurs les recommandations les plus intéressantes possibles.")
    st.markdown("###")
    
    tab1, tab2, tab3 = st.tabs(["Réduction de dimension", "Choix du modèle et optimisation", "Mise en place de l'algorithme"])
    with tab1:
        st.markdown("#### Réduction de dimension")
        st.write("Pour ce type de problèmes, les méthodes de réduction de dimension telles que les méthodes d’analyse de variance sont préférées à la mesure de la dépendance entre la variable cible et les features, plutôt utilisée pour les problèmes supervisés.")
        st.write("Notre choix de méthode de réduction de dimension s’est donc orienté donc vers une méthode de *feature selection*, grâce à laquelle nous espérions pouvoir déterminer un sous-ensemble de features optimales, qui nous serviraient ensuite pour notre algorithme de machine learning.")
        st.markdown("Nous avons utilisé une méthode :blue[ACP (Analyse en Composantes Principales)] sur trois versions différentes de notre dataset : \n- une version avec les 125 genres musicaux ; \n- une version les 125 genres regroupés en 24 grandes familles de genres ; \n- une dernière version sans aucun genre musical.")
        st.markdown("###")
        
        tab_complet, tab_avec, tab_no = st.tabs(["Avec le dataset complet", "Avec regroupement des genres musicaux", "Sans les genres musicaux"])
        with tab_complet:
            tab_var, tab_sum = st.tabs(["Variance expliquée", "Somme cumulative"])
            with tab_var:
                st.write("")
                st.image("varexpsans.png")
                st.write("")
                st.write("La courbe d'évolution de la variance expliquée décroît fortement puis se stabilise autour de 0.0075 pour un nombre de composantes se situant entre 15 et 17. La part de variance expliquée est donc plus importante sur les premières composantes principales puis décroît rapidement.")
            with tab_sum:
                st.write("")
                st.image("cumsumsans.png")
                st.write("")
                st.write("Nous avons ajusté notre ACP sur les données pour conserver 90% de la variance expliquée, et le nombre de composantes retenues a finalement été de 109. C’est une réduction de 15% puisque nous disposions au départ de 129 variables, mais nous avons estimé que le résultat n’était pas à la hauteur de nos attentes.")
        with tab_avec:
            tab_var, tab_sum = st.tabs(["Variance expliquée", "Somme cumulative"])
            with tab_var:
                st.write("")
                st.image("varexpavec.png")
                st.write("")
            with tab_sum:
                st.write("")
                st.image("cumsumavec.png")
                st.write("")
                st.write("La somme cumulative de la variance expliquée sur le graphique ci-dessus nous indique qu'il faut ici 36 Composantes Principales pour obtenir 90% de variance expliquée.")
                st.write("C’est une réduction de 28% puisque nous disposions avec cette version “réduite” du dataset, de 46 variables au départ. Ce résultat est meilleur, mais nous espérions pouvoir réduire davantage le nombre de composantes.")
        with tab_no:
            tab_var, tab_sum = st.tabs(["Variance expliquée", "Somme cumulative"])
            with tab_var:
                st.write("")
                st.image("varexpno.png")
                st.write("")
                st.write("On observe ici que les tendances d’évolution de la variance sont globalement identiques avec cette nouvelle version du dataset ; le nombre optimal de composantes quant à lui semble être égal à 11.")
            with tab_sum:
                st.write("")
                st.image("cumsumno.png")
                st.write("")
                st.write("Après avoir ajusté les données pour conserver 90% de la variance expliquée, nous obtenons un nombre de composantes retenues de 12.")

    with tab2:
        st.markdown("#### Choix du modèle et optimisation")
        st.markdown("Dans la mesure où nous ne disposions pas d’une variable cible propre, nous nous sommes orientés vers un algorithme de clustering non supervisé : :blue[l’algorithme des K-moyennes (K-means)].")
        st.write("")
        tab11, tab22 = st.tabs(["K-means avec genres musicaux regroupés", "K-means sans genres musicaux"])
        with tab11:
            st.write("")
            st.image("kmeans.png")
            st.write("")
            st.write("On remarque que l’inertie stagne à partir de 31 clusters. Nous appliquons alors l’algorithme avec un nombre de clusters égal à 31.")
            st.write("")
            st.image("heatmapK1.png")
            st.image("heatmapK2.png")
            st.write("")
            st.write("Malheureusement, il semble qu’à chaque cluster correspond peu ou prou un genre musical spécifique. Cela n’est pas forcément étonnant dans la mesure où nous avions réduit le nombre de genres musicaux à 25 ; cette valeur est assez proche de notre nombre total de clusters (31).")
            st.write("Il semble donc que notre algorithme se soit basé principalement sur les genres pour constituer ses clusters. C’est une approche intéressante, mais elle risque de ne pas nous apporter beaucoup d’informations complémentaires pour notre travail de prédiction.")
            st.write("Nous avons donc testé le clustering par K-means sur notre jeu de données sans genres musicaux.")
            
        with tab22:
            tab_elbow, tab_k_elbow, tab_silhouette, tab_clusters = st.tabs(["Méthode du coude", "KElbowVisualizer", "Silhouette score", "Clustering"])
            with tab_elbow:
                st.image("elbow.png")
                st.write("")
                st.write("On remarque ici que l’inertie évolue de façon plutôt linéaire et ce graphique ne nous permet pas de déterminer clairement le nombre optimal de clusters.")
                st.write("Nous nous sommes donc appuyés sur une autre représentation de la méthode du coude via un “KElbowVisualizer”, à l’aide duquel le nombre optimal de clusters peut être estimé automatiquement")
            with tab_k_elbow:
                st.image("kelbow.png")
                st.write("")
                st.write("Cette deuxième représentation, basée non pas sur l’inertie mais sur la distorsion, indique un nombre optimal de clusters égal à 8. Cette estimation se base sur le score obtenu par les différents nombres de clusters testés, ainsi que sur le degré d’inflexion du coude. ")
                st.write("Cela nous oriente vers un premier résultat possible, mais nous avons préféré poursuivre notre étude à l’aide d’une analyse de silhouette pour déterminer de façon la plus certaine possible le nombre de clusters optimal.")
            with tab_silhouette:
                st.write("Nous avons donc réalisé une première représentation graphique du silhouette_score pour un nombre de clusters compris entre 6 et 16 :")
                st.write("")
                st.image("silhouette.png")
                st.write("")
                st.write("Les silhouette scores sont globalement plutôt proches de la valeur 0, ce qui n’est pas un bon signe pour la robustesse et l’efficacité de notre algorithme.")
                st.write("On remarque également que le nombre de clusters qui obtient le meilleur silhouette_score est 7. Cette valeur est proche de l’estimation obtenue avec le “KElbowVisualizer”.")
                st.divider()
                st.write("Pour pouvoir déterminer de manière sûre le nombre optimal de clusters, nous avons finalement représenté l’analyse de silhouette pour un nombre de clusters allant de 6 à 10, puisque le nombre idéal de clusters semble se situer entre 7 et 8.")
                st.write("Sur ces graphiques, les clusters sont représentés les uns par rapport aux autres, et mis en perspective au regard de leur score respectif.")
                st.write("")
                tab_6, tab_7, tab_8, tab_9, tab_10 = st.tabs(["6 clusters", "7 clusters", "8 clusters", "9 clusters", "10 clusters"])
                with tab_6:
                    st.image("6.png")
                with tab_7:
                    st.image("7.png")
                with tab_8:
                    st.image("8.png")
                with tab_9:
                    st.image("9.png")
                with tab_10:
                    st.image("10.png")
                st.write("")
                st.write("On peut voir sur les représentations correspondant aux valeurs n_clusters = 8, 9 et 10 que certains clusters sont beaucoup plus grands que d’autres. Nous avons donc orienté notre choix sur un K-means à 6 ou 7 clusters.") 
                st.write("La représentation pour n_clusters = 7 montre que les clusters 2, 3 et 4 sont globalement plus grands que les 4 autres clusters formés. Pour autant, ceux-ci ont tous une forme plus homogène que les clusters constitués pour n_clusters = 6.") 
                st.write("Nous avons donc choisi un nombre de clusters égal à 7, ce qui confirme notre hypothèse précédente basée sur le meilleur silhouette_score.")
            with tab_clusters:
                st.write("Comparons la représentation ci-dessous avec celle réalisée précédemment, et observons les caractéristiques sur lesquels semble s’être appuyé l’algorithme pour effectuer ses regroupements : ")
                st.write("")
                st.image("heatmapclusters.png")   
                st.write("")
                st.write("Il semble que les clusters aient été constitués ici au regard des attributs techniques des morceaux, contrairement à notre première tentative qui avait abouti à une classification par genre musical.") 
                st.write("Même si cela semble logique au vu des variables de notre dataset retravaillé, c’est précisément cette approche vers laquelle nous souhaitions tendre.")

    with tab3:
        st.markdown("#### Mise en place de l'algorithme")
        st.write("La structuration de notre jeu de données initial n’a pas permis une grande réduction de dimensionnement, et après une première analyse des clusters formés sur la base des caractéristiques techniques, les regroupements se sont avérés intéressants mais pas forcément exploitables en l'état." )
        st.write("Nous avons donc optimisé notre algorithme par différents niveaux de filtrage, et nous avons réussi à obtenir une fonction qui propose a priori des recommandations pertinentes dans différents genres musicaux, pour des morceaux aussi bien contemporains que plus anciens.")
        st.write("")
        
        tabAlgo1, tabAlgo2 = st.tabs(["Algorithme principal", "Matrice de similarité"])
        with tabAlgo1:
            tabET1, tabET2, tabET3, tabET4, tabET5, tabET6 = st.tabs(["Étape 1", "Étape 2", "Etape 3", "Étape 4", "Étape 5", "Étape 6"])
            with tabET1:
                st.write(f"**Étape 1 : filtrage selon le cluster Attributs du morceau initial**")
                graph1 = graphviz.Digraph()
                graph1.edge('Morceau', "Cluster 'Attributs'", color = "red")
                st.graphviz_chart(graph1, use_container_width = True)
            
            with tabET2:
                st.write(f"**Étape 2 : filtrage selon plusieurs caractéristiques techniques**")
                graph2 = graphviz.Digraph()
                graph2.edge('Morceau', "Cluster 'Attributs'")
                graph2.edge("Cluster 'Attributs'", 'Tempo', color = "red")
                graph2.edge("Cluster 'Attributs'", 'Loudness', color = "red")
                graph2.edge("Cluster 'Attributs'", 'Energy', color = "red")
                graph2.edge("Cluster 'Attributs'", 'Acousticness', color = "red")
                graph2.edge("Cluster 'Attributs'", 'Danceability', color = "red")
                st.graphviz_chart(graph2, use_container_width = True)
                st.write("")
                st.write("Nous avons stocké les résultats de ce premier filtrage dans un dataframe.")

            with tabET3:
                st.write(f"**Étape 3 : 2ème filtrage selon le/les genres identiques à celui/ceux du morceau initial**")
                graph3 = graphviz.Digraph()
                graph3.edge('Morceau', "Cluster 'Attributs'")
                graph3.edge("Cluster 'Attributs'", 'Tempo')
                graph3.edge("Cluster 'Attributs'", 'Loudness')
                graph3.edge("Cluster 'Attributs'", 'Energy')
                graph3.edge("Cluster 'Attributs'", 'Acousticness')
                graph3.edge("Cluster 'Attributs'", 'Danceability')
                graph3.edge('Morceau', 'Genres similaires', color = "red")
                st.graphviz_chart(graph3, use_container_width = True)
                st.write("")
                st.write("Nous avons stocké les résultats de ce premier filtrage dans un dataframe.")

            with tabET4:
                st.write(f"**Étape 4 : filtrage selon plusieurs caractéristiques techniques**")
                graph4 = graphviz.Digraph()
                graph4.edge('Morceau', "Cluster 'Attributs'")
                graph4.edge("Cluster 'Attributs'", 'Tempo')
                graph4.edge("Cluster 'Attributs'", 'Loudness')
                graph4.edge("Cluster 'Attributs'", 'Energy')
                graph4.edge("Cluster 'Attributs'", 'Acousticness')
                graph4.edge("Cluster 'Attributs'", 'Danceability')
                graph4.edge('Morceau', 'Genres similaires')
                graph4.edge('Genres similaires', 'Tempo', color = 'red')
                graph4.edge('Genres similaires', 'Loudness', color = 'red')
                graph4.edge('Genres similaires', 'Energy', color = 'red')
                graph4.edge('Genres similaires', 'Acousticness', color = 'red')
                graph4.edge('Genres similaires', 'Danceability', color = 'red')
                st.graphviz_chart(graph4, use_container_width = True)
                st.write("")
                st.write("Nous avons également stocké les résultats de ce deuxième filtrage dans un autre dataframe.")

            with tabET5:
                st.write(f"**Étape 5 : Regroupement des autres morceaux du groupe/de l'interprète**")
                graph5 = graphviz.Digraph()
                graph5.edge('Morceau', "Cluster 'Attributs'")
                graph5.edge("Cluster 'Attributs'", 'Tempo')
                graph5.edge("Cluster 'Attributs'", 'Loudness')
                graph5.edge("Cluster 'Attributs'", 'Energy')
                graph5.edge("Cluster 'Attributs'", 'Acousticness')
                graph5.edge("Cluster 'Attributs'", 'Danceability')
                graph5.edge('Morceau', 'Genres similaires')
                graph5.edge('Genres similaires', 'Tempo')
                graph5.edge('Genres similaires', 'Loudness')
                graph5.edge('Genres similaires', 'Energy')
                graph5.edge('Genres similaires', 'Acousticness')
                graph5.edge('Genres similaires', 'Danceability')
                graph5.edge('Morceau', "Autres morceaux du groupe", color = 'red')
                st.graphviz_chart(graph5, use_container_width = True)
                st.write("")
                st.write("Si d'autres morceaux du groupe ou de l'interprète figuraient dans notre dataset initial, nous les avons stockés dans un troisième jeu de données.")

            with tabET6:
                st.write(f"**Étape 6 : définition des priorités dans la remontée des résultats**")
                st.write("")
                tab61, tab62, tab63 = st.tabs(["Priorité 1", "Priorité 2", "Priorité 3"])                    
                with tab61:
                    st.write(f"**Priorité 1 : les autres morceaux du groupe/de l'artiste**")
                    graph11 = graphviz.Digraph()
                    graph11.edge("Filtrages", "Autres morceaux du groupe", color = "red")
                    st.graphviz_chart(graph11, use_container_width = True)
                
                with tab62:
                    st.write(f"**Priorité 2 : les morceaux issus du/des mêmes genres musicaux**")
                    graph22 = graphviz.Digraph()
                    graph22.edge("Filtrages", "Autres morceaux du groupe")
                    graph22.edge("Autres morceaux du groupe", "Genres similaires", label = "Si aucun résultat", color = "red")
                    st.graphviz_chart(graph22, use_container_width = True)
                        
                with tab63:
                    st.write(f"**Priorité 3 : les morceaux issus du cluster 'Attributs'**")
                    graph33 = graphviz.Digraph()
                    graph33.edge("Filtrages", "Autres morceaux du groupe")
                    graph33.edge("Autres morceaux du groupe", "Genres similaires")
                    graph33.edge("Genres similaires", "Cluster 'Attributs'", label = "Si aucun résultat", color = "red")
                    st.graphviz_chart(graph33, use_container_width = True)

        with tabAlgo2:
            st.write("**Mise en place de la matrice de similarité**")
            st.write("")
            st.write("- Construction de la matrice à l'aide des fonctions :blue[pdist] et :blue[squareform] du sous-package scipy.spatial.distance ; \n- Sélection d'une métrique différente de la métrique par défaut (distance euclidienne) : :blue[distance de Mahalanobis] ; \n- Pas de retraitement ou de filtrage additionnel.")
            st.markdown("##")
            st.image("matrice.png", caption = 'Extrait de la matrice de similarité')

# Page 5 - Bilan
if page == pages[5]: 
    st.header("Bilan")

    tab1, tab2, tab3 = st.tabs(['Revue des objectifs initiaux', "Difficultés rencontrées lors du projet", "Pistes d'amélioration"])

    with tab1:
        st.markdown("#### Revue des objectifs initiaux")
        col1, col2 = st.columns([4,1])
        with col1:
            st.write("Notre objectif était de réaliser une recommandation pertinente de musique à un utilisateur, sur la base d’un morceau que l’utilisateur avait préalablement choisi ou écouté.") 
            st.markdown("Nous pouvons fièrement dire que :blue[nous avons atteint cet objectif], et nous l'avons fait de deux manières différentes et complémentaires, que nous allons mettre à l'épreuve dans quelques minutes.")
        with col2:
            st.image(flimage, width = None)
    
    st.markdown('##')     
    
    with tab2:
        st.markdown("#### Difficultés rencontrées lors du projet")
        st.write("Nous avons fait face à plusieurs difficultés techniques, liées notamment : \n- à la volumétrie de certains jeux de données ; \n- à la structure de certains jeux de données ; \n- aux limites de mémoire vive nécessaire à l’utilisation de nos notebooks ; \n- à l’importation de certains modules et packages.")
        st.markdown("Plus globalement, nous disposions au départ d'un volume très important de données, dont seulement une partie nous a été réellement utile. :blue[Nous aurions pu aller beaucoup plus loin dans la mise en place de nos modèles si nous avions disposé de davantages de morceaux différents.]")
        st.write("Le filtrage collaboratif par exemple, est une approche que nous aurions pu comparer aux deux autres méthodes utilisées, mais par manque de données, notre algorithme retournait systématiquement les mêmes résultats. Cette approche n'était donc pas pertinente dans notre contexte.")
    
    st.markdown('##')
    
    with tab3:
        st.markdown("#### Pistes d’amélioration")
        st.markdown("Si nous avions eu plus de temps, nous aurions pu : \n- :blue[agrémenter nos données avec d’autres datasets complémentaires], pour rendre nos algorithmes encore plus performants, et mettre en place un troisième modèle basé sur le filtrage collaboratif ;")
        st.markdown("- :blue[mettre en place un système de notation des recommandations effectuées] pour évaluer quel modèle parmi les deux retenus semble être le plus pertinent selon les utilisateurs, avec le stockage des notes dans une base de donnée.")

# Page 6 - Recommandations musicales
if page == pages[6]: 
    st.header("Recommandations musicales :man_dancing:")
    st.markdown('##')
    st.markdown("##### Entrez le nom de l'artiste ou du morceau que vous souhaitez écouter : ")    
    
    # Fonction de recommandation basée sur les clusters 
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

    # Matrice de similarité
    new_index = list(dfm.index)
    pairwise = pd.DataFrame(squareform(pdist(dfm, 'mahalanobis')))
    pairwise['artist_track'] = new_index
 
    # Fonction de recommandation basée sur la matrice de similarité 
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

    # Lecteur Deezer    
    def player(reco):
        url = "https://api.deezer.com/search?q=" + reco
        request = requests.get(url)
        parsing = json.loads(request.text)
        reco_id = str(parsing['data'][0]['id'])
        link = "https://widget.deezer.com/widget/auto/track/" + reco_id
        return components.html(f'<iframe title="deezer-widget" src={link} width="100%" height="150" frameborder="0" allowtransparency="true" allow="encrypted-media; clipboard-write"></iframe>')

    # Définition du morceau de départ
    OG = "Nirvana - Smells like teen spirit"
    OG_index = int(dfm2[dfm2['artist_track'] == OG].index.values)
    options = dfm2['artist_track']
    search_bar = st.selectbox("", options, index = OG_index, label_visibility = "collapsed", key = "search")
    player(search_bar)
    
    # Fonctions de mise à jour de la barre de recherche
    def callback1():
        st.session_state.search = options[index_reco1]
        
    def callback2():
        st.session_state.search = options[index_reco2]

    # Gestion et affichage des recommandations 
    if search_bar != OG:
        st.markdown('#')
        st.write(f"Vous avez choisi le titre **{search_bar}**, nous vous proposons les recommandations suivantes :")
        track = str(search_bar) 
        reco1 = reco(track)[0]
        reco2 = recom(track)[0]
        index_reco1 = int(dfm2[dfm2['artist_track'] == reco1].index.values)
        index_reco2 = int(dfm2[dfm2['artist_track'] == reco2].index.values)
        col1, col2 = st.columns([1,1])
        with col1:
            st.write(f"**{reco1}**")
            player(reco1)
            bouton1 = st.button('Recommandation suivante', on_click = callback1, key = 1)
        with col2:
            st.write(f"**{reco2}**")
            player(reco2)
            bouton2 = st.button('Recommandation suivante', on_click = callback2, key = 2)  

    
