import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report


def GetX(df):
    X = df[['Genre','Rating','Platform','Global_Sales','User_Score']]
    X = pd.get_dummies(data=X, drop_first=True)
    return X


def GetModel(df):
    X = GetX(df)
    y = df['Critic_Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model,X


def CleanData(df):
    df = df.dropna(how='any')
    df[['Year_of_Release','Critic_Score','Critic_Count', 'User_Count']] = \
    df[['Year_of_Release','Critic_Score','Critic_Count', 'User_Count']].astype(int)

    df.reset_index().drop('index',axis=1)
    
    return df


def GetGameInfo(df):
    name=df['Name'].sort_values().unique().tolist()
    game_name=st.text_input('Please Enter the Name of the Game You Want to Analyze:')

    genre = df['Genre'].sort_values().unique().tolist()
    genre_choice = st.selectbox('Genre:', [' ']+genre)
    
    platform = df['Platform'].sort_values(ascending=False).unique().tolist()
    platform_choice = st.selectbox('Platform:', [' ']+platform)

    ratings_dict = {'E':'E (Everyone)','E10+':'E10+ (Everyone 10+)','AO':'AO (Adults Only 18+)',\
                    'K-A': 'K-A (Kids to Adults)', 'M':'M (Mature 17+)','RP':'RP (Rating Pending','T': 'T (Teen)'}
    inverse_ratings_dict = {v:k for k,v in ratings_dict.items()}
    rating = df['Rating'].sort_values(ascending=True).map(ratings_dict).unique().tolist()
    rating_choice = st.selectbox('ESRB Rating:', [' ']+rating)
    
    if rating_choice!=' ':
        rating_choice = inverse_ratings_dict[rating_choice]

    global_sales = st.slider('Estimated Global Sales (In Million Copies):',0.0,100.0,50.0,0.01)
    
    user_score = st.slider('Self-Rated Score:',0.0,10.0,5.0,0.1)
    
    
    col1, col2, col3, col4,col5 = st.columns(5)
    
    get_rating_button = col3.button('Submit')
    
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
        
    return game_name, genre_choice, platform_choice, rating_choice, global_sales, user_score,get_rating_button


def FilterHistoricalData(df):

    st.sidebar.title('Historical Data Filter')
    
    num_result = st.sidebar.slider('Max Number of Result:',10,200,10,10 )
    
    genre = df['Genre'].sort_values().unique()
    genre_choice = st.sidebar.selectbox('Select Game Genre:', genre)
    
    
    year = df['Year_of_Release'].sort_values(ascending=False).unique()
    start_year, end_year = st.sidebar.slider('Select Year of Release:', int(min(year)), int(max(year)),(2000,2016))
    
    platform = df['Platform'].sort_values(ascending=False).unique()
    platform_choice = st.sidebar.selectbox('Select Platform:', platform)


   # OLD DATAFRAME VISUALIZATION

    st.header('Filtered Historical Data')
    df = df[df.Year_of_Release.isin(list(range(start_year,end_year+1)))]

    df = df.loc[(df['Genre']==genre_choice) & \
            (df['Platform']==platform_choice)].sort_values('Global_Sales',ascending=False)\
            .head(num_result).reset_index().drop('index',axis=1)

    if not df.empty:

        st.dataframe(df)
    else:
        st.write('No Result Found')


    # Altair
    st.header('Altair Chart')
    chart = alt.Chart(df).mark_circle().encode(
        x=alt.X('User_Score', scale=alt.Scale(zero=False)),
        y='Critic_Score',
        size='Global_Sales',
        color=alt.Color('Year_of_Release',
                        scale=alt.Scale(scheme='turbo')),
        tooltip=['Name', 'Critic_Count', "User_Count"]
    ).properties(width=800,
                 height=400)

    st.altair_chart(chart)


   # Pie Chart
    pie1 = px.pie(df, values=df['Global_Sales'][:10],
                  names=df['Name'][:10],
                  title="Top 10 games globally after filtering (shows the proportion of sales each game holds).",
                  color_discrete_sequence=px.colors.sequential.Purp_r)
    pie1.update_traces(textposition='inside', textinfo='percent+label', showlegend=False)


    st.header("Pie Chart")
    st.plotly_chart(pie1)


    # Bar Chart
    st.header('Bar Chart')
    name2 = pd.DataFrame(
        df.groupby("Name")[["NA_Sales"]].mean().sort_values(by=['NA_Sales'], ascending=[False]).reset_index())
    name2.rename(columns={'Name': 'Name_NA'}, inplace=True)

    name3 = pd.DataFrame(
        df.groupby("Name")[["EU_Sales"]].mean().sort_values(by=['EU_Sales'], ascending=[False]).reset_index())
    name3.rename(columns={'Name': 'Name_EU'}, inplace=True)

    name4 = pd.DataFrame(
        df.groupby("Name")[["JP_Sales"]].mean().sort_values(by=['JP_Sales'], ascending=[False]).reset_index())
    name4.rename(columns={'Name': 'Name_JP'}, inplace=True)

    name5 = pd.DataFrame(
        df.groupby("Name")[["Other_Sales"]].mean().sort_values(by=['Other_Sales'], ascending=[False]).reset_index())
    name5.rename(columns={'Name': 'Name_other'}, inplace=True)

    # Concatenating the results.
    name_df = pd.concat([name2, name3, name4, name5], axis=1)
    subplot_name1 = make_subplots(rows=4, cols=1, shared_yaxes=True, subplot_titles=(
    "North American top games", "Europe top games", "Japan top games", "Other regions top games", 'Top games globally'))

    # Subplot for North America
    subplot_name1.add_trace(go.Bar(x=name_df['Name_NA'][:5], y=name_df['NA_Sales'][:5],
                                   marker=dict(color=[1, 2, 3], coloraxis="coloraxis")), 1, 1)

    # Subplot for Europe
    subplot_name1.add_trace(go.Bar(x=name_df['Name_EU'][:5], y=name_df['EU_Sales'][:5],
                                   marker=dict(color=[4, 5, 6], coloraxis="coloraxis")), 2, 1)

    # Subplot for Japan
    subplot_name1.add_trace(go.Bar(x=name_df['Name_JP'][:5], y=name_df['JP_Sales'][:5],
                                   marker=dict(color=[7, 8, 9], coloraxis="coloraxis")), 3, 1)

    # Subplot for other regions
    subplot_name1.add_trace(go.Bar(x=name_df['Name_other'][:5], y=name_df['Other_Sales'][:5],
                                   marker=dict(color=[10, 11, 12], coloraxis="coloraxis")), 4, 1)

    subplot_name1.update_layout(height=1000, width=500, coloraxis=dict(colorscale='agsunset_r'), showlegend=False)
    subplot_name1.update_xaxes(tickangle=45)
    #subplot_name1.show()
    st.plotly_chart(subplot_name1)



    return df





def GetTestData(original_df, genre_choice, platform_choice, rating_choice, global_sales, user_score):

    original_df.at[0,'Genre']= genre_choice
    original_df.at[0,'Platform']= platform_choice
    original_df.at[0,'Rating']= rating_choice
    original_df.at[0,'Global_Sales']= global_sales*0.50
    original_df.at[0,'User_Score']= user_score * 0.50

    X = GetX(original_df).head(1)
    
    return X


def GetSuggestScore(X):

    print(X)

    predicted_result = model.predict(X).round(1)[0]

    print(predicted_result)

    return predicted_result


#Add Background Picture
st.markdown(
     """
     <style>
     .reportview-container {
         background: url("https://get.wallhere.com/photo/simple-simple-background-minimalism-video-game-characters-pink-light-pink-logo-Paper-Mario-Mario-Bros-Super-Mario-silhouette-1381511.jpg")
     }
    .sidebar .sidebar-content {
         background: url("https://get.wallhere.com/photo/simple-simple-background-minimalism-video-game-characters-pink-light-pink-logo-Paper-Mario-Mario-Bros-Super-Mario-silhouette-1381511.jpg")
     }
     </style>
     """,
     unsafe_allow_html=True
 )

# https://i.pinimg.com/originals/e1/ba/d3/e1bad340fb3afbc791939ad083b49dd5.jpg



st.title('Video Games Rating System')
st.write('Video Games Rating System is designed to provide gaming companies a recommended rating score\
        for reference through analyzing the Genre, Platform, ESRB Rating, Estimated Global Sales, \
        and the gaming companies’ Self-Rated Score with Machine Learning.')


df= pd.read_csv('video_game_sales.csv',index_col=False)

df = CleanData(df)

model,X = GetModel(df)

original_df = df.copy()



game_name, genre_choice, platform_choice, rating_choice, global_sales, user_score,get_rating_button = GetGameInfo(df)


# print(game_name, genre_choice, platform_choice, rating_choice, global_sales, user_score,get_rating_button)

X = GetTestData(original_df, genre_choice, platform_choice, rating_choice, global_sales, user_score)

# print(X)

if get_rating_button:

    try:
        if len(game_name)<2:
            st.warning('Please Enter the Game Name (minimum 2 characters)')
        else:

            predicted_result = GetSuggestScore(X)
            st.success('Rating Calculated')

            if predicted_result>100:
                predicted_result =100
            elif predicted_result<0:
                predicted_result = 0



            if predicted_result:


                t = st.empty()
                st.markdown("<style>.big-font {font-size:50px !important;}</style>", unsafe_allow_html=True)

                st.markdown(f'<p class="big-font">Suggested Rating For {game_name}: </p>', unsafe_allow_html=True)
                style = f'<p style="font-family:sans-serif; color:Green; font-size: 150px;">{predicted_result}/100</p>'
                st.markdown(style, unsafe_allow_html=True)

    except ValueError:
        st.warning('Please Complete All Fields Before Submitting')



st.write(' ')
st.write(' ')


st.write('This score is calculated based on Video Games Sales data from 1985 to 2016. Please use it as a\
         reference for your decisions making. You can view the Historical Data by utilziing the Filter (located at the LEFT SIDEBAR).\
         After filtering, you will see different interactive charts based on the applied filter.')

df = FilterHistoricalData(df)

st.write(' ')
st.write(' ')
st.write(' ')
st.write(' ')
st.write(' ')

st.header('References')
st.write("Source data: [Video Game Sales with Ratings | Kaggle](https://www.kaggle.com/rush4ratio/video-game-sales-with-ratings)")
st.write("Link to Github repository: [Github Repository](https://github.com/Jingyuan-Yang/FinalProject)")

