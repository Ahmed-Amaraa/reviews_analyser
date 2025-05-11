import streamlit as st

st.title("Goodreads Review Sentiment Analyzer")

goodreads_url = st.text_input("Enter Goodreads book URL:")
if goodreads_url:
    with st.spinner("Loading and analyzing Goodreads page..."):
        from process import preprocess
        import requests
        from bs4 import BeautifulSoup
        import pandas as pd
        import joblib
        import matplotlib.pyplot as plt

        model = joblib.load('sentiment_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        predicted_reviews = []
        req = requests.get(goodreads_url)
        src = req.content
        soup = BeautifulSoup(src, 'lxml')
        list_of_reviews = []
        reviews = soup.find_all("span", {'class': 'Formatted'})

        for review in reviews:
            for br in review.find_all("br"):
                br.replace_with("\n")
            text = review.get_text().strip()
            list_of_reviews.append(text)

        for review in list_of_reviews:
            preprocessed = preprocess(review)
            X_new = vectorizer.transform([preprocessed])
            prediction = model.predict(X_new)
            label_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
            if prediction[0] == 0:
                predicted_r = 'Negative'
                predicted_reviews.append(predicted_r)
            elif prediction[0] == 1:
                predicted_r = 'Positive'
                predicted_reviews.append(predicted_r)
            else:
                predicted_r = 'Neutral'
                predicted_reviews.append(predicted_r)
        show_analysis = st.button("Show Analysis")
    if show_analysis:
    
        sentiment_counts = pd.Series(predicted_reviews).value_counts()
        st.write(sentiment_counts)

        fig, ax = plt.subplots()
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  
        st.pyplot(fig)



