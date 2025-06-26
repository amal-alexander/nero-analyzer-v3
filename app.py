import streamlit as st
import spacy
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from textstat import flesch_reading_ease, gunning_fog

nlp = spacy.load("en_core_web_sm")
nltk.download('vader_lexicon')
sentiment_analyzer = SentimentIntensityAnalyzer()

ENTITY_COLORS = {
    "PERSON": "#ffadad",
    "ORG": "#ffd6a5",
    "GPE": "#caffbf",
    "PRODUCT": "#9bf6ff",
    "DATE": "#a0c4ff",
    "TIME": "#bdb2ff",
    "MONEY": "#ffc6ff",
    "NORP": "#fffffc",
    "LOC": "#e0f7fa",
    "EVENT": "#fbc4ab",
    "WORK_OF_ART": "#ede0d4",
}

st.set_page_config(page_title="🧠 NERO v3 - Free SEO Analyzer", layout="wide")
st.title("🧠 NERO - Entity & Keyword Analyzer v3 (Free)")

sentiment_toggle = st.sidebar.checkbox("🔁 Enable Sentiment Analysis")
compare_toggle = st.sidebar.checkbox("🔁 Compare with Competitor Content")
readability_toggle = st.sidebar.checkbox("📚 Show Readability Scores")
tfidf_toggle = st.sidebar.checkbox("📈 Show Keyword Density + TF-IDF")
cluster_toggle = st.sidebar.checkbox("🧠 Show Keyword Clusters (KMeans)")

user_input = st.text_area("✍️ Paste your content here:", height=300)
competitor_input = ""
if compare_toggle:
    competitor_input = st.text_area("🆚 Paste competitor content to compare:", height=300)

if st.button("🔍 Analyze"):
    if not user_input.strip():
        st.warning("Please paste content.")
    else:
        doc = nlp(user_input)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        entity_df = pd.DataFrame(entities, columns=["Entity", "Type"])
        entity_summary = (
            entity_df.groupby(["Entity", "Type"])
            .size()
            .reset_index(name="Count")
            .sort_values(by="Count", ascending=False)
        )

        st.subheader("📊 Entity Table")
        st.dataframe(entity_summary, use_container_width=True)
        st.download_button("📥 Download Entity CSV", entity_summary.to_csv(index=False), "entities.csv")

        score = sum(entity_summary['Count']) + entity_summary['Type'].nunique() * 2
        st.metric("📈 Simulated Content Score", f"{score}/100")

        highlighted_text = user_input
        for ent in doc.ents:
            color = ENTITY_COLORS.get(ent.label_, "#f0f0f0")
            pattern = re.escape(ent.text)
            highlighted_text = re.sub(
                pattern,
                f"<mark style='background-color: {color}'>{ent.text}</mark>",
                highlighted_text,
                flags=re.IGNORECASE
            )
        st.subheader("🖍️ Highlighted Entities")
        st.markdown(f"<div style='line-height: 1.6'>{highlighted_text}</div>", unsafe_allow_html=True)

        if sentiment_toggle:
            st.subheader("😊 Sentiment Analysis")
            sentiment_score = sentiment_analyzer.polarity_scores(user_input)
            st.write(sentiment_score)

        if compare_toggle and competitor_input.strip():
            comp_doc = nlp(competitor_input)
            comp_entities = [ent.text for ent in comp_doc.ents]
            user_entities = [ent.text for ent in doc.ents]
            overlap = set(user_entities) & set(comp_entities)
            missing = set(comp_entities) - set(user_entities)
            st.subheader("🧩 Competitor Entity Comparison")
            st.markdown(f"✅ **Overlapping:** `{', '.join(overlap) or 'None'}`")
            st.markdown(f"❌ **Missing:** `{', '.join(missing) or 'None'}`")

        if readability_toggle:
            st.subheader("📚 Readability Scores")
            st.write(f"Flesch Reading Ease: {flesch_reading_ease(user_input):.2f}")
            st.write(f"Gunning Fog Index: {gunning_fog(user_input):.2f}")

        if tfidf_toggle:
            st.subheader("📈 Keyword Density + TF-IDF")
            tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = tfidf.fit_transform([user_input])
            tfidf_df = pd.DataFrame(tfidf_matrix.T.toarray(), index=tfidf.get_feature_names_out(), columns=["TF-IDF"])
            tfidf_df = tfidf_df.sort_values("TF-IDF", ascending=False).head(15)
            st.dataframe(tfidf_df)

        if cluster_toggle:
            st.subheader("🧠 Keyword Clusters (KMeans)")
            tfidf = TfidfVectorizer(stop_words='english')
            X = tfidf.fit_transform([user_input])
            kmeans = KMeans(n_clusters=3, random_state=0).fit(X.T)
            terms = tfidf.get_feature_names_out()
            clusters = pd.DataFrame({"Keyword": terms, "Cluster": kmeans.labels_})
            st.dataframe(clusters.sort_values("Cluster"))