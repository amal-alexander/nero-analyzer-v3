import streamlit as st
import spacy
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from textstat import flesch_reading_ease, gunning_fog, automated_readability_index
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

# Enhanced spaCy model loading with better error handling
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        try:
            # Try alternative loading method
            import en_core_web_sm
            return en_core_web_sm.load()
        except ImportError:
            # If model is not available, use a basic tokenizer
            st.warning("⚠️ Advanced spaCy model not available. Using basic NLP processing.")
            return spacy.blank("en")

# Initialize components
@st.cache_resource
def init_nltk():
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

nlp = load_spacy_model()
sentiment_analyzer = init_nltk()

# Enhanced color scheme for entities
ENTITY_COLORS = {
    "PERSON": "#ff6b6b",
    "ORG": "#ffa726",
    "GPE": "#66bb6a",
    "PRODUCT": "#29b6f6",
    "DATE": "#7986cb",
    "TIME": "#ab47bc",
    "MONEY": "#ec407a",
    "NORP": "#8d6e63",
    "LOC": "#26a69a",
    "EVENT": "#ff7043",
    "WORK_OF_ART": "#bdbdbd",
    "LAW": "#d4e157",
    "LANGUAGE": "#ffb74d",
    "PERCENT": "#81c784",
    "ORDINAL": "#9575cd",
    "CARDINAL": "#64b5f6"
}

# Page configuration
st.set_page_config(
    page_title="🧠 NERO v3 - Advanced SEO Analyzer", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .entity-highlight {
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🧠 NERO v3 - Advanced SEO Content Analyzer</h1>
    <p>Comprehensive NLP-powered content analysis with entity recognition, sentiment analysis, and SEO insights</p>
</div>
""", unsafe_allow_html=True)

# Enhanced sidebar with better organization
st.sidebar.header("⚙️ Analysis Settings")
st.sidebar.markdown("---")

# Core analysis toggles
st.sidebar.subheader("📊 Core Analysis")
sentiment_toggle = st.sidebar.checkbox("😊 Sentiment Analysis", value=True)
readability_toggle = st.sidebar.checkbox("📚 Readability Scores", value=True)
tfidf_toggle = st.sidebar.checkbox("📈 Keyword Density & TF-IDF", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🔬 Advanced Analysis")
cluster_toggle = st.sidebar.checkbox("🧠 Keyword Clustering")
wordcloud_toggle = st.sidebar.checkbox("☁️ Word Cloud Visualization")
compare_toggle = st.sidebar.checkbox("🆚 Competitor Comparison")

st.sidebar.markdown("---")
st.sidebar.subheader("📋 Export Options")
export_format = st.sidebar.selectbox("Export Format", ["CSV", "JSON", "Excel"])

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("✍️ Content Input")
    user_input = st.text_area(
        "Paste your content here:", 
        height=300,
        placeholder="Enter your content for comprehensive SEO analysis..."
    )

with col2:
    st.subheader("📊 Quick Stats")
    if user_input.strip():
        word_count = len(user_input.split())
        char_count = len(user_input)
        sentence_count = len([s for s in nlp(user_input).sents])
        
        st.metric("Words", word_count)
        st.metric("Characters", char_count)
        st.metric("Sentences", sentence_count)
        
        if word_count > 0:
            avg_words_per_sentence = round(word_count / sentence_count, 1)
            st.metric("Avg Words/Sentence", avg_words_per_sentence)

# Competitor comparison input
competitor_input = ""
if compare_toggle:
    st.subheader("🆚 Competitor Content")
    competitor_input = st.text_area(
        "Paste competitor content to compare:", 
        height=200,
        placeholder="Enter competitor content for comparison analysis..."
    )

# Analysis execution
if st.button("🔍 Analyze Content", type="primary"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some content to analyze.")
    else:
        with st.spinner("🔄 Processing your content..."):
            doc = nlp(user_input)
            
            # Check if we have a proper model with NER capabilities
            has_ner = doc.has_annotation("ENT_TYPE")
            
            # Extract entities with enhanced processing
            entities = []
            if has_ner:
                for ent in doc.ents:
                    entities.append({
                        "Entity": ent.text,
                        "Type": ent.label_,
                        "Description": spacy.explain(ent.label_) or ent.label_,
                        "Start": ent.start_char,
                        "End": ent.end_char
                    })
            else:
                # Fallback: Use basic keyword extraction if no NER
                st.warning("⚠️ Using basic keyword extraction. For full entity recognition, ensure spaCy model is properly installed.")
                # Simple keyword extraction based on capitalization and common patterns
                import re
                
                # Find capitalized words (potential proper nouns)
                capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', user_input)
                # Find numbers with currency symbols
                money_patterns = re.findall(r'[\$£€¥]\d+(?:,\d{3})*(?:\.\d{2})?', user_input)
                # Find dates
                date_patterns = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', user_input)
                
                for word in capitalized_words:
                    entities.append({
                        "Entity": word,
                        "Type": "PERSON_OR_ORG",
                        "Description": "Potential person or organization name",
                        "Start": user_input.find(word),
                        "End": user_input.find(word) + len(word)
                    })
                
                for money in money_patterns:
                    entities.append({
                        "Entity": money,
                        "Type": "MONEY",
                        "Description": "Monetary value",
                        "Start": user_input.find(money),
                        "End": user_input.find(money) + len(money)
                    })
                
                for date in date_patterns:
                    entities.append({
                        "Entity": date,
                        "Type": "DATE",
                        "Description": "Date reference",
                        "Start": user_input.find(date),
                        "End": user_input.find(date) + len(date)
                    })
            
            entity_df = pd.DataFrame(entities)
            
            if not entity_df.empty:
                entity_summary = (
                    entity_df.groupby(["Entity", "Type", "Description"])
                    .size()
                    .reset_index(name="Count")
                    .sort_values(by="Count", ascending=False)
                )
                
                # Enhanced scoring algorithm
                unique_entities = len(entity_summary)
                entity_types = entity_summary['Type'].nunique()
                total_mentions = entity_summary['Count'].sum()
                content_length = len(user_input.split())
                
                # Calculate comprehensive score
                entity_density = (total_mentions / content_length) * 100 if content_length > 0 else 0
                diversity_score = min((entity_types / 10) * 100, 100)  # Max 10 different types
                frequency_score = min((unique_entities / 20) * 100, 100)  # Max 20 unique entities
                
                overall_score = round((entity_density * 0.4 + diversity_score * 0.3 + frequency_score * 0.3), 1)
                
                # Display results in tabs
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Entity Analysis", "📈 Performance Metrics", "🎨 Visualizations", "📋 Export"])
                
                with tab1:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("🏷️ Identified Entities")
                        st.dataframe(entity_summary, use_container_width=True)
                    
                    with col2:
                        st.subheader("📊 Entity Distribution")
                        entity_type_counts = entity_summary.groupby('Type')['Count'].sum().sort_values(ascending=False)
                        fig = px.pie(
                            values=entity_type_counts.values, 
                            names=entity_type_counts.index,
                            title="Entity Types Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Enhanced text highlighting
                    st.subheader("🖍️ Entity-Highlighted Text")
                    highlighted_text = user_input
                    
                    if has_ner:
                        # Use spaCy entities for highlighting
                        for ent in sorted(doc.ents, key=lambda x: x.start_char, reverse=True):
                            color = ENTITY_COLORS.get(ent.label_, "#f0f0f0")
                            start, end = ent.start_char, ent.end_char
                            entity_text = highlighted_text[start:end]
                            replacement = f'<span class="entity-highlight" style="background-color: {color}; padding: 2px 4px; border-radius: 3px;" title="{ent.label_}: {spacy.explain(ent.label_) or ent.label_}">{entity_text}</span>'
                            highlighted_text = highlighted_text[:start] + replacement + highlighted_text[end:]
                    else:
                        # Use basic entity highlighting
                        for entity in sorted(entities, key=lambda x: x["Start"], reverse=True):
                            if entity["Start"] >= 0:  # Valid position
                                color = ENTITY_COLORS.get(entity["Type"], "#f0f0f0")
                                start, end = entity["Start"], entity["End"]
                                entity_text = highlighted_text[start:end]
                                replacement = f'<span class="entity-highlight" style="background-color: {color}; padding: 2px 4px; border-radius: 3px;" title="{entity["Type"]}: {entity["Description"]}">{entity_text}</span>'
                                highlighted_text = highlighted_text[:start] + replacement + highlighted_text[end:]
                    
                    st.markdown(f'<div style="line-height: 1.8; padding: 1rem; background: #fafafa; border-radius: 8px;">{highlighted_text}</div>', unsafe_allow_html=True)
                
                with tab2:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-container">
                            <h3>📈 Content Score</h3>
                            <h2 style="color: #667eea;">{overall_score}/100</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-container">
                            <h3>🏷️ Unique Entities</h3>
                            <h2 style="color: #667eea;">{unique_entities}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-container">
                            <h3>📊 Entity Types</h3>
                            <h2 style="color: #667eea;">{entity_types}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown(f"""
                        <div class="metric-container">
                            <h3>🎯 Entity Density</h3>
                            <h2 style="color: #667eea;">{entity_density:.1f}%</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Additional analysis sections
                    if sentiment_toggle:
                        st.subheader("😊 Sentiment Analysis")
                        sentiment_scores = sentiment_analyzer.polarity_scores(user_input)
                        
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            # Sentiment gauge chart
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number+delta",
                                value = sentiment_scores['compound'],
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': "Overall Sentiment"},
                                delta = {'reference': 0},
                                gauge = {
                                    'axis': {'range': [-1, 1]},
                                    'bar': {'color': "darkblue"},
                                    'steps': [
                                        {'range': [-1, -0.1], 'color': "lightgray"},
                                        {'range': [-0.1, 0.1], 'color': "gray"},
                                        {'range': [0.1, 1], 'color': "lightgreen"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 0.9
                                    }
                                }
                            ))
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.subheader("Detailed Sentiment Breakdown")
                            sentiment_df = pd.DataFrame([sentiment_scores]).T
                            sentiment_df.columns = ['Score']
                            sentiment_df.index = ['Negative', 'Neutral', 'Positive', 'Compound']
                            st.dataframe(sentiment_df)
                    
                    if readability_toggle:
                        st.subheader("📚 Readability Analysis")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            flesch_score = flesch_reading_ease(user_input)
                            fog_score = gunning_fog(user_input)
                            ari_score = automated_readability_index(user_input)
                            
                            readability_data = {
                                'Metric': ['Flesch Reading Ease', 'Gunning Fog Index', 'Automated Readability Index'],
                                'Score': [flesch_score, fog_score, ari_score]
                            }
                            
                            fig = px.bar(
                                readability_data, 
                                x='Metric', 
                                y='Score',
                                title="Readability Scores"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.write("**Score Interpretations:**")
                            st.write(f"• **Flesch Reading Ease**: {flesch_score:.1f}")
                            if flesch_score >= 90:
                                st.write("  → Very Easy (5th grade level)")
                            elif flesch_score >= 80:
                                st.write("  → Easy (6th grade level)")
                            elif flesch_score >= 70:
                                st.write("  → Fairly Easy (7th grade level)")
                            elif flesch_score >= 60:
                                st.write("  → Standard (8th-9th grade level)")
                            elif flesch_score >= 50:
                                st.write("  → Fairly Difficult (10th-12th grade level)")
                            elif flesch_score >= 30:
                                st.write("  → Difficult (College level)")
                            else:
                                st.write("  → Very Difficult (Graduate level)")
                
                with tab3:
                    if tfidf_toggle:
                        st.subheader("📈 Keyword Analysis")
                        
                        # TF-IDF Analysis
                        tfidf = TfidfVectorizer(
                            stop_words='english', 
                            ngram_range=(1, 3),
                            max_features=20
                        )
                        tfidf_matrix = tfidf.fit_transform([user_input])
                        tfidf_scores = tfidf_matrix.toarray()[0]
                        feature_names = tfidf.get_feature_names_out()
                        
                        tfidf_df = pd.DataFrame({
                            'Keyword': feature_names,
                            'TF-IDF Score': tfidf_scores
                        }).sort_values('TF-IDF Score', ascending=False)
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.dataframe(tfidf_df, use_container_width=True)
                        
                        with col2:
                            fig = px.bar(
                                tfidf_df.head(10), 
                                x='TF-IDF Score', 
                                y='Keyword',
                                orientation='h',
                                title="Top 10 Keywords by TF-IDF Score"
                            )
                            fig.update_layout(yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig, use_container_width=True)
                    
                    if wordcloud_toggle and user_input.strip():
                        st.subheader("☁️ Word Cloud")
                        try:
                            # Create word cloud
                            wordcloud = WordCloud(
                                width=800, 
                                height=400, 
                                background_color='white',
                                max_words=100,
                                colormap='viridis'
                            ).generate(user_input)
                            
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.axis('off')
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Could not generate word cloud: {str(e)}")
                    
                    if cluster_toggle:
                        st.subheader("🧠 Keyword Clustering")
                        try:
                            # Extract meaningful terms for clustering
                            doc_tokens = [token.lemma_.lower() for token in doc 
                                        if not token.is_stop and not token.is_punct 
                                        and len(token.text) > 2 and token.is_alpha]
                            
                            if len(set(doc_tokens)) >= 6:  # Need enough unique terms
                                tfidf_cluster = TfidfVectorizer(
                                    stop_words='english',
                                    max_features=50
                                )
                                
                                # Create documents from sliding windows
                                window_size = 50
                                documents = []
                                words = user_input.split()
                                for i in range(0, len(words), window_size):
                                    documents.append(' '.join(words[i:i+window_size]))
                                
                                if len(documents) > 1:
                                    X = tfidf_cluster.fit_transform(documents)
                                    
                                    n_clusters = min(3, len(documents))
                                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                                    clusters = kmeans.fit_predict(X)
                                    
                                    terms = tfidf_cluster.get_feature_names_out()
                                    
                                    # Get top terms for each cluster
                                    cluster_centers = kmeans.cluster_centers_
                                    cluster_terms = {}
                                    
                                    for i, center in enumerate(cluster_centers):
                                        top_indices = center.argsort()[-10:][::-1]
                                        cluster_terms[f"Cluster {i+1}"] = [terms[idx] for idx in top_indices]
                                    
                                    # Display clusters
                                    for cluster_name, cluster_words in cluster_terms.items():
                                        st.write(f"**{cluster_name}**: {', '.join(cluster_words[:5])}")
                                else:
                                    st.info("Content too short for meaningful clustering.")
                            else:
                                st.info("Not enough unique terms for clustering analysis.")
                        except Exception as e:
                            st.error(f"Clustering analysis failed: {str(e)}")
                
                with tab4:
                    st.subheader("📋 Export Your Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if export_format == "CSV":
                            csv_data = entity_summary.to_csv(index=False)
                            st.download_button(
                                "📥 Download Entity Data (CSV)",
                                csv_data,
                                "nero_entity_analysis.csv",
                                "text/csv"
                            )
                        elif export_format == "JSON":
                            json_data = entity_summary.to_json(orient='records', indent=2)
                            st.download_button(
                                "📥 Download Entity Data (JSON)",
                                json_data,
                                "nero_entity_analysis.json",
                                "application/json"
                            )
                    
                    with col2:
                        # Summary report
                        report = f"""
# NERO v3 Analysis Report

## Content Statistics
- **Words**: {len(user_input.split())}
- **Characters**: {len(user_input)}
- **Sentences**: {len([s for s in doc.sents])}
- **Content Score**: {overall_score}/100

## Entity Summary
- **Unique Entities**: {unique_entities}
- **Entity Types**: {entity_types}
- **Entity Density**: {entity_density:.1f}%

## Top Entities
{entity_summary.head(10).to_string(index=False)}
"""
                        st.download_button(
                            "📋 Download Full Report",
                            report,
                            "nero_analysis_report.md",
                            "text/markdown"
                        )
                
                # Competitor comparison
                if compare_toggle and competitor_input.strip():
                    st.subheader("🆚 Competitor Analysis")
                    
                    comp_doc = nlp(competitor_input)
                    
                    if has_ner:
                        comp_entities = [ent.text.lower() for ent in comp_doc.ents]
                        user_entities = [ent.text.lower() for ent in doc.ents]
                    else:
                        # Use extracted entities from our fallback method
                        comp_entities = []
                        user_entities = [entity["Entity"].lower() for entity in entities]
                        
                        # Basic entity extraction for competitor content
                        import re
                        comp_capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', competitor_input)
                        comp_entities = [word.lower() for word in comp_capitalized]
                    
                    overlap = set(user_entities) & set(comp_entities)
                    missing_from_user = set(comp_entities) - set(user_entities)
                    unique_to_user = set(user_entities) - set(comp_entities)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("🤝 Shared Entities", len(overlap))
                        if overlap:
                            st.write("**Examples:**", ", ".join(list(overlap)[:5]))
                    
                    with col2:
                        st.metric("❌ Missing Entities", len(missing_from_user))
                        if missing_from_user:
                            st.write("**Examples:**", ", ".join(list(missing_from_user)[:5]))
                    
                    with col3:
                        st.metric("✨ Unique to You", len(unique_to_user))
                        if unique_to_user:
                            st.write("**Examples:**", ", ".join(list(unique_to_user)[:5]))
            
            else:
                st.warning("⚠️ No entities found in the content. Try analyzing longer or more specific content.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🧠 NERO v3 - Advanced SEO Content Analyzer | Powered by spaCy, NLTK & Streamlit</p>
    <p><small>For best results, ensure your content is at least 100 words long and contains specific entities like names, places, organizations, etc.</small></p>
</div>
""", unsafe_allow_html=True)