# 🧠 NERO v3 – Advanced SEO Content Analyzer (Free & Open Source)

If your content looks great to humans but **Google doesn’t understand it**, you're not just missing clicks — you're invisible. ❌

Most content is written for people — but **Google reads like a machine**. It processes content through entities, semantic meaning, sentiment, and readability.

That’s exactly why I built **NERO v3**:  
To help creators, SEOs, and devs **see what Google sees** — and optimize accordingly.

---

## 🔍 What It Does

✅ **Entity Recognition** – Find people, places, products, orgs, dates  
✅ **Sentiment Analysis** – Gauge tone (positive, negative, neutral)  
✅ **Readability Scores** – Flesch & Gunning Fog Index for clarity  
✅ **TF-IDF + Keyword Density** – Understand keyword weight vs noise  
✅ **Keyword Clustering** – See how terms semantically group (via KMeans)  
✅ **Competitor Comparison** – Spot missing entities vs your rivals  
✅ **Visual Highlights** – Instantly see how Google might segment your text  
✅ **Simulated Content Score** – Quick NLP-based quality benchmark

---

## 💡 Why It Matters

Google's algorithm doesn’t just match keywords — it understands **entities**, **context**, and **semantic relationships**. If your content lacks structure or relevance, you're missing out on rankings.

🧠 **NERO v3** helps **bridge the gap** between what you write and what Google understands.

---

## ⚙️ Tech Stack

- Python 3.8+  
- [spaCy](https://spacy.io/) (NER, NLP)  
- [NLTK](https://www.nltk.org/) (Sentiment Analysis)  
- [scikit-learn](https://scikit-learn.org/) (TF-IDF, KMeans Clustering)  
- [textstat](https://pypi.org/project/textstat/) (Readability Metrics)  
- [Streamlit](https://streamlit.io/) (Web UI)

---

## ⚠️ Note

This tool isn’t perfect — but it’s a **powerful place to start** for SEO-focused NLP analysis.  
More features and polish will be added with time.

### 🔮 Planned Features

- Entity salience scoring (like Google NLP API)  
- Entity-keyword heatmaps  
- Internal linking signal visualization  
- Multilingual support

---

## 🚀 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/amal-alexander/nero-analyzer-v3
cd nero-analyzer-v3
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Launch the Streamlit App

```bash
streamlit run app.py
```

---

## 📸 Screenshots

![image](https://github.com/user-attachments/assets/4f5165c6-a939-41eb-b8b6-9e605a7993fb)

---

## 🧪 Example Use Cases

- ✅ Auditing content before publishing  
- 🧩 Comparing competitor content for missing semantic entities  
- 📚 Enhancing topic depth through entity clustering  
- 🔍 Quickly spotting readability and tone mismatches

---

## 🧠 Creator

Built by **Amal Alexander** – SEO Professional & Developer  
Feel free to connect or contribute!

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.  
Let’s collaborate to improve the next version of content analysis tools!

---

## 📄 License

**MIT License**
