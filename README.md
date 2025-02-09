# Emotion Classification and Analysis

## Project Overview
This project focuses on analyzing and classifying emotions from text data using various Natural Language Processing (NLP) techniques. The dataset used is the [Text Emotion Dataset](https://raw.githubusercontent.com/abishekarun/Text-Emotion-Classification/master/text_emotion.csv), which consists of posts and tweets labeled with different emotions.

## Tasks Implemented
### 1. Data Preparation & Category-Wise Dataframes
- Group posts by category.
- Create separate dataframes for each emotion category.

### 2. Vocabulary Analysis
- Construct vocabulary sets for each category.
- Compute statistical metrics including:
  - Vocabulary size
  - Minimum, maximum, average, and standard deviation of post length (in tokens)
  - Average number of pronouns per post
  - Average number of uncommon characters per post
  - Average number of repetitions per post
- Summarize results in a table.

### 3. Vocabulary Overlap Analysis
- Compute the proportion of common vocabulary over the total vocabulary size for every pair of categories.
- Represent results as a **12×12 similarity matrix**.

### 4. Frequent Token Analysis
- Identify the **30 most frequent tokens** in each category.
- Compute the number of common tokens between each pair of categories.
- Output results as a **12×12 matrix**.

### 5. Lexicon-Based Emotion Evaluation
#### Using WordNet Affect
- Extract dominant emotion labels from posts using **WordNet Affect**.
- Compute the **five most dominant emotions**.
- Construct weighted word2vec embeddings for each category.
- Compute cosine similarity between the lexicon-based embedding and the manually assigned label.
- Summarize results in a **13×13 similarity matrix**.

#### Using NRC Emotion Lexicon
- Repeat the same process using the **NRC Emotion Lexicon**.
- Summarize results in a **13×13 similarity matrix**.

### 6. Circumplex Model Evaluation
- Evaluate emotion closeness based on the **Circumplex Model of Affect**.
- Utilize **Doc2Vec embeddings** to quantify similarity between related emotions.
- Summarize results in a table and provide analysis.

### 7. Machine Learning-Based Emotion Classification
- Train and evaluate **three ML models** (e.g., SVM, Random Forest, Logistic Regression).
- Use **TF-IDF features** (without stopword removal).
- Compute **precision, recall, and F1-score**.

### 8. Feature Selection and Stopword Removal
- Repeat ML classification with **stopword removal**.
- Experiment with different TF-IDF feature thresholds (1000, 500, 100).
- Compare performances using **confusion matrices**.

### 9. Bigram Feature Engineering
- Repeat classification using **bigrams** instead of unigrams.
- Compare results with previous models.

### 10. Deep Learning-Based Classification
- Implement a **Convolutional Neural Network (CNN)** for emotion classification.
- Compare CNN performance with previous ML models.

### 11. Findings & Discussion
- Analyze classification results using **relevant literature**.
- Identify additional input features that may improve performance.
- Discuss limitations of the dataset and the processing pipeline.

## Installation & Setup
### Prerequisites
Ensure you have the following installed:
- Python 3.x
- Required Python libraries (install using `pip`):
  ```sh
  pip install pandas numpy nltk scikit-learn gensim wordcloud NRCLex word2vec keras tensorflow
  ```

### Running the Scripts


## Repository Structure
```

```

## Results
- **Vocabulary and Statistical Analysis:** See summary tables.
- **Lexicon-Based Analysis:** Results in similarity matrices.
- **Machine Learning Classification:** Precision, Recall, and F1-score results.
- **Deep Learning Classification:** CNN model accuracy comparison.
- **Emotion Closeness (Circumplex Model):** Table summarizing similarity between emotions.

## Conclusion
This project provides insights into emotion classification and analysis using both **machine learning** and **deep learning** techniques. The study evaluates different preprocessing strategies, lexical resources, and embedding methods, offering a comprehensive pipeline for text-based emotion detection.

## License
This project is licensed under the MIT License.

## Contact
For questions, feel free to reach out!
- **Author:** [Your Name]
- **Email:** your.email@example.com
- **GitHub:** [Your GitHub Profile]

