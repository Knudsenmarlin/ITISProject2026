import TFIDF

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))

plt.hist(TFIDF.scoresHumanArticles, bins=20, alpha=0.5, label="DeepSeek")
plt.hist(TFIDF.scoresDeepSeekArticles, bins=20, alpha=0.5, label="DeepSeek")
plt.hist(TFIDF.scoresChatGPTArticles, bins=20, alpha=0.5, label="DeepSeek")
plt.hist(TFIDF.scoresPerplexityArticles, bins=20, alpha=0.5, label="DeepSeek")
plt.xlabel("BM25 score")
plt.ylabel("Occurences")
plt.title("BM25 Score Distribution Across Article Types")
plt.legend()
plt.show()

