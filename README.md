# Analysis-and-Research-on-Training-Methods
project one in the course mathematical modeling

Here we aim to analyze the training of reading literature based on Large Language Model (LLM).

Practical operation: Nine articles were selected to conduct supervision training without feedback and supervision training with feedback respectively, and the evaluation data was Key information extraction accuracy score Accuracy score of language expression (BLEU score) Integrity score of information (Jaccard distance) The topic relevance score (cosine similarity) has the greatest reference value. It calculates cosine similarity based on TF-IDF, which can reduce the influence of article length, and other items are used as auxiliary evaluation (because they are greatly affected by length). Without training, the first nine rounds sent nine different paper document tests, and the last round sent the first round of documents, comparing the results of the tenth round with those of the first round.

When there is training, Next, I will conduct 10 rounds of training on reading scientific literature for you. Firstly, I will provide you with a document that requires you to read the content and summarize its main content. At the end of each round, I will provide you with feedback on the accuracy score of key information extraction, accuracy score of language expression (BLEU score), completeness score of information (Jaccard distance), and topic relevance score (cosine similarity). You should try to improve these scores as much as possible during the training process. Then, after each round of training, feedback each score and ask it to improve, repeat nine times, and throw the last time to the model to test the paper documents sent in the first round, get the tenth round of scores, and then compare with the first round to get the training results.

Actually the code can be straighlty utilize. 
