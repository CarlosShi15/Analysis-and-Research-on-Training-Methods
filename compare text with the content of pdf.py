import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.metrics import jaccard_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2


# 下载 NLTK 所需的分词模型



# 从 PDF 中提取文本内容
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_page = len(reader.pages)
        print(num_page)
        for page_num in range (num_page):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text


# 定义关键信息提取准确性评分函数
def accuracy_score(summary, reference):
    # 将总结和参考答案分词
    summary_tokens = set(word_tokenize(summary))
    reference_tokens = set(word_tokenize(reference))
    # 计算交集大小
    intersection = len(summary_tokens.intersection(reference_tokens))
    # 计算总词数
    total_words = len(reference_tokens)
    # 计算准确率
    accuracy = intersection / total_words if total_words > 0 else 0
    return accuracy


# 定义语言表达的准确性评分函数（使用 BLEU 分数）
def fluency_score(summary, reference):
    # 将总结和参考答案转为分词列表
    summary_tokens = word_tokenize(summary)
    reference_tokens = word_tokenize(reference)
    # 计算 BLEU 分数
    smoothing = SmoothingFunction().method1
    bleu_score = sentence_bleu([reference_tokens], summary_tokens, smoothing_function=smoothing)
    return bleu_score


# 定义信息的完整性评分函数（Jaccard 距离）
def completeness_score(summary, reference):
    # 将总结和参考答案转为分词列表
    summary_tokens = set(word_tokenize(summary))
    reference_tokens = set(word_tokenize(reference))
    # 计算 Jaccard 距离
    completeness = 1 - jaccard_distance(summary_tokens, reference_tokens)
    return completeness


# 定义主题相关性评分函数（余弦相似度）
def relevance_score(summary, reference):
    # 将总结和参考答案转为 TF-IDF 向量
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([summary, reference])
    similarity_matrix = cosine_similarity(tfidf_matrix)
    relevance = similarity_matrix[0, 1]
    return relevance


if __name__ == "__main__":
    summary = """We explore using neural operators, or neural network representations of nonlinear maps between function
spaces, to accelerate infinite-dimensional Bayesian inverse problems (BIPs) with models governed by nonlinear parametric partial differential equations (PDEs). Neural operators have gained significant attention
in recent years for their ability to approximate the parameter-to-solution maps defined by PDEs using as
training data solutions of PDEs at a limited number of parameter samples. The computational cost of
BIPs can be drastically reduced if the large number of PDE solves required for posterior characterization
are replaced with evaluations of trained neural operators. However, reducing error in the resulting BIP
solutions via reducing the approximation error of the neural operators in training can be challenging and
unreliable. We provide an a priori error bound result that implies certain BIPs can be ill-conditioned to
the approximation error of neural operators, thus leading to inaccessible accuracy requirements in training.
To reliably deploy neural operators in BIPs, we consider a strategy for enhancing the performance of neural
operators: correcting the prediction of a trained neural operator by solving a linear variational problem
based on the PDE residual. We show that a trained neural operator with error correction can achieve a
quadratic reduction of its approximation error, all while retaining substantial computational speedups of
posterior sampling when models are governed by highly nonlinear PDEs. The strategy is applied to two
numerical examples of BIPs based on a nonlinear reaction–diffusion problem and deformation of hyperelastic
materials. We demonstrate that posterior representations of the two BIPs produced using trained neural
operators are greatly and consistently enhanced by error correction."""
    #use you own path:r'C:\Users\***'
    pdf_reference_path = r'C:'
    reference = extract_text_from_pdf(pdf_reference_path)
    accuracy = accuracy_score(summary, reference)
    fluency = fluency_score(summary, reference)
    completeness = completeness_score(summary, reference)
    relevance = relevance_score(summary, reference)
    print("关键信息提取准确性评分：", accuracy)
    print("语言表达的准确性评分（BLEU 分数）：", fluency)
    print("信息的完整性评分（Jaccard 距离）：", completeness)
    print("主题相关性评分（余弦相似度）：", relevance)