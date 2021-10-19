from gensim.models.coherencemodel import CoherenceModel


def calc_topic_coherence(topics, texts, dictionary, model="c_npmi"):
    TC_model = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence=model)
    return TC_model.get_coherence()