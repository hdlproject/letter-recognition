import spacy
from spacy_llm.util import assemble

class NLP:
    def __init__(self):
        # self.nlp = spacy.load("en_core_web_sm")
        self.nlp = assemble("nlp.cfg")

    def get_ner(self, text):
        doc = self.nlp(text)

        for entity in doc.ents:
            print(entity.text, entity.label_)
