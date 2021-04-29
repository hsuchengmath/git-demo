import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel



class Pairwise_Sentence_Model:
    def __init__(self, std_Q=None, threshold=0.05):
        # language model part
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.model = AutoModel.from_pretrained("bert-base-chinese")
        # similarity calculation part
        self.cos_fuc = nn.CosineSimilarity(dim=1, eps=1e-6)
        # init
        self.std_Q = std_Q
        self.std_Q_emb = self.multi_sent_emb_init(self.std_Q)
        self.threshold = threshold    


    def multi_sent_emb_init(self, multi_sent=None):
        inputs = self.tokenizer(multi_sent,padding=True,truncation=True, return_tensors="pt")
        multi_sent_emb = self.model(**inputs)[1] # it is torch[N,768] and N is sent num
        return multi_sent_emb


    def calculate_similarity(self, std_Q_emb=None, customer_unit_emb=None):
        customer_similarity = self.cos_fuc(std_Q_emb, customer_unit_emb).tolist()
        return customer_similarity # it is list and N element is std_Q


    def threshold_bar_determine(self,customer_similarity=None, matched_multi_sent=None):
        determine_bool, matched_sent = False, list()
        for i, sent in enumerate(matched_multi_sent):
            sim = customer_similarity[i]
            if sim >= self.threshold:
                determine_bool = True
                matched_sent.append((sim, sent))
        if len(matched_sent) != 0:
            return determine_bool, max(matched_sent)[1]
        else:
            return determine_bool, matched_sent


    def forward(self, customer_multi_sent=None):
        in_std_faq, match_std_feq, not_in_std_faq = set(), set(), set()
        customer_multi_sent_emb = self.multi_sent_emb_init(customer_multi_sent)
        for i, customer_sent in enumerate(customer_multi_sent):
            customer_unit_emb = customer_multi_sent_emb[i].view(1,-1)
            customer_similarity = self.calculate_similarity(self.std_Q_emb, customer_unit_emb)
            determine_bool, matched_sent = self.threshold_bar_determine(customer_similarity, self.std_Q)
            if determine_bool is True:
                in_std_faq.add(customer_sent)
                match_std_feq.add(matched_sent)
            else:
                not_in_std_faq.add(customer_sent)
        return list(in_std_faq), list(match_std_feq), list(not_in_std_faq)


