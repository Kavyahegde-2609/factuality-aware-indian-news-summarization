"""
Factuality Verification Module
Author: Kavya Hegde
Date: January 2025

Novel contribution: Multi-dimensional factuality checking.
"""

import spacy
from sentence_transformers import SentenceTransformer
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import FACTUALITY_WEIGHTS


class FactualityChecker:
    
    def __init__(self):
        print(" Loading factuality checker...")
        
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            print(" Downloading spaCy model...")
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
            self.nlp = spacy.load('en_core_web_sm')
        
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        print(" Factuality checker ready\n")
    
    def extract_entities(self, text):
        doc = self.nlp(text)
        entities = {'PERSON': [], 'ORG': [], 'GPE': [], 'DATE': []}
        
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
        
        return entities
    
    def check_entity_consistency(self, summary, source):
        summary_ents = self.extract_entities(summary)
        source_ents = self.extract_entities(source)
        
        total = sum(len(v) for v in summary_ents.values())
        if total == 0:
            return 1.0
        
        matched = 0
        for ent_type, ents in summary_ents.items():
            for ent in ents:
                if ent in source_ents[ent_type]:
                    matched += 1
                elif any(ent.lower() in s.lower() for s in source_ents[ent_type]):
                    matched += 0.5
        
        return min(matched / total, 1.0)
    
    def check_temporal_consistency(self, summary, source):
        patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'\d{4}',
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}'
        ]
        
        summary_dates = set()
        source_dates = set()
        
        for pattern in patterns:
            summary_dates.update(re.findall(pattern, summary, re.IGNORECASE))
            source_dates.update(re.findall(pattern, source, re.IGNORECASE))
        
        if not summary_dates:
            return 1.0
        
        matched = summary_dates.intersection(source_dates)
        return len(matched) / len(summary_dates)
    
    def check_semantic_consistency(self, summary, source):
        summary_emb = self.encoder.encode([summary])
        source_emb = self.encoder.encode([source])
        
        sim = cosine_similarity(summary_emb, source_emb)[0][0]
        return float(sim)
    
    def compute_factuality_score(self, summary, source):
        entity_score = self.check_entity_consistency(summary, source)
        temporal_score = self.check_temporal_consistency(summary, source)
        semantic_score = self.check_semantic_consistency(summary, source)
        
        overall = (
            FACTUALITY_WEIGHTS['entity'] * entity_score +
            FACTUALITY_WEIGHTS['temporal'] * temporal_score +
            FACTUALITY_WEIGHTS['semantic'] * semantic_score
        )
        
        return {
            'entity': round(entity_score, 3),
            'temporal': round(temporal_score, 3),
            'semantic': round(semantic_score, 3),
            'overall': round(overall, 3)
        }


def test():
    print(" Testing factuality checker\n")
    
    checker = FactualityChecker()
    
    summary1 = "PM Modi announced Rs 20 lakh crore package on May 12, 2020."
    source1 = "Prime Minister Narendra Modi announced economic package of Rs 20 lakh crore on May 12, 2020."
    
    scores1 = checker.compute_factuality_score(summary1, source1)
    print("Test 1 (Good summary):")
    print(f"  Entity: {scores1['entity']}")
    print(f"  Temporal: {scores1['temporal']}")
    print(f"  Semantic: {scores1['semantic']}")
    print(f"  Overall: {scores1['overall']}\n")
    
    summary2 = "PM Modi announced package on June 15, 2020."
    
    scores2 = checker.compute_factuality_score(summary2, source1)
    print("Test 2 (Wrong date):")
    print(f"  Entity: {scores2['entity']}")
    print(f"  Temporal: {scores2['temporal']}")
    print(f"  Semantic: {scores2['semantic']}")
    print(f"  Overall: {scores2['overall']}\n")
    
    print(" Test complete")


if __name__ == "__main__":
    test()
