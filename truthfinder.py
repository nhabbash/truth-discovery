import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from strsimpy.cosine import Cosine

def cosine_sim(f1, f2):
    cosine = Cosine(2)
    p0 = cosine.get_profile(f1)
    p1 = cosine.get_profile(f2)
    return cosine.similarity_profiles(p0, p1)

def character_token_sim(f1, f2):
    return (fuzz.token_set_ratio(f1.lower(), f2.lower()) / 100)

class TruthFinder(object):
    '''
    TruthFinder model implementation, finds true values about objects from conflicting sources (Veracity problem).

    Attributes:
        df (DataFrame): DataFrame containing the data,
        fact (string): Fact/Attribute column name in the DataFrame
        obj (string): Object/Identifier column name in the DataFrame
        implication (function): Similarity function between strings
        initial_trust (float): Initial sources trustworthiness
        dampening_factor (float): Dampening factor (gamma) to account for source dependance
        relatedness_factor (float): Relatedness factor (rho) to account for the influence of related facts
        base_sim (float): Threshold for positive implication
    '''
    def __init__(self, df, fact, obj, implication = None, initial_trust = 0.9, dampening_factor = 0.3, relatedness_factor = 0.5, base_sim = 0.5):
        self.df = df
        self.fact = fact
        self.object = obj
    
        if implication==None:
            self.implication = cosine_sim
        else:
            self.implication = implication

        self.initial_trust = initial_trust
        self.dampening_factor = dampening_factor
        self.relatedness_factor = relatedness_factor
        self.base_sim = base_sim

    def compute(self, max_it = 10, tolerance = 0.001, progress = False):
        '''
        Iterative computation of fact confidences and source trustworthiness until stability
        '''

        self.df['trust'] = self.initial_trust
        self.df['confidence'] = 0.0

        for i in range(max_it):
            t1 = self.df.drop_duplicates("source")["trust"]

            self.compute_fact_confidence()
            self.compute_source_trust()

            t2 = self.df.drop_duplicates("source")["trust"]

            # Convergence of the process is measured by the change in trustworthiness of sources
            error = (t1 @ t2.T) / (np.linalg.norm(t1)*np.linalg.norm(t2))
            error = 1 - error

            if progress:
                print("Iteration: {}, Error: {}".format(i, error))

            if error > tolerance:
                break
        
        return self.extract_truth()

    def compute_fact_confidence(self):
        '''
        Computes a fact's confidence in three steps
        '''
        for obj in self.df[self.object].unique():
            idx = self.df[self.object] == obj
            data = self.df.loc[idx].copy(deep=True)
            data = self._initial_confidence(data)
            data = self._adjust_confidence_related(data)
            data = self._adjust_confidence_dependance(data)
            self.df.loc[idx] = data
    
    def _initial_confidence(self, data):
        '''
        Computes a fact's initial confidence score by summing all its sources' trustworthiness scores
        '''
        for i, row in data.iterrows():
            # Extracting source trustworthiness
            t_s = data.loc[data[self.fact] == row[self.fact], "trust"]
            # Computing source trustworthiness score (tau_s) and fact confidence score (sigma_f) at once
            sigma_f = sum(-np.log(1-(t-1e-5)) for t in t_s)
            data.at[i, "confidence"] = sigma_f
        return data

    def _adjust_confidence_related(self, data):
        '''
        Adjust a fact's confidence score by how influenced it is by other similar facts
        '''
        adjusted_confidences = {}
        for i, i_row in data.iterrows():
            f1 = i_row[self.fact]
            relatedness = 0.0
            for j, j_row in data.drop_duplicates(self.fact).iterrows():
                f2 = j_row[self.fact]
                if f1 == f2:
                    continue
                relatedness += j_row["confidence"] * (self.implication(f2, f1) - self.base_sim)

            adjusted_confidences[i] = self.relatedness_factor * relatedness + (1 - self.relatedness_factor) * i_row["confidence"]
            
        data["confidence"] = adjusted_confidences.values()
        return data

    def _adjust_confidence_dependance(self, data):
        '''
        Adjusts a fact's confidence to account for dependancies (dampening factor gamma) between sources and negative probabilities (by using a sigmoid)
        '''
        for i, row in data.iterrows():
            confidence = row["confidence"]
            data.at[i, "confidence"] = 1 / (1 + np.exp(-self.dampening_factor * confidence))

        return data

    def compute_source_trust(self):
       '''
       Recomputes every source trustworthiness as the average confidence of all the fact supplied by said source
       '''
       for source in self.df["source"].unique():
           idx = self.df["source"] == source
           t_s = self.df.loc[idx, "confidence"]
           self.df.loc[idx, "trust"] = sum(t_s) / len(t_s)
    
    def extract_truth(self):
        '''
        Extracts the singular objects with their maximum confidence facts
        '''
        object_collection = pd.DataFrame(columns=self.df.columns)
        for obj in self.df[self.object].unique():
            idx = self.df[self.object] == obj
            data = self.df.loc[idx].copy(deep=True)
            true_fact = data.loc[data["confidence"].idxmax()]
            object_collection = object_collection.append([true_fact], ignore_index=True)
        return object_collection


#### Other
    def _adjust_confidence_related2(self, data):
        '''
        Adjust a fact's confidence score by how influenced it is by other similar facts
        NOTE: vectorized similarity not implemented yet
        '''
        facts_set = data[self.fact].unique()
        facts_confidence = {x[self.fact]: x['confidence'] for _, x in data.iterrows()}
        confidences_array = np.array(list(facts_confidence.values()))
        
        for i, f in enumerate(facts_set):
            # For each source that provides a certain fact, update its confidence with similiarity factor
            similarity_sum = confidences_array[i] + self.relatedness_factor * sum(self.implication(f, facts_confidence, vect=True) * facts_array)
            # Update confidence score
            data.loc[data[self.fact] == f, 'confidence'] = similarity_sum

        return data
