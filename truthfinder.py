import numpy as np
import pandas as pd
import jellyfish

pd.options.mode.chained_assignment = 'raise'

class TruthFinder(object):
    '''
    TruthFinder model implementation, finds true values about objects from conflicting sources (Veracity problem).

    Attributes:
        df (DataFrame): Dataframe containing the data,
        TODO
    '''
    def __init__(self, df, fact, obj, implication=None, dampening_factor=0.3, relatedness_factor=0.5):
        self.df = df
        self.fact = fact
        self.object = obj

        if implication==None:
            self.implication = self._std_implication
        else:
            self.implication = implication

        self.dampening_factor = dampening_factor
        self.relatedness_factor = relatedness_factor


    def compute(self, max_it = 200, tolerance = 0.001, initial_trust = 0.9):
        self.df['trust'] = initial_trust
        self.df['confidence'] = 0

        for i in range(max_it):
            t1 = self.df.drop_duplicates("source")["trust"]

            self.compute_fact_confidence()
            self.compute_source_trust()

            t2 = t1 = self.df.drop_duplicates("source")["trust"]

            # Convergence of the process is measured by the change in trustworthiness of sources
            error = (t1 @ t2.T) / (np.linalg.norm(t1)*np.linalg.norm(t2))
            error = 1 - error

            if error > tolerance:
                break
        
        return self.df

    def compute_fact_confidence(self):
        '''
        Computes a fact's confidence in three steps
        '''
        for obj in self.df[self.object].unique():
            idx = self.df[self.object] == obj
            data = self.df[idx].copy()
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
            sigma_f = sum(np.log(1-t) for t in t_s)
            data.at[i, "confidence"] = sigma_f
        return data

    def _adjust_confidence_related(self, data):
        '''
        Adjust a fact's confidence score by how influenced it is by other similar facts
        '''

        adjusted_confidences = {}
        for i, i_row in data.iterrows():
            f1 = i_row[self.fact]
            sigma_s = 0
            for j, j_row in self.df.drop_duplicates(self.fact).iterrows():
                f2 = j_row[self.fact]
                if f1 == f2:
                    continue
                sigma_s += j_row["confidence"] * self.implication(f2, f1)
            adjusted_confidences[i] = self.relatedness_factor * sigma_s + i_row["confidence"]
        
        data["confidence"] = adjusted_confidences.values()
        
        return data

    def _std_implication(self, f1, f2):
        return jellyfish.jaro_winkler(f1.lower(), f2.lower())

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
