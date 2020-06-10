import numpy as np
import pandas as pd
import jellyfish

class TruthFinder(object):
    '''
    TruthFinder model implementation, finds true values about objects from conflicting sources (Veracity problem).

    Attributes:
        df (DataFrame): Dataframe containing the data,
        TODO
    '''
    def __init__(self, df, fact, obj, implication, dampening_factor=0.3, similarity_influence=0.5):
        self.df = df
        self.fact = fact
        self.object = obj
        self.implication = implication
        self.dampening_factor = dampening_factor
        self.similarity_influence = similarity_influence

    def compute(self, max_it = 10, tolerance = 0.001, initial_trust = 0.9):
        
        self.df['trust'] = initial_trust
        self.df['confidence'] = 0

        for i in range(max_it):
            t1 = self.df.drop_duplicates("source")["trust"]

            self.compute_fact_confidence()
            self.compute_source_trust()

            t2 = t1 = self.df.drop_duplicates("source")["trust"]

            # Convergence of the process is measured by the change in trustworthiness of sources
            error = (t1 @ t2.T) / (np.norm(t1)*np.norm(t2))
            error = 1 - error

            if error > tolerance:
                break
        
        return self.df

    def compute_fact_confidence(self):
        for obj in df[self.obj].unique():
            idx = df[self.object] == obj
            data = df[idx]
            data = self._compute_confidence(data)
            data = self._adjust_confidence_related(data)
            data = self._adjust_confidence_dependance(data)
            self.df.loc[idx] = data
    
    def _compute_confidence(self, data):
        '''
        Computes a fact's confidence score by summing all its sources' trustworthiness scores
        '''

        for i, row in df.iterrows():
            # Extracting source trustworthiness
            t_s = df.loc[df[self.fact] == row[self.fact], "trust"]
            # Computing source trustworthiness score (tau_s) and fact confidence score (sigma_f) at once
            sigma_f = sum(np.log(1-t) for t in ts)
            data.at[i, "confidence"] = sigma_f

        return data

    def _adjust_confidence_related(self, data):
        '''
        Adjust a fact's confidence score by how influenced it is by other similar facts
        '''
        
        return data
    

###########

def compute_confidence(df, objects, source_trust, attribute_key):
    all_objects_data = pd.DataFrame()
    
    for obj in objects:
        data = df[df['object'] == obj]
        
        # Sub-step 1: Compute from source trust
        data, confidence = compute_source_confidence(data, 
                                                    source_trust, 
                                                    attribute_key)
        # Sub-step 2: Adjust confidence with similarity between facts
        data, confidence = adjust_confidence_with_similarity(data,
                                                           confidence,
                                                           attribute_key)
        # Sub-step 3: Handle negative probabilities and independance dampening
        data, confidence = adjust_confidence_with_dampening(data, confidence)
        
        all_objects_data = pd.concat([all_objects_data, data])
        
    return all_objects_data

def compute_source_confidence(data, source_trust, attribute_key):
    #
    # Sums the confidence score for all sources for a specific fact
    #
    
    for idx, fact in data.iterrow():
        sources = get_sources_for_fact(data, fact, attribute_key)
        ln_sum = sum([source_trust[source] for source in sources])
        data.at[idx, 'confidence'] = ln_sum
    confidence = data['confidence'].values
    return (data, confidence)

def adjust_confidence_with_similarity(data, confidence, attribute_key):
    #
    # Adjust the fact's confidence score by how influenced it is by other facts
    #
    
    facts_set = data[attribute_key].unique()
    facts_confidence = {x[attribute_key]: x['confidence'] for _, x in data.iterrows()}
    facts_array = np.array(list(facts_confidence.values()))
    # Create a copy to assign new adjusted confidence values
    new_facts_array = copy.deepcopy(facts_array)
    for i, f in enumerate(facts_set):
        # For each source that provides a certain fact, update its confidence with similiarity factor
        similarity_sum = (1 - SIMILARITY_CONSTANT) * facts_array[i] + SIMILARITY_CONSTANT * sum(implicates(f, facts_confidence) * facts_array)
        # Update confidence score
        data.loc[data[attribute_key] == f, 'confidence'] = similarity_sum

    return (data, new_facts_array)
        
def implicates(fact, fact_sources):
    return [jellyfish.jaro_winkler(fact.lower(), f.lower)) -0.5 for f in fact_sources]


def adjust_confidence_with_dampening(data, confidence):
    #
    # Adjusts the fact's confidence to account for dependances and negative probabilities
    #
    for idx, claim in data.iterrows():
        data.at[idx, 'confidence'] = 1 / (1 + np.exp(-DAMPENING_FACTOR * claim['confidence']))
    return (data, confidence)

def compute_source_trust(data, sources):
    #
    # Computes every source trustworthyness by the average confidence of all facts supplied by a source
    #

    for source in sources:
        # t(w) trustworthiness of source w
        t_w = sum([confidence for confidence in data[data['source'] == source]['confidence'].values]) / len(data[data['source'] == source].index)
        # tau(w) trustworthiness score of source w
        tau_w = -np.log(1 - t_w)

        sources[source] = tau_w
    return sources

    
def truthfinder():
    while error > tol and it < max_iteration_count:
        source_trustworthiness_old = copy.deepcopy(source_trustworthiness)

        # 1. Compute fact confidence score
        data = compute_confidence(data, 
                                objects, 
                                source_trustworthiness, 
                                attribute_key)

        # 2. Compute source trustworthiness score
        source_trustworthiness = compute_source_trust(data,
                                                    source_trustworthiness)

        # Check convergence of the process
        error = 1 - np.dot(list(source_trustworthiness.values()), 
                        list(source_trustworthiness_old.values()) / 
                        (np.linalg.norm(list(source_trustworthiness.values())) * 
                        np.linalg.norm(list(source_trustworthiness_old.values()))))
