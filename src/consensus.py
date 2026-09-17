import re

class LexicalConsensusEngine:
    """
    Computes statistical text disagreement and calibration confidence
    by measuring the lexical overlap (word sharing) between multiple strings.
    """
    
    def tokenize_and_clean(self, text: str) -> set:
        """
        Helper method to clean raw text and split it into unique word units.
        Removes punctuation and converts strings to lowercase.
        """
        # Lowercase everything to prevent "Gold" vs "gold" mismatch
        lowered_text = text.lower()
        
        # Regex expression keeping only alphanumeric characters and white spaces
        cleaned_text = re.sub(r'[^\w\s]', '', lowered_text)
        
        # Split string on spaces and store as a mathematical Set of unique words
        return set(cleaned_text.split())

    def calculate_jaccard_similarity(self, text_a: str, text_b: str) -> float:
        """
        Computes the classic Jaccard Similarity Coefficient between two text blocks.
        Formula: J(A, B) = |A ∩ B| / |A ∪ B|
        """
        set_a = self.tokenize_and_clean(text_a)
        set_b = self.tokenize_and_clean(text_b)
        
        # Edge case: If both responses are completely empty, they match perfectly
        if not set_a and not set_b:
            return 1.0
            
        intersection = set_a.intersection(set_b)  # Words found in BOTH sets
        union = set_a.union(set_b)                # Total unique words combining BOTH sets
        
        # Compute the ratio of shared vocabulary
        return len(intersection) / len(union)

    def evaluate_ensemble_consensus(self, responses: dict) -> dict:
        """
        Compares all pairs within the ensemble dictionary to determine group consensus.
        
        Parameters:
            responses (dict): Output dictionary from OllamaEnsembleManager.
        Returns:
            dict: Contains individual pair scores, average consensus, and systemic confidence rating.
        """
        model_names = list(responses.keys())
        similarity_scores = []
        
        # Pairwise comparison loops: (Model 0 vs 1), (Model 0 vs 2), (Model 1 vs 2)
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                m1, m2 = model_names[i], model_names[j]
                
                # Compute lexical similarity score between the two distinct models
                score = self.calculate_jaccard_similarity(responses[m1], responses[m2])
                similarity_scores.append(score)
                
        # Calculate the mathematical mean across all combinations
        avg_consensus = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
        
        # Map raw lexical consensus directly to a Systemic Calibration Confidence Score (0-100%)
        confidence_percentage = round(avg_consensus * 100, 2)

         # Safeguard: If less than two models successfully replied, consensus cannot be calculated
        if len(similarity_scores) == 0:
            return {
                "average_consensus": 0.0,
                "confidence_score": 0.0,
                "is_calibrated": False
            }

        # Calculate the mathematical mean across all valid working combinations
        avg_consensus = sum(similarity_scores) / len(similarity_scores)
        
        # Map raw lexical consensus directly to a Systemic Calibration Confidence Score (0-100%)
        confidence_percentage = round(avg_consensus * 100, 2)
        
        
        return {
            "average_consensus": avg_consensus,
            "confidence_score": confidence_percentage,
            # If confidence is dangerously low (< 35%), tag it as a hallucination warning zone
            "is_calibrated": confidence_percentage >= 35.0
        }
