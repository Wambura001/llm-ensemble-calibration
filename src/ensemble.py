import requests
import json

class OllamaEnsembleManager:
    """
    Manages communication with local Ollama model instances to gather 
    independent answers from distinct neural network architectures.
    """
    def __init__(self, base_url: str = "http://localhost:11434"):
        # The local URL where Ollama serves its native HTTP REST API
        self.api_url = f"{base_url}/api/generate"
        
        # The specific local models downloaded via terminal.
        # We mix different architectures (Meta, Microsoft, Google) to ensure independent errors.
        self.models = ["llama3.2:1b", "phi3", "gemma2:2b"]

    def query_single_model(self, model_name: str, prompt: str) -> str:
        """
        Dispatches a single HTTP POST request to a designated local model.
        
        Parameters:
            model_name (str): The exact identifier string for the Ollama model.
            prompt (str): The raw text question/instruction from the user.
        Returns:
            str: The raw text response produced by the model.
        """
        # Configuration payload matching Ollama's API specifications
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,  # Requesting the full string at once instead of token streaming
            "options": {
                "temperature": 0.2  # Low temperature minimizes structural noise, capturing core facts
            }
        }
        
        try:
            # Issue synchronous POST request with a 45-second fallback timeout
            response = requests.post(self.api_url, json=payload, timeout=45)
            response.raise_for_status()  # Throws an exception for HTTP errors (e.g., 404, 500)
            
            # Extract and return the stripped text value from the response payload
            return response.json().get("response", "").strip()
            
        except requests.exceptions.RequestException as e:
            # Catching network glitches, missing models, or engine crashes gracefully
            return f"Error executing inference on {model_name}: {str(e)}"

    def get_ensemble_responses(self, prompt: str) -> dict:
        """
        Orchestrates sequential queries across all listed models.
        Filters out any models that return errors to protect downstream math metrics.
        """
        results = {}
        print(f"\nDispatching prompt to local ensemble models...")
        
        for model in self.models:
            print(f" -> Querying model: {model}")
            answer = self.query_single_model(model, prompt)
            
            # If the response contains an error prefix string, omit it from the final dictionary
            if "Error executing inference" in answer or not answer:
                print(f"Warning: {model} failed to respond. Skipping from consensus matrix.")
                continue
                
            results[model] = answer
            
        return results
