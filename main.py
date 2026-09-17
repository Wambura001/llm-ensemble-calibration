import sys
import os
# Import our custom modules from the src folder
from src.ensemble import OllamaEnsembleManager
from src.consensus import LexicalConsensusEngine

def print_separator(char="=", length=60):
    """Helper utility to draw clean layout lines in the terminal interface."""
    print(char * length)

def main():
    print_separator()
    print("INITIALIZING CALIBRATED LLM ENSEMBLE SYSTEM")
    print_separator()

    # Step 1: Initialize our data manager and mathematical engine classes
    try:
        ensemble_manager = OllamaEnsembleManager()
        consensus_engine = LexicalConsensusEngine()
    except Exception as e:
        print(f"Initialization Failed: Ensure source structures match. Details: {e}")
        sys.exit(1)

    # Step 2: Establish a continuous loop to test user prompts interactively
    while True:
        try:
            print("\nEnter a challenging question or trick prompt below to test calibration.")
            print("(Type 'exit' or 'quit' to terminate the program)")
            user_query = input("\nUser Query: ").strip()

            # Break out of loop if the user specifies exit parameters
            if user_query.lower() in ['exit', 'quit', '']:
                print("\nShutting down pipeline. Good luck with your presentation!")
                break

            print_separator("-")
            
            # Step 3: Query the multiple local models sequentially via the manager module
            raw_responses = ensemble_manager.get_ensemble_responses(user_query)
            
            print_separator("-")
            print("COLLECTED ENSEMBLE OUTPUTS:")
            print_separator("-")
            
            # Print out what each independent neural network generated
            for model_name, response_text in raw_responses.items():
                print(f"[{model_name}] baseline output:")
                print(f"   \"{response_text}\"\n")

            print_separator("-")
            print("RUNNING STATISTICAL CONSENSUS EVALUATION...")
            print_separator("-")

            # Step 4: Pass model outputs to the consensus engine to calculate lexical overlap metrics
            metrics = consensus_engine.evaluate_ensemble_consensus(raw_responses)

            # Step 5: Render the final calibrated system diagnostic to the user
            print(f"Raw Average Jaccard Consensus Index: {metrics['average_consensus']:.4f}")
            print(f"Derived Calibration Confidence Score: {metrics['confidence_score']}%")
            
            # Conditional rendering acting as an agentic guardrail gating the validity of answers
            if metrics['is_calibrated']:
                print("STATUS: CALIBRATED CONSENSUS REACHED.")
                print("   The models largely agree on the structural wording. High probability of accuracy.")
                # Tie it back to the biggest model available for final answer synthesis
                print(f"\nSuggested Representative Answer (from Llama):\n   {raw_responses['llama3.2:1b']}")
            else:
                print("STATUS: CRITICAL HALLUCINATION/DIVERGENCE WARNING.")
                print("   The models produced fundamentally different lexical frameworks.")
                print("   Advice: Do not trust the baseline generation. The system lacks reliable facts to answer safely.")

            print_separator()

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully without printing nasty tracebacks to the user
            print("\n\nPipeline execution interrupted. Exiting program safely.")
            break
        except Exception as e:
            print(f"An error occurred during execution run: {e}")
            print_separator()

if __name__ == "__main__":
    main()
