"""
Quick Start Script for Monopoly Human vs AI

This script helps you quickly set up and play the game.
"""

import os
import sys

def check_model_exists():
    """Check if trained model exists."""
    model_path = os.path.join("models", "best_model", "dqn_policy.pt")
    return os.path.exists(model_path)

def main():
    print("=" * 70)
    print("MONOPOLY: HUMAN VS AI - QUICK START")
    print("=" * 70)
    print()
    
    # Check if model exists
    if not check_model_exists():
        print("No trained model found!")
        print()
        print("You need to train a model first. This will take some time.")
        print("Training will run for 500 episodes and save the model.")
        print()
        
        response = input("Would you like to train a model now? (yes/no): ").strip().lower()
        
        if response in ['yes', 'y']:
            print("\nStarting training...")
            print("This may take 15-30 minutes depending on your hardware.")
            print("You can watch the progress as it trains.\n")
            
            try:
                import subprocess
                result = subprocess.run([sys.executable, "train.py"], check=True)
                
                if check_model_exists():
                    print("\nTraining complete! Model saved.")
                    print()
                    response = input("Would you like to play now? (yes/no): ").strip().lower()
                    
                    if response in ['yes', 'y']:
                        print("\nStarting game...\n")
                        subprocess.run([sys.executable, "play_human_vs_ai.py"])
                    else:
                        print("\nYou can play later by running: python play_human_vs_ai.py")
                else:
                    print("\nTraining completed but model file not found.")
                    print("Please check train.py for errors.")
                    
            except subprocess.CalledProcessError as e:
                print(f"\nTraining failed with error: {e}")
                print("Please check the error messages above.")
            except KeyboardInterrupt:
                print("\n\nTraining interrupted by user.")
        else:
            print("\nNo problem! Train a model later with: python train.py")
            print("Then play with: python play_human_vs_ai.py")
    else:
        print("Trained model found!")
        print()
        print("You're ready to play against the AI.")
        print()
        
        response = input("Would you like to start a game? (yes/no): ").strip().lower()
        
        if response in ['yes', 'y']:
            print("\nStarting game...\n")
            
            try:
                import subprocess
                subprocess.run([sys.executable, "play_human_vs_ai.py"])
            except KeyboardInterrupt:
                print("\n\nGame interrupted by user.")
        else:
            print("\nYou can play later by running: python play_human_vs_ai.py")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
