#!/usr/bin/env python
import os
import sys
import warnings

from datetime import date

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stock_advisor.crew import USStockAdvisor  # Updated import

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    inputs = {
        'symbol': 'TSLA',
        'current_date': str(date.today())
    }
    
    try:
        USStockAdvisor().crew().kickoff(inputs=inputs)  # Updated class name
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        'symbol': 'MSFT',
        'current_date': str(date.today())
    }
    try:
        USStockAdvisor().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)  # Updated class name

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        USStockAdvisor().crew().replay(task_id=sys.argv[1])  # Updated class name

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        'symbol': 'AAPL',
        'current_date': str(date.today())
    }
    
    try:
        USStockAdvisor().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)  # Updated class name

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
