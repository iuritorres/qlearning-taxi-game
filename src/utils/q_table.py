"""This module contains functions to save and load Q-tables."""

import pickle


def load_q_table(filename):
    """Load a Q-table from a file."""
    with open(filename, "rb") as file:
        q = pickle.load(file)
    return q


def save_q_table(q, filename):
    """Save a Q-table to a file."""
    with open(filename, "wb") as file:
        pickle.dump(q, file)
