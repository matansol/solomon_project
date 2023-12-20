import matplotlib.pyplot as plt
import pickle

from utils import *
from google_sheet import get_people_responses

from dotenv import load_dotenv


def main():
    load_dotenv()
    top = Topic("Backpack")
    responses = get_people_responses()
    solutions_str = ""
    problems_str = ""
    for response in responses:
        problems_str += response["problem"] + '. the reporter is ' + response['email'] + "\n"
        solutions_str += response["solution"] + '. the reporter is ' + response['email'] + "\n"

    # create problems and solutions from the people responses
    top.classify_problems(problems_str)
    top.classify_solutions(solutions_str)
    
    # for each problem create factors and triplets for the knowledge graph
    for prob in top.problems:
        prob.create_factors()
    
    
    
    # save to pickle file
    with open("demo.pickle", "wb") as file:
        pickle.dump(top, file)

if __name__ == '__main__':
    main()