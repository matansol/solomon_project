from utils import *

from flask import Flask, render_template
import pickle
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route('/')
def display_content():
    # Load data from your 'data.pickle' file
    topic_info, prob_str, prob_tree_plot, topic_challenges, sol_str, sol_tree_plot, sol_grades = load_data()

    return render_template('index.html',
                           topic_info=topic_info,
                           prob_str=prob_str,
                           prob_tree_plot=prob_tree_plot,
                           topic_challenges=topic_challenges,
                           sol_str=sol_str,
                           sol_tree_plot=sol_tree_plot,
                           sol_grades=sol_grades)

def load_data():
    # Load data from your 'data.pickle' file
    topic_info = ""
    prob_tree_plot = None
    topic_challenges = ""
    sol_tree_plot = None
    sol_grades = None

    with open('data.pickle', 'rb') as file:
        top = pickle.load(file)
        topic_info = "Topic is " + top.name + "\n"
        prob_str = top.get_problems_str()
        prob_tree_plot = plot_to_base64(top.plot_hierarchy_problems())
        s1 = "\nWe look at 1 problem in particular and create from it a developed challenge. \n"
        topic_challenges = s1 + top.get_challenges_str()
        s2 = "\nFrom that challenge, we create 3 optional solutions:\n"
        sol_str = s2 + top.challenges[0].get_solutions_str()

        sol_tree_plot = plot_to_base64(top.challenges[0].plot_hierarchy_solutions((10, 4)))
        sol_grades = plot_to_base64(top.challenges[0].plot_solutions_polygons(to_show=False))

    return topic_info, prob_str, prob_tree_plot, topic_challenges, sol_str, sol_tree_plot, sol_grades

def plot_to_base64(plt_obj):
    buffer = io.BytesIO()
    plt_obj.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode()
    return plot_data

if __name__ == '__main__':
    app.run()
