import streamlit as st
import matplotlib.pyplot as plt
import io
import base64
import pickle
from PIL import Image

from utils import *


import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)


# def plot_to_base64(plt_obj):
#   """Converts a matplotlib plot to a base64 string."""

#   buffer = io.BytesIO()
#   plt_obj.savefig(buffer, format='png')
#   buffer.seek(0)
#   plot_data = base64.b64encode(buffer.read()).decode()
#   return plot_data

def load_data():
  """Loads data from the 'data.pickle' file."""

  topic_info = ""
  prob_tree_plot = None
  topic_challenges = ""
  sol_tree_plot = None
  sol_grades = None

  with open('data.pickle', 'rb') as file:
    top = pickle.load(file)
    topic_info = "Topic is " + top.name + "\n"
    prob_str = top.get_problems_str()
    prob_tree_fig = top.plot_hierarchy_problems()
    s1 = "\nWe look at 1 problem in particular and create from it a developed challenge. \n"
    topic_challenges = s1 + top.get_challenges_str()
    s2 = "\nFrom that challenge, we create 3 optional solutions:\n"
    sol_str = s2 + top.challenges[0].get_solutions_str()

    sol_tree_plot = top.challenges[0].plot_hierarchy_solutions((10, 4))
    sol_grades = top.challenges[0].plot_solutions_polygons(to_show=False)

  return topic_info, prob_str, prob_tree_fig, topic_challenges, sol_str, sol_tree_plot, sol_grades

def main():
    # Load data from your 'data.pickle' file
    topic_info, prob_str, prob_tree_fig, topic_challenges, sol_str, sol_tree_plot, sol_grades = load_data()
    # topic_info, prob_str, prob_tree_fig = load_data()

    # Display the topic infoz
    st.markdown(f"## {topic_info}")
    
    st.markdown(f"### The problems are:")
    st.pyplot(prob_tree_fig)

    st.write(prob_str)

    # Display the challenges
    st.markdown(f"### Challenges")
    st.markdown(topic_challenges)

    # Display the solutions tree
    st.markdown(f"### Solutions Tree")
    st.pyplot(sol_tree_plot)

    # Display the solution grades
    st.markdown(f"### Solution Grades")
    st.write(sol_str)
    st.pyplot(sol_grades)
    
    chatbot_main()






# Chat bot code

def init():
    # Load the OpenAI API key from the environment variable
    load_dotenv()
    
    # test that the API key exists
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    # # setup streamlit page
    # st.set_page_config(
    #     page_title="Connversetion with AI-Agent",
    #     page_icon="🤖"
    # )


def chatbot_main():
    init()

    topic = "Backpack"
    chat = ChatOpenAI(temperature=0)

    # initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content=f"We are a company that makes {topic} , we want to upgrade our product. For that end we would like you to help our imployes understand and analyze the problems with the product and the solutions for those problems")
        ]

    st.header("discussion with AI-Bot🤖")

    # sidebar with user input
    with st.sidebar:
        user_input = st.text_input("Your message: ", key="user_input")

        # handle user input
        if user_input:
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner("Thinking..."):
                response = chat(st.session_state.messages)
            st.session_state.messages.append(
                AIMessage(content=response.content))

    # display message history
    messages = st.session_state.get('messages', [])
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + '_user')
        else:
            message(msg.content, is_user=False, key=str(i) + '_ai')

    
    
    
if __name__ == '__main__':
  main()
