import os
import ast
import re
import sys
# sys.path.append('../')
from constants import OPENAI_API_KEY

import openai
from langchain.chat_models import ChatOpenAI
# from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


triplets_template = """given the factors: {factors}, give me the top 10 cross influence or relationship between these factors with no relate to {problem}
pls arrange it as a tripple A is connected in a certain way to B. like in knowledge graph.      
pls compose it as tripples, one factor in relation to one other factor.  
output as a triplets: (A) -[connected in a certain way]-> (B)
do not add any introduction to your result.
"""

class Topic:
    def __init__(self, topic_name):
        self.name = topic_name
        self.summary = ""
        self.challenges = []
        self.problems = []
        self.solutions = []
        self.kg = None
        self.triplets = []
        self.factors = []
        
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        
    def generate_problems(self, num_problems=5, num_classes=5):
        template = """we are a company that build {main_topic}, give me a list of {num} problems about our product that we need to solve. explaine each problem in one sentance.
        output in the format of python list"""#: [problem1, problem2, problem3, problem4, problem5, problem6, ...]"""
        #start each problem with "problem:" and end it with a new line."""
        
        llm = ChatOpenAI(temperature = 0.7, model="gpt-3.5-turbo")
        prompt = PromptTemplate(input_variables={"main_topic", "num"}, template=template)
        chain = LLMChain(llm=llm, prompt=prompt)
        problems = chain.run(main_topic = self.name, num = num_problems)
        print(problems)
        if not problems.startswith("["):
            problems = problems[problems.find("["):problems.find("]")+1]
        # problems = ast.literal_eval(problems)
        # combained_problems = self.combain_simillar_problems(problems)
        self.classify_problems(problems)
    
    def combain_simillar_problems(self, problems):
        template = """if there are similar problems merge them into 1 problem.
problems: {problems}
output the problems as python list"""
        llm = ChatOpenAI(temperature = 0, model="gpt-3.5-turbo")
        prompt = PromptTemplate(input_variables={"problems"}, template=template)
        chain = LLMChain(llm=llm, prompt=prompt)
        problems = chain.run(problems = problems)
        return problems
    
    # given string of list of problems, classify them to main and sub classes, and create problem objects and enter to the topic list
    def classify_problems(self, problems_str, num_classes=5):
        llm = ChatOpenAI(temperature = 0.3, model="gpt-3.5-turbo")
        # with appropriate names for each class
        classify_template = """ Classify the below problems in hierarchy way classes, with appropriate and meaningful names for each class. 
        add the problems themself. don't add any introduction to your result.
        problems: {problems}.
        output in the format of python list of dictionary: [["main_class": "main_class", "sub_class": "sub_class", "description": "description", "reporter": 'reporter'],...]"""

        prompt = PromptTemplate(input_variables={"problems"}, template=classify_template)
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run(problems = problems_str)#, num_classes = num_classes)
        prob_list = ast.literal_eval(result)
        for prob in prob_list:
            print(prob)
            self.problems.append(Problem(self.name, prob["description"], prob["main_class"], prob["sub_class"], prob["reporter"]))
    
    def classify_exsisting_problems(self, new_problems = ""):
        problems = self.get_problems_descriptions()
        self.problems = []
        self.classify_problems(problems + new_problems)
          
    def add_problem(self, problem):
        llm = ChatOpenAI(temperature = 0.5, model="gpt-3.5-turbo")
        template = """given the classifictions: {problems_classes} and problems: {descriptions}, and a new problem: {problem}.
    if the new problem is already exist in the problem list (could be in defferent words) return only the word "exist" without any introduction to your result. 
    else:
        find the best way to classify the new problem from the existing classofications.
        if none of the given classofication match perfectly to the new problem, create new class with meaningful name for the new problem.
        output in the format of:
        full problem:
        problem short name:
        new class:
    """
        prompt = PromptTemplate(input_variables={"problems_classes", "descriptions", "problem"}, template=template)
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run(problems_classes = self.get_problems_classes(), descriptions = self.get_problems_descriptions() ,problem = problem)
        print(result)
        if result.find("is already exist") != -1:
            print("problem already exist")
            return
        # result = result[result.find("\n"):]
        # print(result)
        dict_template = """the below text describes a problem and classification for that problem.
         output the problem and class in the format of python dictionary: ["class": "class_name", "problem_name": "name", "description": "description1", "reporter": 'reporter']
         the text: {input}"""
        llm2 = ChatOpenAI(temperature = 0, model="gpt-3.5-turbo")
        prompt2 = PromptTemplate(input_variables={"input"}, template=dict_template)
        chain2 = LLMChain(llm=llm2, prompt=prompt2)
        result = chain2.run(input = result)
        print(result)
        #check if "already exist" in result
        
        prob = ast.literal_eval(result)
        # self.problems.append(Problem(self.name, prob["description"], prob["class"], prob["problem_name"]))
        

    def plot_hierarchy_problems(self):
        G = nx.Graph()
        G.add_node(self.name)
        for p in self.problems:
            G.add_node(p.main_class)
            G.add_node(p.sub_class)
            G.add_edge(p.main_topic, p.main_class)
            G.add_edge(p.main_class, p.sub_class)

        fig, ax = plt.subplots(figsize=(15, 6)) #figure
        layout = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
        nx.draw(G, pos=layout, with_labels=True, font_size=10, font_weight='bold', node_size=1000)#, node_color=node_colors)
        # plt.show()
        return fig
    
    def create_all_triplets(self, factor_number=4):
        for prob in self.problems:
            prob.create_factors(factor_number)
            self.factors += prob.factors
        
        #create triplets
        print("create triplets")
        factors = [factor.name for factor in self.factors]
        factors += [prob.sub_class for prob in self.problems]
        prompt = PromptTemplate(input_variables={"factors", "problem"}, template=triplets_template)
        llm = ChatOpenAI(temperature = 0.2, model="gpt-3.5-turbo")
        chain = LLMChain(llm=llm, prompt=prompt)
        triplets_str = chain.run(factors = factors, problem = self.name)
        self.triplets = triplets_str.split("\n")
            
    
    def build_knowledge_graph(self):
        self.kg, edges_labels = create_graph(self.triplets)
        drow_KG(self.kg, edges_labels)
        
    #generate chllenge from the problem list with the given index
    def create_challenge(self, problem_index = 0):
        if len(self.problems) == 0:
            print("there is no problems in this topic")
            return
        if problem_index >= len(self.problems):
            print("problem index is out of range")
            problem_index = 0
            
        problem = self.problems[problem_index]
        template = """ pls write a simple challenge for the innovation team that improves {main_topic} {problem}: {description}
        give a short explanation about the problem and 2 espects to improve/kpi.
        also output the kpi as a python list: ['kpi1', 'kpi2', ...]
        """
        #""" pls write a technological challenge for the innovation team that improves {topic} {sub_class}: {description}"""
        prompt = PromptTemplate(input_variables={"main_topic", "problem", "description"}, template=template)
        llm = ChatOpenAI(temperature = 0.3, model='gpt-3.5-turbo')
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run(main_topic=problem.main_topic, problem=problem.sub_class, description=problem.description)
        kpi = ast.literal_eval(result[result.find("["):result.find("]")+1])
        result = result[:result.find("Python list")-1]  
        self.challenges.append(Challenge(self.name, problem.sub_class, result, kpi))
    
    def get_problems_classes(self):
        result = ""
        for prob in self.problems:
            result += prob.main_class + ", "
        return result
    
    def get_problems_descriptions(self):
        result = ""
        for prob in self.problems:
            result += prob.description + ", "
        return result
    
    def get_problems_str(self):
        result = ""
        for prob in self.problems:
            result += prob.problem_str()
            result += "\n\n"
        return result
    
    def get_challenges_str(self):
        result = ""
        for chall in self.challenges:
            result += chall.description
            result += "\n\n"
        return result
    
    def split_to_problems_solutions(text):
        template = """pls classify the following text to problem and solution description like so:
                    main topic: [main topic in 3 words]
                    problems: [problem description], explain each problem in one sentance
                    solutions: [solution description]
                    if there is no any problem description, just say "no problem"
                    if there is no any solution description, just say "no solution". 
                    
                    text to classify: {input}
                    """ #If there is no solution Ask to elaborate…
        prompt = PromptTemplate(input_variables={"input"}, template=template)
        llm = ChatOpenAI(temperature = 0.5, model="gpt-3.5-turbo")
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run(input=text)
        return result

    def read_conversation(self):
        try:
            with open(self.discussion_file, 'r') as file:
                conversation_parts = file.readlines()
            return conversation_parts
        except FileNotFoundError:
            print(f"File '{self.discussion_file}' not found.")
            return []

    def append_to_conversation(self, new_part):
        with open(self.discussion_file, 'a') as file:
            file.write(new_part + '\n')


class User:
    def __init__(self, username):
        self.username = username
        self.discussions = []
        self.kg = None
        
    def add_discussion(self, discussion):
        self.discussions.append(discussion)
        
class KG:
    def __init__(self, topic):
        self.topic = topic
        self.triplets = []
        self.nodes = []
        self.edges = []

class Factor:
    def __init__(self, name, description):
        self.name = ""
        self.description = ""

# from a string of triplets, create a knowledge graph
# return the graph and the edges labels
def create_graph(triplets):
        # Create a directed graph
        G = nx.DiGraph()
        edges_labels = {}

        # Iterate over the triplets and add them to the graph
        for triplet in triplets[:]:
            if triplet == '':
                continue

            print(triplet)
            subject = triplet.split('-')[0]
            subject = re.sub(r'^[^a-zA-Z]*|[^a-zA-Z]*$', '', subject)
            predicate = triplet.split('-[')[1].split(']-')[0]
            obj = triplet.split("->")[1]
            obj = re.sub(r'^[^a-zA-Z]*|[^a-zA-Z]*$', '', obj)
            print(subject, predicate, obj)
            G.add_edge(subject, obj, label=predicate)
            edges_labels[(subject, obj)] = predicate

        return G, edges_labels
    
def drow_KG(kg, edges_labels = []):
    # Print the knowledge graph
    layout = nx.spring_layout(kg)
    plt.figure(figsize=(15, 8))
    nx.draw(kg, pos=layout, with_labels=True, font_size=10, font_weight='bold', node_size=1000)#, node_color=node_colors)
    nx.draw_networkx_edge_labels(kg, pos=layout, edge_labels=edges_labels)
    plt.show()
        
        
class Problem:
    def __init__(self, topic, description, super_class, sub_class, reporter = None):
        self.description = description
        self.main_topic = topic
        self.main_class = super_class
        self.sub_class = sub_class
        self.reporter = reporter
        self.factors = []
        self.triplets = []
        self.user = None
        self.kg = None
        self.edges_labels = None
        
    
    def create_factors(self, factors_number=5):
        llm = ChatOpenAI(temperature = 0.5, model="gpt-3.5-turbo")
        template = """can you pls list the top {factors_number} most segnificate factors that influence: {problem}.
        output as a pytnon list of dictionary:""" + '[["factor_name": , "explanation": ], ...]'
        prompt = PromptTemplate(input_variables={"problem", "factors_number"}, template=template)
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run(problem = self.description, factors_number=factors_number)
        print(f"raw result: {result}")
        if not result.startswith("["):
            result = result[result.find("["):result.find("]")+1]
        
        factors_list = ast.literal_eval(result)
        # factors_list = factors_list[:10]
        print(f"factors list: {factors_list}")
        for factor in factors_list:
            self.factors.append(Factor(factor["factor_name"], factor["explanation"]))
        
        #create triplets
        triplets_template2 = """given the factors: {factors}, give me the top cross influence or relationship between these factors with no relate to {problem}
pls arrange it as a tripple A is connected in a certain way to B. like in knowledge graph.      
pls compose it as tripples, one factor in relation to one other factor.  
output as a triplets: (A) -[connected in a certain way]-> (B)
do not add any introduction to your result.
"""
        print("create triplets")
        factors = ""
        for f in self.factors:
            factors += f.name + ", "
        # factors = [f.name for f in self.factors]
        # print(factors)
        # factors = ",".join(factors)
        print(factors)
        # prompt = PromptTemplate(input_variables={"factors", "problem"}, template=triplets_template2)
        # chain = LLMChain(llm=llm, prompt=prompt)
        # triplets_str = chain.run(factors = factors, problem = self.sub_class)
        # self.triplets = triplets_str.split("\n")
        # for factor in factors:
        #     self.triplets.append(factor + " -[cause]-> " + self.sub_class)
        # print(self.triplets)
        

    def build_knowledge_graph(self):
        self.kg, self.edges_labels = create_graph(self.triplets)

        drow_KG(self.kg, self.edges_labels)
    
    # def __init__(self, description, topic):
    #     self.description = description
    #     self.main_topic = topic
    #     self._break_problem()
    
    # def _break_problem(self):
        class_schema = ResponseSchema(
            name="problem class",
            description="how to discribe the aspect of the problem? for example: supportive, mechanical, IT, etc.",)
        sub_class_schema = ResponseSchema(
            name="sub_class", description="a specific title to the problem.")
        # description_schema = ResponseSchema(
        #     name="problem",
        #     description="a description of the problem.",)

        response_schemas = [class_schema, sub_class_schema]#, description_schema]
        parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = parser.get_format_instructions()

        template = """Classify the below problem to general class and more detailed class in a hierarchical way.
                        just return the jason, do not add ANYTHING, NO INTERPRETATIONS!
        
                        the problem is: {problem}
                        
                        {format_instructions}
                        """
        prompt = ChatPromptTemplate.from_template(template=template)

        format_instructions = parser.get_format_instructions()

        messages = prompt.format_messages(
            problem=self.description,
            format_instructions=format_instructions,
        )

        chat = ChatOpenAI(temperature=0.0)
        response = chat(messages)
        output_dict = parser.parse(response.content)
        # print("output_dict", output_dict)
        self.main_class = output_dict["problem class"]
        self.sub_class = output_dict["sub_class"]
        

    def problem_str(self):
        prob_str =  """
    problem class: {main_class}
    sub class: {sub_class}
    problem details: {problem}""".format(main_class=self.main_class, sub_class=self.sub_class, problem=self.description)
        return prob_str
        
  
class Challenge:
    def __init__(self, topic, prob, text, kpi = []):
        base_kpi = ['cost', 'feasibility', 'efficiency']
        self.topic = topic
        self.description = text
        self.related_problem = prob
        self.kpi = base_kpi + kpi
        self.spec = ""
        self.spec_link = "" 
        self.solutions = []
    
    #for the challenge, generate a solutions, classify them and evaluate each solution 
    def create_solutions(self, number_of_solutions=2):
        # print("start create solutions")
        template = """ from this challenge: {challenge}, pls give me {num} solutions that can solve this challenge.
        each solution evaluate from 1 to 5 on the following kpi: {kpi}
        """
        prompt = PromptTemplate(input_variables={"challenge", "num", "kpi"}, template=template)
        llm = ChatOpenAI(temperature = 0.7, model='gpt-3.5-turbo')
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run(challenge = self.description, num = number_of_solutions, kpi = self.kpi)
        self.classify_solutions(result)
        
        # classify the solutions and create solution objects
    def classify_solutions(self, solutions_str):
        llm = ChatOpenAI(temperature = 0.5, model="gpt-3.5-turbo")
        #[Design , Technologic, Educational and Training, Environmental, Biological and Medical...]
        classify_template = """classify the below solution into hierarchical classes give each class short and meaningful name. the main class should indecate
        what the solution is fixing, and the sub class should indecate how the solution is fixing it. both classes names should be shor - no more than 3 words.
        add the solution themself. don't add any introduction to your result.
        solutions: {solutions}. 
        output in the format of python list of dictionary: ["main_class": "main_class", "sub_class": "sub_class", "description": "description1", "grade": "evaluate dict"...]"""

        prompt = PromptTemplate(input_variables={"solutions"}, template=classify_template)
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run(solutions = solutions_str)
        sol_list = ast.literal_eval(result)
        for sol in sol_list:
            print(sol)
            self.solutions.append(Solution(sol["description"], sol["main_class"], sol["sub_class"], sol["grade"]))
    
    def create_fundamental_problems(self):
        for sol in self.solutions:
            sol.fundamental_problem()
            
        
    def get_solutions_str(self):
        result = ""
        for sol in self.solutions:
            result += sol.sub_class + ": " + sol.description + "\n" + f"{sol.fundamental_problems}"
            result += "\n\n"
        return result
    
    def plot_hierarchy_solutions(self, size = (15,6)):
        G = nx.Graph()
        root = "solutions for " + self.related_problem
        G.add_node(root)
        for s in self.solutions:
            G.add_node(s.main_class)
            G.add_node(s.sub_class)
            G.add_edge(root, s.main_class)
            G.add_edge(s.main_class, s.sub_class)

        fig,ax = plt.subplots(figsize=size)
        layout = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
        nx.draw(G, pos=layout, with_labels=True, font_size=10, font_weight='bold', node_size=1000)#, node_color=node_colors)
        # plt.show()
        return fig
    
    
    def plot_solutions_polygons(self, to_show = True):
        max_score = 5

        def plot_polygon(sol, color, ax, lable = False):
            grade = sol[0]
            # Define the center point
            center = (0, 0)

            # Define the labels and scores for the vertices
            vertices = [{"label": key, "score": val} for key, val in grade.items()]

            # Calculate the angles and distances for the vertices
            angles = np.linspace(0, 2 * np.pi, len(vertices), endpoint=False)
            distances = [vertex["score"] for vertex in vertices]

            # Calculate the coordinates of the vertices
            x_coords = center[0] + distances * np.cos(angles)
            y_coords = center[1] + distances * np.sin(angles)

            # Plot the center
            ax.plot(center[0], center[1], 'ko', label='Center')

            # Plot the vertices and add labels
            for i, vertex in enumerate(vertices):
                if lable:
                    label = vertex["label"]
                else:
                    label = "" #str(vertex["score"])
                ax.plot(x_coords[i], y_coords[i], f'{color}o', label=f'{vertex["label"]} (Score: {vertex["score"]})')
                ax.text(x_coords[i], y_coords[i], label, fontsize=12, ha='center', va='center')

            # Connect all the vertices to form a polygon
            for i in range(len(vertices)):
                next_i = (i + 1) % len(vertices)
                ax.plot([x_coords[i], x_coords[next_i]], [y_coords[i], y_coords[next_i],], color + '--')

        # Create a figure and axis
        fig, ax = plt.subplots()
        
        # Define the list of solutions and colors
        sols_list = []  # Fill this list with your solutions
        for sol in self.solutions:
            sols_list.append((sol.grade, sol.sub_class))
        colors = ["b", "g", "k", "p"]  # Define colors for each solution
        colors_dict = {'b':'blue', 'g':'green', 'p':'purple', 'o':'orange', 'k':'black'}
        
        # Add the maximum polygon (with distance max_score=5 on all vertices)
        max_grade = {}
        # grades = [sol[0] for sol in sols_list]
        for key in sols_list[0][0].keys():
            max_grade[key] = max_score
        plot_polygon((max_grade, 0), 'r', ax, True)

        # Plot multiple polygons one on top of the other
        patches = []
        lable = False
        for sol, color in zip(sols_list, colors):
            plot_polygon(sol, color, ax, lable)
            patches.append(mpatches.Patch(color=colors_dict[color], label=sol[1]))

        ax.legend(handles=patches, loc='upper left')

        # Set axis limits
        max_distance = max_score
        ax.set_xlim(-max_distance - 1, max_distance + 1)
        ax.set_ylim(-max_distance - 1, max_distance + 4)
        
        # Show the plot
        plt.grid(True)
        if to_show:
            print("show plot")
            plt.show()
        return fig

    
# each solution related to a challenge      
class Solution:
    def __init__(self, description, main_class, sub_class, grades, user = None):
        self.description = description
        self.summary = general_summary(description)# temp = 0.3, max_words = 1
        self.user = user
        self.main_class = main_class
        self.sub_class = sub_class
        self.validate = None # validate the solution on the api
        self.grade = grades # grade the solution on the kpi
        self.fundamental_problems = ""

    def break_solution(self):
        class_schema = ResponseSchema(
            name="solution class",
            description="how to discribe the aspect of the solution? for example: supportive, mechanical, IT, etc.",)
        sub_class_schema = ResponseSchema(
            name="sub_class", description="a specific title to the solution.")

        response_schemas = [class_schema, sub_class_schema]#, description_schema]
        parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = parser.get_format_instructions()

        template = """Classify the below solution to general class and more detailed class in a hierarchical way.
                        just return the jason, do not add ANYTHING, NO INTERPRETATIONS!
        
                        the solution is: {solution}
                        
                        {format_instructions}
                        """
        prompt = ChatPromptTemplate.from_template(template=template)

        format_instructions = parser.get_format_instructions()

        messages = prompt.format_messages(
            solution=self.description,
            format_instructions=format_instructions,
        )

        chat = ChatOpenAI(temperature=0.0)
        response = chat(messages)
        output_dict = parser.parse(response.content)
        print("output_dict", output_dict)
        self.main_class = output_dict["solution class"]
        self.sub_class = output_dict["sub_class"]
       
     
    def update_solution(self, free_text):
        update_template = """" given the solution: {solution}, pls update the solution with the following text: {text}.
        add any new details to the solution summary and update the evaluation(grade) of the solution.
        keep on the solution format. output the updated solution as a python dictionary: ["main_class": "class1", "sub_class": "sub_class1", "description": "description1", "grade": "evaluate dict"...]
        """
        llm = ChatOpenAI(temperature = 0.0, model="gpt-3.5-turbo")
        prompt = PromptTemplate(input_variables={"solution", "text"}, template=update_template)
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run(solution = self.solution_str(), text = free_text)
        print(result)
        new_solution = ast.literal_eval(result[result.find("{"):])
        self.description = new_solution["description"]
        self.grade = new_solution["grade"]
        self.main_class = new_solution["main_class"]
        self.sub_class = new_solution["sub_class"]
        
        
    def fundamental_problem(self):
        template = """given a solution, pls identify the basic problems so that if they are solved we can make the solution work.
        solution: {solution}
        output in the format of: fundamental problems:
        1. problem1
        2. problem2"""
        llm = ChatOpenAI(temperature = 0.7, model="gpt-3.5-turbo")
        prompt = PromptTemplate(input_variables={"solution"}, template=template)
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run(solution = self.description)
        self.fundamental_problems = result
        # print(result)
        return result
        
        
    def solution_str(self):
        sol_str = """
                [solution class: {solution_class} \n
                sub class: {sub_class} \n
                solution description: {solution} \n
                grade: {grades} \n
                {fp}]""".format(solution_class=self.main_class, sub_class=self.sub_class, solution=self.description, grades=self.grade, fp=self.fundamental_problems)
        return sol_str
    
    def plot_grades_histogram(self):
        #make the names to not go over one another
        
        plt.bar(self.grade.keys(), self.grade.values())
        # plt.xticks(rotation='vertical')
        plt.xticks(rotation=320)
        plt.show()
        

def general_summary(text, temp = 0.3, max_sentances = 1):
    template = """ summary the following text up to {max_sentances}, the text: '{text}'
    """
    prompt = PromptTemplate(input_variables={"max_sentances", "text"}, template=template)
    llm = ChatOpenAI(temperature = temp, model='gpt-3.5-turbo')
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(max_sentances = max_sentances, text = text)
    return result


