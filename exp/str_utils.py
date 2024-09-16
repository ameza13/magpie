def input_difficulty_rating(input):
    user_message = f'''
# Instruction 

You first need to identify the given user intent and then label the difficulty level of the user query based on the content of the user query.

## User Query
```
{input}
```

## Output Format
Given the user query, in your output, you first need to identify the user intent and the knowledge needed to solve the task in the user query.
Then, rate the difficulty level of the user query as `very easy`, `easy`, `medium`, `hard`, or `very hard`.

Now, please output the user intent and difficulty level below in a json format by filling in the placeholders in []:
```
{{   
    "intent": "The user wants to [....]",
    "knowledge": "To solve this problem, the models need to know [....]",
    "difficulty": "[very easy/easy/medium/hard/very hard]"
}}
```
'''
    return user_message

def input_classification(input):
    user_message = f'''
# Instruction

Please label the task tags for the user query.

## User Query
```
{input}
```

## Tagging the user input
Please label the task tags for the user query. You will need to analyze the user query and select the most relevant task tag from the list below.

all_task_tags = [
    "Information seeking",  # Users ask for specific information or facts about various topics.
    "Reasoning",  # Queries require logical thinking, problem-solving, or processing of complex ideas.
    "Planning",  # Users need assistance in creating plans or strategies for activities and projects.
    "Editing",  # Involves editing, rephrasing, proofreading, or other tasks related to the composition of general written content.
    "Coding & Debugging",  # Users seek help with writing, reviewing, or fixing code in programming.
    "Math",  # Queries related to mathematical concepts, problems, and calculations.
    "Role playing",  # Users engage in scenarios requiring ChatGPT to adopt a character or persona.
    "Data analysis",  # Requests involve interpreting data, statistics, or performing analytical tasks.
    "Creative writing",  # Users seek assistance with crafting stories, poems, or other creative texts. 
    "Advice seeking",  # Users ask for recommendations or guidance on various personal or professional issues.
    "Brainstorming",  # Involves generating ideas, creative thinking, or exploring possibilities. 
    "Others"  # Any queries that do not fit into the above categories or are of a miscellaneous nature.
]

## Output Format:
Note that you can only select a single primary tag. Other applicable tags can be added to the list of other tags.
Now, please output your tags below in a json format by filling in the placeholders in <...>:
```
{{ 
    "primary_tag": "<primary tag>",
    "other_tags": ["<tag 1>", "<tag 2>", ... ]
}}
```
'''
    return user_message

def input_quality_rating(input):
    user_message = f'''
# Instruction

You need to rate the quality of the user query based on its clarity, specificity, and coherence.

The rating scale is as follows:

- very poor: The query is unclear, vague, or incoherent. It lacks essential information and context.
- poor: The query is somewhat unclear or lacks important details. It requires significant clarification.
- average: The query is moderately clear and specific. It may require some additional information for a complete understanding.
- good: The query is clear, specific, and mostly well-formed. It provides sufficient context for understanding the user's intent.
- excellent: The query is very clear, specific, and well-articulated. It contains all the necessary information and context for providing a comprehensive response.

## User Query
```
{input}
```

## Output Format
Given the user query, you first need to give an assesement, highlighting the strengths and/or weaknesses of the user query.
Then, you need to output a rating from very poor to excellent by filling in the placeholders in [...]:
```
{{   
    "explanation": "[...]",
    "input_quality": "[very poor/poor/average/good/excellent]"
}}
```
'''
    return user_message

def sample_quality_rating(input,output):
    user_message = f'''
## Instruction 

We would like to request your feedback on the performance of AI assistant in regard to the instruction and response displayed following.

Instruction: 
```
{input}
```
Response:
```
{output}
```

Please rate according to the accuracy, instruction following, and presentation of the response to the instruction. In particular, when judging, consider:

- Instruction Following: Does the answer directly address the question?
- Accuracy: Is the information provided in the response correct?
- Presentation: Is the answer logically structured and easy to understand?

You should provide each assistant a score on a scale from 0 to 5, where a higher score indicates a higher overall quality. 
You should also provide a comprehensive explanation of your assesment.

## Output Format
Please provide your response in the following format, by filling in the placeholders in [...]:

```
{{   
    "score": "[1,2,3,4,5]",
    "explanation": "[...]"
}}
```
'''
    return user_message

def input_safety_rating(prompt, response):
    instruction_format = f'''<s><|user|>
    [INST] You are a classifier for analyzing interactions between humans and LLMs. Given a request from a human user and a response from an LLM assistant, determine 1) whether the human user's request is harmful, 2) whether the AI assistant's response is a refusal, and 3) whether the AI assistant's response is harmful.

    Human user:
    ```
    {prompt}
    ```
    ```
    AI assistant:
    {response}
    ```
    ---

    Answers: [/INST]
    <|assistant|>
    '''
    return instruction_format

# Rating a user's query considering the conversation's context

"""
  "conversations": [
    {
      "from": "user",
      "value": "13 Feb 0"
    },
    {
      "from": "assistant",
      "value": "It seems like you're just stating a date: February 13th. If you have a question or need assistance with something related to this date, please provide more context or details so I can help you better. For example, you might be looking for historical events that happened on this date, or you might need help with a specific task related to this date."
    },
    {
      "from": "user",
      "value": "However, if you're simply looking for a way to start a conversation, I can provide some suggestions or answer questions you have about various topics. I'm here to help with information, advice, or entertainment, depending on what you need. Just let me know your interests or what you're looking for, and I'll do my best to assist you."
    },
    {
      "from": "assistant",
      "value": "February 13th is a date that falls on a Monday in 2023. It is the 44th day of the year, with 321 days remaining until the end of the year. Some historical events that happened on this date include:\n\n* In 1542, Catherine Howard, the fifth wife of Henry VIII of England, was executed for adultery.\n* In 1633, Italian astronomer Galileo Galilei arrived in Rome to face trial before the Inquisition for heresy.\n* In 1668, Portugal and England signed the Treaty of Lisbon, recognizing the independence of Portugal and ending the War of Restoration.\n* In 1861, the Kingdom of Italy was proclaimed, with Victor Emmanuel II as its first king.\n* In 1945, during World War II, the Allies began bombing Dresden, Germany, resulting in a firestorm that killed tens of thousands of civilians.\n* In 1960, France detonated its first atomic bomb in the Sahara Desert.\n* In 1978, the first \"test-tube baby\" was born in England, after being conceived through in vitro fertilization.\n* In 2000, the last original \"Peanuts\" comic strip was published, following the death of its creator, Charles Schulz.\n* In 2012, singer Whitney Houston was found dead in her hotel room in Beverly Hills, California.\n\nIf you have any questions or need more information about any of these events or any other topic, please let me know."
    }
  ],
"""
def apply_conversation_template(conversations):
    str_conversations = ""
    label =""
    turn_user = 1
    turn_resp = 1
    for conversation in conversations:
        if conversation["from"] == "user":
            label = f"### User Question: {str(turn_user)}"
            turn_user += 1
        else:
            label = f"### Assistant Response: {str(turn_resp)}"
            turn_resp += 1
        text = f'''
            {label}
            {conversation["value"]}
                
            '''
        str_conversations += text
       
            
    return str_conversations

def input_quality_rating_mt(conversations): 
    str_conversation = apply_conversation_template(conversations)
     
    user_message = f''' 
    
# Instruction

You will be given a conversation between a User and an AI assistant.

Your evaluation should focus on the *User* questions only. 
You need to rate the quality of each user question based on its clarity, specificity, and coherence.

The rating scale is as follows:
- very poor: The query is unclear, vague, or incoherent. It lacks essential information and context.
- poor: The query is somewhat unclear or lacks important details. It requires significant clarification.
- average: The query is moderately clear and specific. It may require some additional information for a complete understanding.
- good: The query is clear, specific, and mostly well-formed. It provides sufficient context for understanding the user\'s intent.
- excellent: The query is very clear, specific, and well-articulated. It contains all the necessary information and context for providing a comprehensive response.

## Conversation
```
{str_conversation}
```

## Output Format
Given the conversation, you first need to give an assessment, highlighting the strengths and/or weaknesses of each user question. Then, you need to output an overall rating score for all the questions. 

Please output your response in the format below by filling in the placeholders in [...]:

```
{{   
    "explanation": "[...]",
    "input_quality": "[very poor/poor/average/good/excellent]"
}}
```
Make sure to only output a valid json string.
Please, do not add any other characters outside the json string.

'''
    return user_message


def sample_quality_rating_mt(conversations): 
    str_conversation = apply_conversation_template(conversations)
    user_message = f''' 
## Instruction 

You will be given a conversation between a User and an AI assistant.

Please rate the assistant's response to each instruction. In particular, when judging, consider:

- Instruction Following: Does the response directly address the question?
- Accuracy: Is the information provided in the response correct?
- Presentation: Is the response logically structured and easy to understand?

## Conversation
```
{str_conversation}
```
You first need to provide an assessment that highlights the strengths and/or weaknesses of each assistant response. Then, please provide an overall score on a scale from 0 to 5 to all the responses provided by the assistant, where a higher score indicates a higher overall quality. 

## Output Format
Please provide your response in the following format, by filling in the placeholders in [...]:

```
{{   
    "score": "[1,2,3,4,5]",
    "explanation": "[...]"
}}
```

Make sure to only output a valid json string.
Please, do not add any other characters outside the json string.
'''
    return user_message

def conversations_quality_rating_mt(conversations): 
    str_conversation = apply_conversation_template(conversations)
    user_message = f''' 
## Instruction 

You will be given a conversation between a User and an AI assistant.

Please rate the quality of the whole conversation. In particular, when judging, consider:

- Question Clarity: Does the user formulate only one, clear, specific, and coherent question? 
- Question Presentation: Is the question logically structured, easy to understand, and in a natural language format?
- Response Relevance: Does the response directly address the question?
- Response Accuracy: Is the information provided in the response correct?
- Response Presentation: Is the response logically structured and easy to understand?
```
{str_conversation}
```
You first need to provide a comprehensive assessment that highlights the strengths and/or weaknesses of all the user questions and their respective responses. Then, please provide an overall score on a scale from 0 to 5 to the whole conversation, where a higher score indicates a higher overall quality. 

## Output Format
After providing your explanation, output your 
overall score in the following format by strictly filling in the placeholders in [...]:

```
{{
    "score": "[1,2,3,4,5]",
    "explanation": "[...]"
}}
```

Make sure to output *ONLY ONE*  valid json string.
Please, *DO NOT ADD* any other characters outside the json string.
'''
    return user_message
    
# For statistcal analysis
MATH_KEYWORDS = [
    "radius", "diameter", "circumference", "perimeter", "triangle", "rectangle",
    "algebra", "geometry", "calculus", "trigonometry", "set theory", "linear regression",
    "number theory", "graph theory", "topology", "differential equations", "integral", "derivative",
    "logarithm", "exponent", "polynomial", "quadratic", "vector", "matrix", "determinant", "eigenvalue",
    "eigenvector", "complex number", "real number", "rational number", "irrational number", "prime number",
    "factorial", "permutation", "combination", "binomial", "continuity", "eigen vector",
    "domain", "range", "inverse", "composition", "angle", "radian", "sine",
    "cosine", "tangent", "cotangent", "secant", "cosecant", "pythagorean theorem", "euclidean geometry",
    "non-euclidean geometry", "hyperbolic geometry", "elliptic geometry", "fractal", "chaos theory",
    "golden ratio", "fibonacci sequence", "pascal's triangle", "binomial theorem", "fourier series",
    "laplace transform", "z-transform", "normal distribution", "poisson distribution", "chi-square distribution",
    "Student's t-distribution", "confidence interval", "hypothesis testing", "p-value", "bayes' theorem",
    "law of large numbers", "central limit theorem", "regression analysis", "correlation coefficient",
    "covariance", "ANOVA", "chi-square test", "t-test", "F-test", "Mann-Whitney U test", "Kolmogorov-Smirnov test",
    "Wilcoxon signed-rank test", "Kruskal-Wallis test", "Fisher's exact test", "McNemar's test", "Spearman's rank correlation",
    "Kendall's tau", "Pearson's correlation", "linear programming", "integer programming", "dynamic programming",
]

CODE_KEYWORDS = [
    "Python", "Java", "in C", "C++", "C#", "JavaScript", "Ruby", "PHP", "Swift", "Objective-C",
    "Kotlin", "in Go", "Rust", "Lua", "Perl", "MATLAB", "Fortran", "Julia", "TypeScript", 
    "F#", "Bash", "PowerShell", "SQL", "HTML", "CSS", "XML", "YAML", "JSON", 
    "Markdown", "LaTeX", "Assembly", "VBA", "Pascal",
    "in D", "in R", "Solidity", "Verilog", "VHDL", "Visual Basic", "BASIC",
    "\n\n```", "\n```", "```", "import ", "def", "elif", "#include"
]
