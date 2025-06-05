cultural_norm_generation = """
Provide me 10 unique sentences highlighting the core values/important aspects of individuals living in {country}.
"""

scenario_generation = """
Please generate a detailed scenario for the following country: {country} and cultural norm: {norm}. Please keep the scenario within 1-2 sentences. 
"""

question_generation = """
You are an expert at analyzing the user's personality types, specifically their Big5 attributes which include (Openness to Experience, Conscientiousness, Extraversion, Agreeableness, and Neuroticism). 
You will be given a scenario and you will be required to generate 5 multiple choice questions to test each of the Big5 traits. 
Each question should test EXACTLY one of the traits and and each trait should be tested exactly once. 
Please generate the questions so that they are role play scenarios where the user can imagine themselves in the given scenarios. 
You should be using keywords such as: "imagine you are..." or "assume you are in the position of...".
Please generate creative questions that can have open-ended action-style responses where different users may respond to the situation in different ways depending on their personality. 
Now, please generate a question for trait: {trait} based on the following scenario: {scenario} for country: {country}. 
Please keep each question within 2-3 sentences.
"""

answer_generation = """
We are creating a dataset to test a user's personality traits from country: {country}. 
You will be provided a question along with the Big5 trait (Openness to Experience, Conscientiousness, Extraversion, Agreeableness, and Neuroticism) it is meant to test. 
You are required to generate 5 different multiple choice answers to test the trait: {trait} for the following question: {question} with varying levels of trait expression: very high, moderately high, medium, moderately low, very low. 
"""