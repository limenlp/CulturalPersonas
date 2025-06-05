import os
import re
import json
import argparse
import openai
from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import BaseRetriever, HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory

from prompts import norm_generation, scenario_generation, question_generation, answer_generation
from model_checks_prompts import scenario_check, question_check1, question_check2, answer_check1, answer_check2

def load_documents(root_directory: str) -> List:
    documents = []
    for folder, _, files in os.walk(root_directory):
        for file in files:
            file_path = os.path.join(folder, file)
            try:
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    return documents

def create_retriever(documents: List, api_key: str) -> BaseRetriever:
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)
    vector_db = FAISS.from_documents(split_docs, embeddings)
    return vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

def query_gpt4(client, memory, retriever, user_query: str, record: bool = True) -> str:
    context = "\n\n".join([doc.page_content for doc in retriever.get_relevant_documents(user_query)])
    chat_history = memory.chat_memory.messages

    prompt = (
        "You are a helpful assistant. Use the following context to answer the question:\n\n"
        f"Context:\n{context}\n\n"
        "Conversation History:\n" +
        "\n".join([
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}" for m in chat_history
        ]) + f"\n\nUser: {user_query}\nAssistant:"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    reply = response.choices[0].message.content.strip()
    if record:
        memory.chat_memory.add_user_message(user_query)
        memory.chat_memory.add_ai_message(reply)
    return reply

def validate(client, memory, retriever, check_prompt: str) -> bool:
    response = query_gpt4(client, memory, retriever, check_prompt, record=False)
    return re.match(r'^yes\b', response.strip().lower()) is not None

def generate_norms(client, memory, retriever, country: str) -> List[Dict]:
    prompt = norm_generation.format(country=country)
    response = query_gpt4(client, memory, retriever, prompt)
    norms = [
        {"id": idx + 1, "text": line.strip()}
        for idx, line in enumerate(response.split("\n"))
        if line.strip() and (line[0].isdigit() or line.startswith("-"))
    ]
    return norms

def generate_scenarios(client, memory, retriever, norms: List[Dict], country: str) -> List[Dict]:
    scenarios = []
    for norm in norms:
        attempts = 0
        max_attempts = 5
        while attempts < max_attempts:
            prompt = scenario_generation.format(norm=norm['text'], country=country)
            response = query_gpt4(client, memory, retriever, prompt)
            scenario_lines = [
                line.strip() for line in response.split("\n")
                if line.strip() and line[0].isdigit() and line[1] in [".", ")"]
            ]
            valid_lines = [s for s in scenario_lines if validate(client, memory, retriever, scenario_check.format(scenario=s, norm=norm['text']))]
            if len(valid_lines) == len(scenario_lines):
                for line in valid_lines:
                    scenarios.append({
                        "id": len(scenarios) + 1,
                        "norm_id": norm["id"],
                        "norm_text": norm["text"],
                        "text": line
                    })
                break
            attempts += 1
    return scenarios

def generate_questions(client, memory, retriever, scenarios: List[Dict], country: str) -> List[Dict]:
    traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    questions = []
    for scenario in scenarios:
        for trait in traits:
            attempts = 0
            max_attempts = 5
            while attempts < max_attempts:
                prompt = question_generation.format(trait=trait, scenario=scenario["text"], country=country)
                response = query_gpt4(client, memory, retriever, prompt)
                check1 = question_check1.format(question=response, scenario=scenario["text"])
                check2 = question_check2.format(question=response, trait=trait)
                if validate(client, memory, retriever, check1) and validate(client, memory, retriever, check2):
                    questions.append({
                        "id": len(questions) + 1,
                        "scenario_id": scenario["id"],
                        "scenario_text": scenario["text"],
                        "trait": trait,
                        "text": response.strip()
                    })
                    break
                attempts += 1
    return questions

def generate_answers(client, memory, retriever, questions: List[Dict], country: str) -> List[Dict]:
    answers = []
    for question in questions:
        attempts = 0
        max_attempts = 5
        while attempts < max_attempts:
            prompt = answer_generation.format(trait=question["trait"], question=question["text"], country=country)
            response = query_gpt4(client, memory, retriever, prompt)
            answer_lines = [line.strip() for line in response.split("\n") if line.strip()]
            if len(answer_lines) != 5:
                attempts += 1
                continue
            answers_preview = " ".join(answer_lines)
            check1 = answer_check1.format(answers=answers_preview, question=question["text"])
            check2 = answer_check2.format(answers=answers_preview, trait=question["trait"])
            if validate(client, memory, retriever, check1) and validate(client, memory, retriever, check2):
                for line in answer_lines:
                    answers.append({
                        "question_id": question["id"],
                        "question_text": question["text"],
                        "trait": question["trait"],
                        "text": line
                    })
                break
            attempts += 1
    return answers

def save_json(data: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate norms, scenarios, questions, and answers from PDFs.")
    parser.add_argument("--root_directory", required=True)
    parser.add_argument("--country", required=True)
    parser.add_argument("--norms_file", required=True)
    parser.add_argument("--scenarios_file", required=True)
    parser.add_argument("--questions_file", required=True)
    parser.add_argument("--answers_file", required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    api_key = os.getenv("OPEN_AI_API")
    client = openai.OpenAI(api_key=api_key)
    documents = load_documents(args.root_directory)
    retriever = create_retriever(documents, api_key)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    norms = generate_norms(client, memory, retriever, args.country)
    scenarios = generate_scenarios(client, memory, retriever, norms, args.country)
    questions = generate_questions(client, memory, retriever, scenarios, args.country)
    answers = generate_answers(client, memory, retriever, questions, args.country)

    save_json({"country": args.country, "norms": norms}, args.norms_file)
    save_json({"country": args.country, "scenarios": scenarios}, args.scenarios_file)
    save_json({"country": args.country, "questions": questions}, args.questions_file)
    save_json({"country": args.country, "answers": answers}, args.answers_file)

if __name__ == "__main__":
    main()
