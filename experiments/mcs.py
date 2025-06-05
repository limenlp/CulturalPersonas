import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch.nn.functional as F
import argparse
from huggingface_hub import login
import pandas as pd
import openai
import random 
import numpy as np 
from metrics import calc_kl, calc_ks

# Avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def test_setup(test_type, country_norms, questions_file, country):
    norm1, norm2 = country_norms[0], country_norms[1]

    if test_type == "standard":
        questions = pd.read_csv(questions_file)['Text'].values
        prompt_template = f"""
        You are someone who was born and brought up in {country}.
        You are very familiar with {country}'s culture and traditions and practice many of them, two of which include: {norm1} and {norm2}.
        Please read the following statement carefully and choose the answer that best describes you.
        Question: I {{question}}
        Answers: 
            a: Very inaccurate 
            b: Moderately inaccurate 
            c: Neither accurate nor inaccurate 
            d: Moderately accurate 
            e: Very accurate
        Please respond with ONLY a, b, c, d, or e.
        """
        prompts = [prompt_template.format(question=q) for q in questions]
        return prompts, ["a", "b", "c", "d", "e"]

    elif test_type == "trait":
        df = pd.read_csv(questions_file)
        prompt_template = f"""
        You are someone who was born and brought up in {country}.
        You are very familiar with {country}'s culture and traditions and practice many of them, two of which include: {norm1} and {norm2}.
        Please read the following statement carefully and choose the answer that best describes you.
        Question: {{situation}} {{question}}
        Answers:
            a: {{high1}}
            b: {{low1}}
            c: {{high2}}
            d: {{low2}}
        Please respond with ONLY a, b, c, d.
        """
        prompts = [
            prompt_template.format(
                situation=row['situation'], question=row['question'],
                high1=row['high1'], low1=row['low1'],
                high2=row['high2'], low2=row['low2']
            )
            for _, row in df.iterrows()
        ]
        return prompts, ["a", "b", "c", "d"]

    elif test_type == "cp":
        df = pd.read_csv(questions_file)
        prompt_template = f"""
        You are someone who was born and brought up in {country}.
        You are very familiar with {country}'s culture and traditions and practice many of them, two of which include: {norm1} and {norm2}.
        Please read the following statement carefully and choose the answer that best describes you.
        Question: {{scenario}} {{question}}
        Answers:
            a: {{mod_high}}
            b: {{low}}
            c: {{high}}
            d: {{med}}
            e: {{mod_low}}
        Please respond with ONLY a, b, c, d, e.
        """
        prompts = [
            prompt_template.format(
                scenario=row['scenario_text'], question=row['question'],
                high=row['high'], mod_high=row['moderately_high'],
                med=row['medium'], mod_low=row['moderately_low'],
                low=row['low']
            )
            for _, row in df.iterrows()
        ]
        return prompts, ["a", "b", "c", "d", "e"]

    else:  # assume test_type == "big5chat"
        df = pd.read_csv(questions_file)
        prompt_template = f"""
        You are someone who was born and brought up in {country}.
        You are very familiar with {country}'s culture and traditions and practice many of them, two of which include: {norm1} and {norm2}.
        Imagine you are a human. 
        Please read the following dialogue carefully and choose how you might best respond.
        Question: {{dialogue}}
        Answers:
            a: {{high}}
            b: {{low}}
        Please respond with ONLY a or b.
        """
        prompts = [
            prompt_template.format(
                dialogue=row['train_input'],
                high=row['high_output'],
                low=row['low_output']
            )
            for _, row in df.iterrows()
        ]
        return prompts, ["a", "b"]

def generate_gpt4o_response(prompt, options, model="gpt-4o"):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=1,
        temperature=0.0,
        logprobs=5,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    top_logprobs = response['choices'][0]['logprobs']['top_logprobs'][0]
    result = {}
    for opt in options:
        key = " " + opt  # OpenAI logprobs keys include the leading space
        result[f"{opt}_logprob"] = top_logprobs.get(key, float('-inf'))
    result["prompt"] = prompt
    return result

def get_samples(data, reverse, traits, test_type, samples=100):
    o_, c_, e_, a_, n_ = [], [], [], [], []

    if test_type in ["standard", "cp"]:
        probs = ['a_prob', 'b_prob', 'c_prob', 'd_prob', 'e_prob']
    elif test_type in ["trait"]: 
        probs = ['a_prob', 'b_prob', 'c_prob', 'd_prob'] 
    else: 
        probs = ['a_prob', 'b_prob']

    for i, row in data.iterrows():
        question_probs = np.exp(row[probs].values).tolist()
        if test_type == "standard":
            if reverse[i] == "F":
                sampled_answers = random.choices(population=[1, 2, 3, 4, 5], weights=question_probs, k=samples)
            else:
                sampled_answers = random.choices(population=[5, 4, 3, 2, 1], weights=question_probs, k=samples)
        elif test_type == "trait":
            sampled_answers = random.choices(population=[5, 1, 5, 1], weights=question_probs, k=samples)
        elif test_type == "cp":
            sampled_answers = random.choices(population=[4, 1, 5, 3, 2], weights=question_probs, k=samples)
        else: #big5chat 
            sampled_answers = random.choices(population=[5, 1], weights=question_probs, k=samples)

        trait = traits[i]
        if trait == 'O':
            o_.extend(sampled_answers)
        elif trait == 'C':
            c_.extend(sampled_answers)
        elif trait == 'E':
            e_.extend(sampled_answers)
        elif trait == 'A':
            a_.extend(sampled_answers)
        else:
            n_.extend(sampled_answers)

    def avg_trait(trait_list):
        arr = np.array(trait_list).reshape(-1, samples)  # shape: (num_questions, samples)
        return np.mean(arr, axis=0)

    return (
        avg_trait(o_).tolist(), avg_trait(c_).tolist(), avg_trait(e_).tolist(),
        avg_trait(a_).tolist(), avg_trait(n_).tolist()
    )

def generate_model_dist(args.out_file, reverse_values, traits, test_type, args.ground_truth, country_code):
    data = pd.read_csv(args.out_file)
    ground_truth = pd.read_csv(args.ground_truth)
    c_gt = ground_truth[ground_truth.country == country_code]
    o, c, e, a, n = get_samples(data, reverse_values, traits, test_type, samples=len(c_gt))

    sample_df = pd.DataFrame() 
    sample_df['O'] = o 
    sample_df['C'] = c 
    sample_df['E'] = e 
    sample_df['A'] = a 
    sample_df['N'] = n 

    return sample_df

def generate_metrics(country_code, model_scores, gt):
    divergences = calc_kl(gt, model_scores, country_code)
    ks_stats, ks_pvals = calc_ks(gt, model_scores, country_code)

    df = pd.DataFrame(
        [divergences, ks_stats, ks_pvals],
        index=['KL Divergence', 'KS Stat', 'KS P-Value'],
        columns=['O', 'C', 'E', 'A', 'N']
    )
    return df


def main():
    hf_token = os.getenv("HF_TOKEN")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment variables.")
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    login(token=hf_token)

    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--questions', default='ipip-120.csv')
    parser.add_argument('-t', '--test_type', default="standard")
    parser.add_argument('-c', '--country', default="USA", choices=["usa", "brazil", "southafrica", 
                                                                     "saudiarabia", "india", "japan"])
    # parser.add_argument('-o', '--gen_dist', default="llama_model_dist.csv", help="location to save generated model OCEAN distribution") 
    parser.add_argument('-r', '--results_file', default="results_USA_llama.csv", help="file to store metrics (W, KL, KS)")
    parser.add_argument('-n', '--norm_file', default="country_norms.csv")
    parser.add_argument('-m', '--model', default="llama", choices=["llama", "qwen", "gpt4o"])
    parser.add_argument('-gt', '--ground_truth', default="datasets/ground-truth/big-five-ocean.csv")
    args = parser.parse_args()

    country_norms = pd.read_csv(args.norm_file)
    country_norms = country_norms[country_norms.country.str.strip() == args.country].norm.values
    if len(country_norms) < 2:
        raise ValueError(f"Not enough norms found for {args.country}")
    
    reverse = pd.read_csv(args.questions)
    if args.test_type == "standard":
        reverse_values = reverse.Reverse.values
        traits = reverse.Key.values 
    else:
        reverse_values = []
        traits = reverse.trait.values 

    traits = [t[0].upper() for t in traits]
    country_map = {
        "usa": "USA", "brazil": "Brazil", "india": "India", "japan": "Japan",
        "saudiarabia": "Saudi Arab", "southafrica": "South Afri"
    }
    country_code = country_map[args.country]

    prompts, options = test_setup(args.test_type, country_norms, args.questions, args.country)
    rows = []

    if args.model == "gpt4o":
        for prompt in prompts:
            row = generate_gpt4o_response(prompt, options)
            rows.append(row)
    else:
        if args.model == "llama":
            model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            model_name = "Qwen/Qwen2-7B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        tokenizer.pad_token = tokenizer.eos_token
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=(args.model == "qwen")
        )
        model.eval()
        option_token_ids = [tokenizer(o, add_special_tokens=False).input_ids[0] for o in options]

        batch_size = 16
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                next_token_logits = outputs.logits[:, -1, :]
            for j, prompt in enumerate(batch_prompts):
                row = {"prompt": prompt}
                logits_i = next_token_logits[j, option_token_ids]
                logprobs_i = torch.log_softmax(logits_i, dim=-1)
                for opt, logprob in zip(options, logprobs_i):
                    row[f"{opt}_prob"] = logprob.item()
                rows.append(row)
            torch.cuda.empty_cache()

    # pd.DataFrame(rows).to_csv(args.out_file, index=False)

    # generate the model distribution
    samples = generate_model_dist(args.out_file, reverse_values, traits, args.test_type, args.ground_truth, country_code)
    # samples.to_csv(args.gen_dist, index=False)

    # generate final metric scores 
    gt = pd.read_csv(args.ground_truth)
    metrics = generate_metrics(country_code, samples, gt) 
    metrics.to_csv(args.results_file)

if __name__ == '__main__':
    main()