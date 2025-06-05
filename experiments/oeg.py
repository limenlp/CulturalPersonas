import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch.nn.functional as F
import argparse
from huggingface_hub import login
import pandas as pd
import openai
import numpy as np
from sentence_transformers import SentenceTransformer, util
import random
from metrics import calc_kl, calc_ks

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def test_setup(test_type, country_norms, questions_file, country):
    norm1, norm2 = country_norms[0], country_norms[1]
    df = pd.read_csv(questions_file)

    if test_type == "trait":
        prompt_template = f"""
        You are someone who was born and brought up in {country}.
        You are very familiar with {country}'s culture and traditions and practice many of them, two of which include: {norm1} and {norm2}.
        Please read the following statement carefully.
        Question: {{situation}} {{question}}
        Please respond with how you would react in EXACTLY 1 sentence.
        """
        prompts = [
            prompt_template.format(
                situation=row['situation'], question=row['question']
            ) for _, row in df.iterrows()
        ]
        answer_texts = [
            [row['high1'], row['low1'], row['high2'], row['low2']]
            for _, row in df.iterrows()
        ]
        return prompts, ["a", "b", "c", "d"], answer_texts

    elif test_type == "cp":
        prompt_template = f"""
        You are someone who was born and brought up in {country}.
        You are very familiar with {country}'s culture and traditions and practice many of them, two of which include: {norm1} and {norm2}.
        Please read the following statement carefully.
        Question: {{scenario}} {{question}}
        Please respond with how you would react in EXACTLY 1 sentence.
        """
        prompts = [
            prompt_template.format(
                scenario=row['scenario_text'], question=row['question']
            ) for _, row in df.iterrows()
        ]
        answer_texts = [
            [row['moderately_high'], row['low'], row['high'], row['medium'], row['moderately_low']]
            for _, row in df.iterrows()
        ]
        return prompts, ["a", "b", "c", "d", "e"], answer_texts

    else:
        prompt_template = f"""
        You are someone who was born and brought up in {country}.
        You are very familiar with {country}'s culture and traditions and practice many of them, two of which include: {norm1} and {norm2}.
        Imagine you are a human.
        Please read the following dialogue carefully.
        Question: {{dialogue}}
        Please respond with how you would respond in EXACTLY 1 sentence.
        """
        prompts = [
            prompt_template.format(
                dialogue=row['train_input']
            ) for _, row in df.iterrows()
        ]
        answer_texts = [
            [row['high_output'], row['low_output']]
            for _, row in df.iterrows()
        ]
        return prompts, ["a", "b"], answer_texts

def generate_gpt4o_response(prompt, model="gpt-4o"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        top_p=1.0,
        max_tokens=50
    )
    return {"prompt": prompt, "response": response["choices"][0]["message"]["content"].strip()}

def get_samples(data, reverse, traits, test_type, samples=100):
    o_, c_, e_, a_, n_ = [], [], [], [], []
    if test_type == "cp":
        probs = ['sim_a', 'sim_b', 'sim_c', 'sim_d', 'sim_e']
        population = [4, 1, 5, 3, 2]
    elif test_type == "trait":
        probs = ['sim_a', 'sim_b', 'sim_c', 'sim_d']
        population = [5, 1, 5, 1]
    else:
        probs = ['sim_a', 'sim_b']
        population = [5, 1]

    for i, row in data.iterrows():
        weights = row[probs].values.astype(float).tolist()
        sampled_answers = random.choices(population=population, weights=weights, k=samples)
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
        arr = np.array(trait_list).reshape(-1, samples)
        return np.mean(arr, axis=0)

    return (
        avg_trait(o_).tolist(), avg_trait(c_).tolist(), avg_trait(e_).tolist(),
        avg_trait(a_).tolist(), avg_trait(n_).tolist()
    )

def generate_model_dist(out_file, reverse_values, traits, test_type, ground_truth_file, country_code):
    data = pd.read_csv(out_file)
    ground_truth = pd.read_csv(ground_truth_file)
    c_gt = ground_truth[ground_truth.country == country_code]
    o, c, e, a, n = get_samples(data, reverse_values, traits, test_type, samples=len(c_gt))
    return pd.DataFrame({"O": o, "C": c, "E": e, "A": a, "N": n})

def generate_metrics(gt, model_scores, country_code): 
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
    parser.add_argument('-t', '--test_type', default="trait")
    parser.add_argument('-c', '--country', default="USA", choices=["usa", "brazil", "southafrica", "saudiarabia", "india", "japan"])
    parser.add_argument('-o', '--out_file', default="model_outputs.csv")
    parser.add_argument('-o2', '--gen_dist', default="llama_model_dist.csv")
    parser.add_argument('-n', '--norm_file', default="country_norms.csv")
    parser.add_argument('-m', '--model', default="llama", choices=["llama", "qwen", "gpt4o"])
    parser.add_argument('-gt', '--ground_truth', default="datasets/ground-truth/big-five-ocean.csv")
    args = parser.parse_args()

    country_norms = pd.read_csv(args.norm_file)
    country_norms = country_norms[country_norms.country.str.strip().str.lower() == args.country.lower()].norm.values
    if len(country_norms) < 2:
        raise ValueError(f"Not enough norms found for {args.country}")

    reverse_df = pd.read_csv(args.questions)
    if args.test_type == "standard":
        reverse_values = reverse_df.Reverse.values
        traits = reverse_df.Key.values
    else:
        reverse_values = []
        traits = reverse_df.trait.values
    traits = [t[0].upper() for t in traits]

    country_map = {
        "usa": "USA", "brazil": "Brazil", "india": "India", "japan": "Japan",
        "saudiarabia": "Saudi Arab", "southafrica": "South Afri"
    }
    country_code = country_map[args.country]

    prompts, options, all_answer_choices = test_setup(args.test_type, country_norms, args.questions, args.country)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    rows = []

    if args.model == "gpt4o":
        for i, prompt in enumerate(prompts):
            row = generate_gpt4o_response(prompt)
            generated = row["response"]
            choices = all_answer_choices[i]
            emb_gen = embedder.encode(generated, convert_to_tensor=True)
            emb_opts = embedder.encode(choices, convert_to_tensor=True)
            sims = util.cos_sim(emb_gen, emb_opts)[0]
            sim_probs = F.softmax(sims, dim=0)
            best_idx = torch.argmax(sim_probs).item()

            row["matched_option"] = options[best_idx]
            row["matched_text"] = choices[best_idx]
            row["cosine_sim"] = sim_probs[best_idx].item()
            for opt, sim in zip(options, sim_probs):
                row[f"sim_{opt}"] = sim.item()
            rows.append(row)

    else:
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct" if args.model == "llama" else "Qwen/Qwen2-7B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=(args.model == "qwen"))
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

        batch_size = 8
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False, return_dict_in_generate=True)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for j, (prompt, response) in enumerate(zip(batch_prompts, decoded)):
                generated = response.strip()
                choices = all_answer_choices[i + j]
                emb_gen = embedder.encode(generated, convert_to_tensor=True)
                emb_opts = embedder.encode(choices, convert_to_tensor=True)
                sims = util.cos_sim(emb_gen, emb_opts)[0]
                sim_probs = F.softmax(sims, dim=0)
                best_idx = torch.argmax(sim_probs).item()

                row = {
                    "prompt": prompt,
                    "response": generated,
                    "matched_option": options[best_idx],
                    "matched_text": choices[best_idx],
                    "cosine_sim": sim_probs[best_idx].item(),
                }
                for opt, sim in zip(options, sim_probs):
                    row[f"sim_{opt}"] = sim.item()
                rows.append(row)
            torch.cuda.empty_cache()

    # pd.DataFrame(rows).to_csv(args.out_file, index=False)
    sample_df = generate_model_dist(args.out_file, reverse_values, traits, args.test_type, args.ground_truth, country_code)
    # sample_df.to_csv(args.gen_dist, index=False)


    # generate final metric scores 
    gt = pd.read_csv(args.ground_truth)
    metrics = generate_metrics(country_code, sample_df, gt) 
    metrics.to_csv(args.results_file)

if __name__ == '__main__':
    main()
