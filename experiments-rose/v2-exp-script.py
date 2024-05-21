## VERSION 2: 
## gather logits for more data 

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json

### response not really necessary 
nonadv_prompt_responses = {}
adv_prompt_responses = {}

### LOAD DATA
directory = '../data/Llama2-base/'
json_files = [f for f in os.listdir(directory) if f.endswith('.json') and f.split('_')[0] == 'final']

nonadv_prompts = set()
adv_prompts = set()

for file_name in json_files:
    file_path = os.path.join(directory, file_name)
    print(file_name)
    with open(file_path, 'r') as file:
        
        data = json.load(file)
        adv_prompts.update([value for value in data.values()][0])
        nonadv_prompts.update([key for key in data.keys()])

adv_prompts = list(adv_prompts)
nonadv_prompts = list(nonadv_prompts)


### function for next token logits 
def predict_next_token(model, inp):
    """
    Predict the next token logits using a given model and input.

    Args:
        model (torch.nn.Module): The neural network model to use for prediction.
        inp (dict): A dictionary of input tensors required by the model. Typically includes 'input_ids' and other necessary inputs.

    Returns:
        tuple:
            preds (torch.Tensor): A tensor containing the predicted token indices for each input sequence.
            p (torch.Tensor): A tensor containing the probabilities of the predicted tokens.
    """
    out = model(**inp)["logits"]
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.max(probs, dim=1)
    return preds, p

### function for intervened prompt logits / ie score
def prompt_tokens_ie_score(model, tokenizer, prompt, intervene_token): 
    """
    1. Changes input prompt by 'intervene_token' 
    2. Calculates the intervened prompt logits for each token in the input prompt.
    
    Args:
        model: The language model.
        tokenizer: The tokenizer used for text preprocessing.
        prompt: The input prompt as a string.
        intervene_token: The token to substitute for each token in the prompt.
        
    Returns:
        prompt_logits_list: A list of logits for each intervened prompt.
        prompt_ie_list: A list of indirect effect scores for each token in the prompt.
    """
    ## independent variable: character to change prompt tokens to
    intervened_token_num = tokenizer(intervene_token)['input_ids'][0]

    # generate original logits
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    pred, p = predict_next_token(model, inputs)
    ref_prob = (pred, p)

    prompt_logits_list = []

    for i in range(len(inputs['input_ids'].flatten())):

        # tokenise inputs
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        # change prompts with new token
        inputs['input_ids'].flatten()[i] = intervened_token_num
        new_inp = inputs

        # get prompt logits 
        preds, p = predict_next_token(model, new_inp)
        prompt_logits_list.append((preds, p))

    ## record prompt indirect effect scores
    prompt_ie_list = [abs(ref_prob[1].item() - x[1].item()) for x in prompt_logits_list]

    return prompt_ie_list


## load model 
tokenizer = AutoTokenizer.from_pretrained(
    "/root/autodl-tmp/llama2/base/model", torch_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(
    "/root/autodl-tmp/llama2/base/model", torch_dtype=torch.float16
)

model.eval()
model.to("cuda")

for i in range(len(nonadv_prompts)):
    
    # inputs 
    inputs = tokenizer(nonadv_prompts[i], return_tensors="pt").to("cuda")

    ## response
    outputs = model.generate(**inputs)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f'Harmful prompt {i}: {nonadv_prompts[i]}, ', generated_text)

    ## prompt logits with intervened token
    prompt_tokens_ie_list = prompt_tokens_ie_score(model, tokenizer, nonadv_prompts[i], '-')

    nonadv_prompt_responses[nonadv_prompts[i]] = {'response': generated_text}
    nonadv_prompt_responses[nonadv_prompts[i]]['prompt_ie_score'] = prompt_tokens_ie_list
    # TODO: nonadv_prompt_responses[nonadv_prompts[i]]['prompt_logits'] = prompt_tokens_logits_list
    
    # print(nonadv_prompt_responses)
    with open('harmful_data_w_scores_2.json', 'w') as json_file:
        json.dump(nonadv_prompt_responses, json_file, indent=4)   

    print('\n\n saved response')


for i in range(len(adv_prompts)):
    ### adv data ###

    inputs = tokenizer(adv_prompts[i], return_tensors="pt")
    inputs.to("cuda")

    ## response
    outputs = model.generate(**inputs)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f'adversarial prompt {i}: {adv_prompts[i]}, ', generated_text)
    
    ## prompt logits with intervened token
    prompt_tokens_ie_list = prompt_tokens_ie_score(model, tokenizer, nonadv_prompts[i], '-')

    adv_prompt_responses[adv_prompts[i]] = {'response': generated_text}
    adv_prompt_responses[adv_prompts[i]]['prompt_ie_score'] = prompt_tokens_ie_list
    # TODO: adv_prompt_responses[adv_prompts[i]]['prompt_logits'] = prompt_tokens_logits_list
    
    # print(adv_prompt_responses)
    with open('adversarial_data_w_scores_2.json', 'w') as json_file:
        json.dump(adv_prompt_responses, json_file, indent=4)  

    print('\n\n saved response') 

