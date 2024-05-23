import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import gc

# Load the model and tokenizer with the specified attention implementation
model_name = '/root/autodl-tmp/llama2/large/model/'
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True, attn_implementation="eager")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Get the vocabulary size
vocab_size = tokenizer.vocab_size
print(f"Vocabulary size: {vocab_size}")

# Check if a CUDA-enabled GPU is available
torch.cuda.set_device(1)  # Use GPU 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to half precision and then to the appropriate device
if torch.cuda.is_available():
    model.half()
model.to(device)

# Function to visualize attention weights for a specific iteration and specific layers

def visualize_attention(attentions, iteration, token_ids, tokenizer, layers_to_print, heads_per_row=2, heads_to_print=None):
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    for layer_num, layer_attentions in enumerate(attentions):
        if layer_num not in layers_to_print:
            continue
        
        num_heads = layer_attentions.size(1)  # Get the number of heads from the attention shape
        heads_to_print = heads_to_print if heads_to_print is not None else list(range(num_heads))
        
        total_rows = (len(heads_to_print) + heads_per_row - 1) // heads_per_row  # Calculate how many rows are needed
        
        # Loop over each set of heads
        for row in range(total_rows):
            start_index = row * heads_per_row
            end_index = min(start_index + heads_per_row, len(heads_to_print))
            selected_heads = heads_to_print[start_index:end_index]

            fig, axes = plt.subplots(1, len(selected_heads), figsize=(20, 5))
            if len(selected_heads) == 1:
                axes = [axes]  # Make axes iterable if only one plot

            # Plot each head in the current set
            for i, head_num in enumerate(selected_heads):
                ax = axes[i]
                head_attention = layer_attentions[0, head_num]  # Select the first in batch and the current head
                im = ax.imshow(head_attention.detach().cpu().numpy(), cmap='viridis')
                ax.set_title(f'Iter {iteration}, Layer {layer_num + 1}, Head {head_num + 1}')
                ax.set_xticks(range(len(tokens)))
                ax.set_yticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=90)
                ax.set_yticklabels(tokens)

            fig.colorbar(im, ax=axes, orientation='horizontal')
            plt.show()



# Function to generate and print model's response with selective attention visualization
def generate_response_with_attention(prompt, model, tokenizer, max_length=100, temperature=0.7, top_k=50, top_p=0.9, visualize_every_n=5, layers_to_print=[0, 39]):
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs.input_ids.to(device)  # Ensure input_ids are on the appropriate device
    response_ids = input_ids

    for i in range(max_length):
        outputs = model(input_ids, output_attentions=True)
        attentions = outputs.attentions
        logits = outputs.logits
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        
        # Visualize attention weights for initial input, every nth token, and the final token
        if i == 0 or (i + 1) % visualize_every_n == 0 or next_token_id == tokenizer.eos_token_id:
            visualize_attention(attentions, i + 1, response_ids[0], tokenizer, layers_to_print, heads_per_row=3, heads_to_print=[0, 15, 39])
        
        # Append the next token and update input_ids
        response_ids = torch.cat([response_ids, next_token_id.to(device)], dim=-1)  # Ensure next_token_id is on the appropriate device
        input_ids = response_ids

        # Stop if the end-of-sequence token is generated
        if next_token_id == tokenizer.eos_token_id:
            break

    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return response



# Main Entry of Program

while True:
    try:
        print()
        prompt = input("Enter your input prompt (or type 'exit' to quit): ")
        if prompt.lower() == 'exit':
            break
        response = generate_response_with_attention(prompt, model, tokenizer)
        print(f"Model's response: {response}")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")
        break

print()

clear = input('Do you wish to clear model from GPU?')
if clear == 'y':
    del model
    torch.cuda.empty_cache()
    gc.collect()
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
                del obj
        except:
            pass

    torch.cuda.empty_cache()
    gc.collect()
    
    
print()
print("Exited the loop. Goodbye!")