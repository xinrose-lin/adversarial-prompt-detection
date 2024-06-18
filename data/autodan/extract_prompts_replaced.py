import json 

with open("autodan_hga/llama2_0_normal.json", 'r') as f: 
    data = json.load(f)

def filter_goal_and_target(original_dict):
    filtered_dict = {}
    for key, value in original_dict.items():
        # following autodan code repo: utils/string_utils.py line 28
        final_suffix = value['final_suffix'].replace('[REPLACE]', value['goal'].lower())
        filtered_dict[key] = {
            'goal': value['goal'],
            'final_suffix': final_suffix, 
            'attack_is_success': value['is_success']
            }
    return filtered_dict

# Get the filtered dictionary
filtered_dict = filter_goal_and_target(data)

with open('filtered_results.json', 'w') as json_file:
    json.dump(filtered_dict, json_file, indent=4)

print("Filtered dictionary written to filtered_results.json")
