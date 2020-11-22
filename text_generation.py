import sys, os
import warnings
warnings.filterwarnings("ignore")

def generate_output_text(input_text, generator_pipeline):
    result = generator_pipeline(input_text, pad_token_id=50256)[0]['generated_text']
    split_result = result.split('.')
    split_result = split_result[:-1] #remove trailing text with no period at the end
    final_result = ''
    for text in split_result:
        final_result += text + '.'

    final_result = final_result.replace('\n', ' ')
    final_result = final_result.replace('>', '')
    final_result = final_result.replace('"', '')
    return final_result