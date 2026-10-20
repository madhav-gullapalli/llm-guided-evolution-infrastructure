import os
import sys
import re
import time
import glob
import numpy as np
import transformers
import argparse
from pathlib import Path

# Ensure repo root is on sys.path so `src` imports work even when launched from nested dirs
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cfg.constants import *
from utils.print_utils import box_print

from llm_utils import (split_file, submit_mixtral, submit_mixtral_hf, 
                       llm_code_qc, str2bool, generate_augmented_code, 
                       extract_note, clean_code_from_llm, retrieve_base_code,
                       fill_prompt_template, active_block_indices, block_note,
                       reject_reason, NEW_GENE_ATTEMPTS)

def augment_network(input_filename='network.py', output_filename='network_x.py', template_txt=None,
                    top_p=0.15, llm_model=LLM_DEEPSEEK, temperature=0.1, apply_quality_control=False):
    
    print(f'Loading {input_filename} code')
    print('Using')
    parts = split_file(input_filename)
    # prompt_templates = glob.glob(f'{ROOT_DIR}/templates/FixedPrompts/*/*.txt')
    # template_path = np.random.choice(prompt_templates)
    # template_path = f'{ROOT_DIR}/templates/{fname}'

    fname = os.path.join(ROOT_DIR, template_txt)
    with open(fname, 'r') as file:
        template_txt = file.read()

    for attempt in range(1, NEW_GENE_ATTEMPTS + 1):
        # select one of the blocks that actually runs (skips the inactive surrogate family)
        augment_idx = int(np.random.choice(active_block_indices(parts)))
        code2llm = parts[augment_idx]
        # add code to be augmented (escape braces inside the code block)
        txt2llm = block_note(code2llm) + fill_prompt_template(template_txt, code2llm.strip())
        code_from_llm = generate_augmented_code(txt2llm, augment_idx-1, apply_quality_control,
                                                top_p, llm_model, min(temperature + 0.1 * (attempt - 1), 1.0))

        # Do not persist failed LLM outputs (literal "ERROR" / empty) — those produce
        # NameError genes and flood the evaluator queue.
        if (not code_from_llm) or str(code_from_llm).strip() == "ERROR":
            reason = "no valid code from LLM"
        else:
            new_parts = list(parts)
            new_parts[augment_idx] = f"\n{extract_note(code2llm)}{code_from_llm}\n"
            reason = reject_reason(new_parts, [parts])
        if reason is None:
            break
        box_print(f"Mutation attempt {attempt}/{NEW_GENE_ATTEMPTS} rejected: {reason}",
                  print_bbox_len=120, new_line_end=False)
    else:
        box_print(
            "Mutation failed: no acceptable code from LLM; not writing gene file",
            print_bbox_len=120,
            new_line_end=False,
        )
        sys.exit(1)

    parts = new_parts
    # prompt_log = f'# Parent Prompt: {template_path} Root Code: {input_filename}\n'
    # python_network_txt = prompt_log + '# --OPTION--'.join(parts)
    python_network_txt = '# --OPTION--'.join(parts)
    # Write the text to the file
    output_file = Path(output_filename)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    output_file.write_text(python_network_txt)

        
    box_print(f"Python code saved to {os.path.basename(output_filename)}", print_bbox_len=120, new_line_end=False)
    print('Job Done')


    
if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Augment Python Network Script.')

    # Add arguments
    parser.add_argument('input_filename', type=str, help='Input file name')
    parser.add_argument('output_filename', type=str, help='Output file name')
    parser.add_argument('template_txt', type=str, help='Template txt')
    parser.add_argument('--llm_model', type=str, default=False, help='LLM Model Name')
    parser.add_argument('--top_p', type=float, default=0.15, help='Top P value for text generation')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature value for text generation')
    parser.add_argument('--apply_quality_control', type=str2bool, default=False, help='Use LLM QC')

    # Parse the arguments
    args = parser.parse_args()
    

    # Call the function with the parsed arguments
    augment_network(input_filename=args.input_filename,
                    output_filename=args.output_filename,
                    template_txt=args.template_txt,
                    llm_model=args.llm_model,
                    top_p=args.top_p, 
                    temperature=args.temperature,
                    apply_quality_control=args.apply_quality_control,
                   )