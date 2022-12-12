import os
from crel.conv_el import ConvEL

from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"]="1"


file_pretrained = Path("S:\\rel\\bert_conv-td")
s2e_pe_model = Path("S:\\rel\\s2e_ast_onto")
base_url = Path("S:\\rel\\rel_data")

CONFIG = {
            'file_pretrained': str(file_pretrained),
            'base_url': str(base_url),
            's2e_pe_model': str(s2e_pe_model),
            }


def print_results(results):
    for res in results:
        print(f'{res["speaker"][:4]}: {res["utterance"]}')
        if res["speaker"] == 'SYSTEM': continue
        for ann in res['annotations']:
            print('\t', ann)


def test_conv1():
    cel = ConvEL(config=CONFIG)

    example = [
        
        {"speaker": "USER", 
        "utterance": "I think science fiction is an amazing genre for anything. Future science, technology, time travel, FTL travel, they're all such interesting concepts.",}, 

        # System turn should not have mentions or pems
        {"speaker": "SYSTEM", 
        "utterance": "Awesome! I really love how sci-fi storytellers focus on political/social/philosophical issues that would still be around even in the future. Makes them relatable.",},

        {"speaker": "USER", 
        "utterance": "I agree. One of my favorite forms of science fiction is anything related to time travel! I find it fascinating.",},
    ]

    result = cel.annotate(example)
    print_results(result)

def test_conv2():
    cel = ConvEL(config=CONFIG)

    example = [
        {"speaker": "USER", 
        "utterance": "I am allergic to tomatoes but we have a lot of famous Italian restaurants here in London.",}, 

        # System turn should not have mentions or pems
        {"speaker": "SYSTEM", 
        "utterance": "Some people are allergic to histamine in tomatoes.",},

        {"speaker": "USER", 
        "utterance": "Talking of food, can you recommend me a restaurant in my city for our anniversary?",},
    ]

    result = cel.annotate(example)
    print_results(result)