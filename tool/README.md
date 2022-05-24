[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TXoecXn9-JeS-hd4a0vtUQPN7xJGc2C0?usp=sharing)

This README describes how to use our method on a local machine.
We also have [Google Colab notebook](https://colab.research.google.com/drive/1TXoecXn9-JeS-hd4a0vtUQPN7xJGc2C0?usp=sharing), you can try our method just running the notebook. See more description here XXX.

# Start with your local environment

## Step 1: Download models
First, download the models below:

- **MD for concepts and NEs**: Download the model from [here](http://gem.cs.ru.nl/rel_conv_project_folder.tar.gz) and put the `bert_conv-td` dir in `./tool`
- **ED for concepts and NEs**: Download the model from [here](https://drive.google.com/file/d/1OoC2XZp4uBy0eB_EIuIhEHdcLEry2LtU/view?usp=sharing) and put the folder ``rel_conv_project_folder`` in the dir `./tool`
- **Personal Entity Linking**: Download the model from [here](https://drive.google.com/file/d/1-jW8xkxh5GV-OuUBfMeT2Tk7tEzvH181/view?usp=sharing) and put the folder `s2e_ast_onto` in the dir `./tool/s2e_pe/model/`


## Step 2: Install packages
Second, install the all necessary packages with:
```sh
pip install -r requirements.txt
```

## Step 3: Run the code

Open the notebook `conversational_entity_linking.ipynb` and run the code.

The code is like:

```py
from conv_el import ConvEL
cel = ConvEL()

conv_example_1 = [
    {"speaker": "USER", 
    "utterance": "I am allergic to tomatoes but we have a lot of famous Italian restaurants here in London.",}, 

    # System turn should not have mentions or pems
    {"speaker": "SYSTEM", 
    "utterance": "Some people are allergic to histamine in tomatoes.",},

    {"speaker": "USER", 
    "utterance": "Talking of food, can you recommend me a restaurant in my city for our anniversary?",},
]

result_1 = cel.annotate(conv_example_1)
print_results(result_1) # This function is defined in start.ipynb

# Output:
# 
# USER: I am allergic to tomatoes but we have a lot of famous Italian restaurants here in London.
# 	 [17, 8, 'tomatoes', 'Tomato']
# 	 [54, 19, 'Italian restaurants', 'Italian_cuisine']
# 	 [82, 6, 'London', 'London']
# SYST: Some people are allergic to histamine in tomatoes.
# USER: Talking of food, can you recommend me a restaurant in my city for our anniversary?
# 	 [11, 4, 'food', 'Food']
# 	 [40, 10, 'restaurant', 'Restaurant']
# 	 [54, 7, 'my city', 'London']
```

