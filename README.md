# SKGC: Scientific Knowledge Graph Construction

This repository contains code, data, prompts and results related to the task of automated topic extraction from scientific publications using Large Language Models (LLMs).

## Overview

SKGC.py is a university project aimed to investigate the potential of utilizing Large Language Models (represented by the  OpenAI GPT-4o model) for the task of topic extraction from scientific publications. The project employs a methodology that leverages the title, author keywords, and abstract of a publication to extract relevant topics. The extracted topics are then subjected to evaluation against a human expert annotated gold standard and compared with the results from the [CSO Classifier](https://github.com/angelosalatino/cso-classifier), which represents the current state-of-the-art in this domain.

## Contents

- **SKGC.py**: The main script for topic extraction and evaluation.
- **.env**: Contains the OpenAI API keys and the Organization ID.
- **GoldStandard.json**: Example input file containing publication data.
- **prompts_gpt_agent.yaml**: Prompts for the GPT agent for topic extraction.
- **prompts_gpt_assistant.yaml**: Prompts for the GPT assistant for checking the output format of the GPT agent for topic extraction.
- **prompts_gpt_agent_eval.yaml**: Prompts for the GPT agent for evaluation.
- **prompts_gpt_assistant_eval.yaml**: Prompts for the GPT assistant for checking the output format of the GPT agent for evaluation.
- **results.txt**: Contains the extraction results and evaluation details of the test run (running the program with testing mode C - see script - and the input file GoldStandard.json) in a user-friendly format for human readers.
- **results.json**: Contains the extraction and evaluation results  of the test run (running the program with testing mode C - see script - and the input file GoldStandard.json) in a machine-readable format.

## Usage

### Step 1: Install Dependencies
Ensure you have all the required dependencies installed. You can do this by running:
```bash
pip install -r requirements.txt
```

### Step 2: Fill in the .env File
Add your OpenAI API keys and Organization ID in the following format to the `.env` file:
```
API_KEY_AGENT="[your_openai_api_key_for_agent]"
API_KEY_ASSISTANT="[your_openai_api_key_for_assistant]"
ORGANIZATION="[your_organization_id]"
```

### Step 3: Adjust the Waiting Time Between API Calls (Optional)
The script includes a sleep time between API calls to avoid surpassing the tokens per minute (TPM) limit of the OpenAI API. The default sleep time is set to 10 seconds. It may be adjusted based on the specific usage tier of the organization in question:
- For usage tier 1 (30,000 TPM), 10 seconds is recommended.
- For usage tier 2 (450,000 TPM), you can reduce the sleep time to 1 second.

### Step 4: Ensure Required Files are in Place
Make sure the following files are in the same directory as `SKGC.py`:
- `prompts_gpt_agent.yaml`
- `prompts_gpt_assistant.yaml`
- `prompts_gpt_agent_eval.yaml`
- `prompts_gpt_assistant_eval.yaml`
- `GoldStandard.json` (or another JSON file with the required structure)

The required structure for the JSON file is as follows:
```json
{
    "[OBJECT_ID]": {
        "title": "[TITLE]",
        "abstract": "[ABSTRACT]",
        "keywords": [
            "[KEYWORD1]",
            "[KEYWORD PHRASE 2]",
            "[KEYWORD3]"
        ],
        "cso_output": {
            "final": [
                "[KEYWORD1]",
                "[KEYWORD PHRASE 2]",
                "[KEYWORD3]"
            ]
        },
        "gold_standard": {
            "majority_vote": [
                "[KEYWORD1]",
                "[KEYWORD PHRASE 2]",
                "[KEYWORD3]"
            ]
        }
    }
}
```

### Step 5: Run the Script
Execute the script by running:
```bash
python SKGC.py
```

## Acknowledgement
The input file GoldStandard.json containing the publication data of 70 scientific publications including the results of the CSO Classifier and a human expert annotated gold standard was obtained from the [CSO classifier repository](https://github.com/angelosalatino/cso-classifier) without any modifications. In their publication ([Salatino et al. 2021](https://doi.org/10.1007/s00799-021-00305-y)), the creators of the CSO Classifier indicate that "further evaluations by other members of the research community" are an intended use case of the gold standard.

## Transparency
ChatGPT was employed to assist with script documentation and refactoring. Additionally, it was utilized for specific code components that were not deemed to have cross-cutting relevance for the program logic, such as the DualOutput class.

## License
This project is licensed under the MIT license - see the [LICENSE](LICENSE) file for more information.
