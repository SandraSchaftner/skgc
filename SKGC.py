""" SKGC: Scientific Knowledge Graph Construction
This script performs the extraction and evaluation of topics from scientific publications using the OpenAI GPT-4o model.
It allows users to input publication data either via a JSON file or direct text input, and then processes the data to
extract topics and evaluate the results against a gold standard and the CSO Classifier (CSOC) results.

The script follows these main steps:
1. User Input: The user can choose to input publication data either from a JSON file or via direct text input.
2. Topic Extraction: Topics are extracted from the publication data using a series of prompts to the GPT-4o model.
3. Evaluation: The extracted topics are evaluated against a gold standard and compared with the CSOC results.
4. Output: The results, including the extracted topics and evaluation metrics, are printed to the console and can be
   saved to a file.

Classes:
    DualOutput: A class to handle simultaneous output to both the console and a file.

Functions:
    user_selection_file_or_text() -> str: Asks the user to choose between reading publication data from a file or
    direct text input.

    user_text_publication() -> Dict[str, any]: Prompts the user to enter the details of a scientific publication.

    read_publication_from_file() -> List[Dict[str, any]]: Reads publication details from a JSON file.

    get_publications_data() -> List[Dict[str, any]]: Gets publication data based on user choice.

    select_publications(publications: List[Dict[str, Any]]) -> List[Dict[str, Any]]: Allows the user to select which
    publications to use.

    list_to_comma_separated_string(in_list: List) -> str: Converts a list to a comma-separated string.

    extract_topics_one(publication: Dict[str, Any], messages_history: List[Dict[str, str]]) -> List[str]: Extracts
    topics from a single publication.

    response_to_integer(response: str) -> int: Converts a string to an integer.

    eval_one(publication: Dict[str, Any], messages_history: List[Dict[str, str]]): Evaluates the topics extraction
    result for a single publication.

    extract_topics_all(publications: List[Dict[str, Any]], messages_history_all: List[List[Dict[str, str]]]) -> List[
    Dict[str, Any]]: Extracts topics from all publications.

    eval_all(publications_and_topics: List[Dict[str, Any]], messages_history_all: List[List[Dict[str, str]]]) -> List[
    Dict[str, Any]]: Evaluates all extracted topics.

    print_eval_details(publications_and_topics: List[Dict[str, Any]]): Prints evaluation details to the console and
    optionally saves them to a file.

    skgc_topics_and_eval_to_json(publications_and_topics: List[Dict[str, Any]]): Writes the evaluation results to a
    JSON file.

    query_gpt_agent(prompt: str, messages_history: List[Dict[str, str]]) -> str: Queries the GPT Agent API with the
    provided prompt.

    query_gpt_assistant(prompt: str) -> str: Queries the GPT Assistant API with the provided prompt.

    load_prompts_from_yaml(file_name: str) -> List[str]: Loads prompts from a YAML file.

    print_messages_history(messages_history_all: List[List[Dict[str, str]]]): Prints all conversations about all
    selected publications.

Usage:
    Before running the script, please install the dependencies with the command pip install -r requirements.txt

    Please also make sure to have the following files set up:
    - .env: containing the OpenAI API keys and the Organization ID in the following format:
        API_KEY_AGENT="[key]"
        API_KEY_ASSISTANT="[key]"
        ORGANIZATION="[id]"
    - prompts_gpt_agent.yaml: containing the prompts to the gpt agent for topic extraction in yaml format
    - prompts_gpt_assistant.yaml: containing the prompts to the gpt assistant for topic extraction in yaml format
    - prompts_gpt_agent_eval.yaml: containing the prompts to the gpt agent for evaluation in yaml format
    - prompts_gpt_assistant_eval.yaml: containing the prompts to the gpt agent for evaluation in yaml format

    Adjust the waiting time between the calls to the OpenAI API according to your organization's usage tier and the
    resulting TPM (tokens per minute) limit of the OpenAI API. For the default setting, the waiting time was set to 10
    seconds based on the TPM limit for usage tier 1 of the GPT-4o model as of May 31st 2024,
    which was 30,000. The message_history at the end of the extraction and evaluation pipeline in the present use case
    is 4,500-5,500 tokens, former message_history objects are smaller, which is why 10s waiting time was chosen as a
    safe limit. As the API usage tiers are assigned by OpenAI on organization level according to the API use, the
    number of seconds can be adjusted according to the usage tier of the user's organization. Usage tier 2 for example
    allows for 450,000 TPM which would enable to reduce the waiting time to 1s to be on the safe side.

    Run the script and follow the prompts to input publication data, extract topics, and evaluate the results. The
    results can be printed to the console and saved to a file.

Example:
    $ python SKGC.py
"""


import json
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from openai import OpenAI
import yaml
from scipy.stats import hmean
import sys
import time


class DualOutput:
    """
    A class to handle simultaneous output to both the console and a file.

    Attributes:
        file (TextIO): A file object to write the output to.
        console (object): The original standard output (sys.stdout).

    Methods:
        write(message: str):
            Writes a message to both the console and the file.

        flush():
            Flushes both the console and the file buffers.
    """

    def __init__(self, file):
        """
        Initializes the DualOutput class with a file object.

        Args:
            file (TextIO): A file object where the output will be written.
        """
        self.file = file
        self.console = sys.stdout

    def write(self, message):
        """
        Writes a message to both the console and the file.

        Args:
            message (str): The message to write.
        """
        self.console.write(message)  # Write to console
        self.file.write(message)  # Write to file

    def flush(self):
        """
        Flushes both the console and the file buffers.
        """
        self.console.flush()
        self.file.flush()


def user_selection_file_or_text() -> str:
    """
    Asks the user to enter 'f' (standing for file) or 't' (standing for text).
    Keeps asking until valid input is received.

    Returns:
        str: The user's input ('f' or 't').
    """
    print("Would you like to read the publication data from a JSON file ('f') or input it directly via text inputs"
          " ('t')?")
    while True:
        user_input = input("Please enter 'f' or 't': ").lower()
        if user_input in ['f', 't']:
            return user_input
        else:
            print("It seems like you didn't enter 'f' or 't'. Please enter either 'f' or 't'.")


def user_text_publication() -> Dict[str, any]:
    """
    Asks the user to enter the details of a scientific publication.

    Returns:
        Dict[str, any]: A dictionary containing the publication details.
    """
    title = input("Please enter the title of the scientific publication: ")
    keywords_string = input("Please enter the author keywords in a comma-separated list (e.g. keyword1, keyword"
                            " phrase 2, keyword3): ")
    keywords = keywords_string.split(',')
    abstract = input("Please enter the abstract of the publication: ")
    print("Next, I'm asking you for two keyword lists that will be used for the evaluation of the SKGC result. If you"
          " don't have these keyword lists available or you don't want to evaluate the extracted topics, press Enter.")
    csoc_result_string = input("Please enter the keyword list resulting from the CSO Classifier if available"
                               " (otherwise, press Enter): ")
    csoc_result = csoc_result_string.split(',')
    gold_standard_string = input("Please enter the human expert annotated keyword list that serves as the gold standard"
                                 " if available (otherwise, press Enter): ")
    gold_standard = gold_standard_string.split(',')
    publication_details = {
        "title": title.strip(),
        "keywords": [keyword.strip() for keyword in keywords],
        "abstract": abstract.strip(),
        "csoc_result": [keyword.strip() for keyword in csoc_result],
        "gold_standard": [keyword.strip() for keyword in gold_standard]
    }
    return publication_details


def read_publication_from_file() -> List[Dict[str, any]]:
    """
    Read publication details from a JSON file. Keeps asking until a valid file is given by the user.

    Returns:
        List[Dict[str, any]]: A list of dictionaries containing publication details.
    """
    print("For reading the publication data from a JSON file the following structure is required:")
    print('{')
    print('        [OBJECT_ID]: {')
    print('            "title": "[TITLE]",')
    print('            "abstract": "[ABSTRACT]",')
    print('            "keywords": [')
    print('                [KEYWORD1],')
    print('                [KEYWORD PHRASE 2],')
    print('                [KEYWORD3]')
    print('                ],')
    print('            "cso_output": {')
    print('                "final": [')
    print('                    [KEYWORD1],')
    print('                    [KEYWORD PHRASE 2],')
    print('                    [KEYWORD3]')
    print('                    ]')
    print('            }')
    print('            "gold_standard": {')
    print('                "majority_vote": [')
    print('                   [KEYWORD1],')
    print('                    [KEYWORD PHRASE 2],')
    print('                    [KEYWORD3]')
    print('                ]')
    print('            }')
    print('        }')
    print('        }')
    print()
    print("Info: There might be additional keys and values. These will be ignored by the program.")
    print("For proper working of the topic extraction, at least one element out of title, keywords and abstract"
          " must be provided. For proper working of the evaluation, the elements final cso_output and majority vote"
          " gold standard are essential.")

    while True:
        file_name = input("Please enter the JSON file name (ending with .json): ")
        try:
            with open(file_name, 'r') as file:
                data = json.load(file)
                publications = []
                for key, value in data.items():
                    if "title" in value:
                        title = value["title"]
                    if "keywords" in value:
                        keywords = value["keywords"]
                    if "abstract" in value:
                        abstract = value["abstract"]
                    if "cso_output" in value and "final" in value["cso_output"]:
                        csoc_result = value["cso_output"]["final"]
                    if "gold_standard" in value and "majority_vote" in value["gold_standard"]:
                        gold_standard = value["gold_standard"]["majority_vote"]

                    publication = {
                        "title": title.strip(),
                        "keywords": keywords,
                        "abstract": abstract.strip(),
                        "csoc_result": csoc_result,
                        "gold_standard": gold_standard
                    }
                    publications.append(publication)
                return publications
        except FileNotFoundError:
            print(f"File {file_name} not found. Make sure that the JSON file is located in the same folder like the"
                  f" script file. Please enter the file name again.")
        except json.JSONDecodeError:
            print(
                f"Error reading JSON from file {file_name}. Please ensure the file contains valid JSON and try again.")


def get_publications_data() -> List[Dict[str, any]]:
    """
    Get publications data based on user choice.

    Returns:
        List[Dict[str, any]]: A list of dictionaries containing publication details.
    """
    user_choice = user_selection_file_or_text()
    if user_choice == 't':
        publication_details = user_text_publication()
        print(f"Publication details: {publication_details}")
        return [publication_details]
    elif user_choice == 'f':
        publications = read_publication_from_file()
        return publications


def select_publications(publications: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Allow the user to select which publications to use based on their choice.

    Args:
        publications (List[Dict[str, Any]]): The list of publications that were read in before.

    Returns:
        publications (List[Dict[str, Any]]): The list of selected publications based on user input.
    """
    num_publications = len(publications)
    while True:
        print(
            f"{num_publications} publications were read in from the file."
            f" Please input A, B, C or D to choose the further process.")
        print("A: Use all publications that were read in.")
        print("B: Training mode: Use every second publication that was read in, starting with 3, then 5, 7, ...")
        # not starting with 1 because 1 and 2 are the examples in the prompt
        print("C: Testing mode: Use every second publication that was read in, starting with 4, then 6, 8, ...")
        # not starting with 2 because 1 and 2 are the examples in the prompt
        print("D: Use only one publication which will be selected in the next step.")

        choice = input("Please enter your choice (A, B, C, or D): ").upper()

        if choice in ['A', 'B', 'C', 'D']:
            if choice == 'A':
                return publications
            elif choice == 'B':
                return publications[2:36]  # not starting with 0 because 0 and 1 are the examples in the prompt
            elif choice == 'C':
                return publications[36:70]  # not starting with 1 because 0 and 1 are the examples in the prompt
            elif choice == 'D':
                while True:
                    one_publication_number = input(
                        f"Please input the number of the publication that should be selected."
                        f" Input a number between 1 and {num_publications}: ")
                    if one_publication_number.isdigit():
                        one_publication_number = int(one_publication_number)
                        if 1 <= one_publication_number <= num_publications:
                            return [publications[one_publication_number - 1]]
                    print(f"Invalid input. Please enter a number between 1 and {num_publications}.")
        else:
            print("Invalid input. Please enter A, B, C, or D.")


def list_to_comma_separated_string(in_list: List) -> str:
    """
    Helper function to convert a list to a comma-separated string.

    Args:
        in_list (List): Any list.

    Returns:
        string (str): Comma-separated string
    """
    string = ', '.join(map(str, in_list))
    return string


def extract_topics_one(publication: Dict[str, Any], messages_history: List[Dict[str, str]]) -> List[str]:
    """
        Extracts the topics of a given publication based on the data about the publication (title, keywords and
        abstract). This is done via three calls to the GPT agent that performs the extraction task in three steps:
        1. syntactic component: literal keywords
        2. semantic component: additional relevant keywords
        3. review and confirmation
        After each call to the GPT agent, there is a call to the GPT assistant that is only used to check the
        correctness of the output format of the GPT agent's answers.

        Args:
            publication (Dict[str, Any]): Publication of which the topics should be extracted.
            messages_history (List[Dict[str, str]]): GPT agent messages history regarding the topic extraction of
            the current publication. The messages history contains only the messages with the GPT agent, not with
            the GPT assistant that is only used to check the correctness of the output format of the GPT agent's
            answers. The Open AI API currently doesn't store the history itself
            ( https://platform.openai.com/docs/guides/text-generation/chat-completions-api).

        Returns:
            topics (List[str]): List of extracted topics for the given publication
        """
    prompts_gpt_agent = load_prompts_from_yaml("prompts_gpt_agent.yaml")
    prompts_gpt_assistant = load_prompts_from_yaml("prompts_gpt_assistant.yaml")
    # messages format according to the OpenAI API reference: https://platform.openai.com/docs/api-reference/
    messages_history.append({"role": "system", "content": "Hello GPT, you are my very helpful and intelligent assistant"
                                                          " for a difficult task today."})

    # 1st call to GPT agent API: Using prompt 1
    prompt1 = prompts_gpt_agent[0].replace("XXXtitleXXX", publication["title"])  # use of templates and placeholders
    prompt1 = prompt1.replace("XXXkeywordsXXX", list_to_comma_separated_string(publication["keywords"]))
    prompt1 = prompt1.replace("XXXabstractXXX", publication["abstract"])
    print("Sending prompt 1 to gpt agent...")  # In the terminal the user can see the progress of the extraction
    # and evaluation process
    response1 = query_gpt_agent(prompt1, messages_history)
    print("Received response 1 by gpt agent.")

    # 1st call to GPT assistant API: Using prompt 1a
    prompt1a = prompts_gpt_assistant[0].replace("XXXagent1XXX", response1)  # "XXXagent1XXX" is the placeholder
    # for the GPT agent response that will be checked by the assistant
    print("Sending prompt 1a to gpt assistant...")
    response1 = query_gpt_assistant(prompt1a)
    print("Received response 1a by gpt assistant.")
    messages_history.append({"role": "assistant", "content": response1})  # the messages history records all the
    # of the conversation about one publication. Instead of storing the direct responses of the GPT agent, the checked
    # responses of the GPT assistant are stored.

    # 2nd call to GPT agent API: Using prompt 2
    prompt2 = prompts_gpt_agent[1].replace("XXXresponse1XXX", response1)
    print("Sending prompt 2 to gpt agent...")
    response2 = query_gpt_agent(prompt2, messages_history)
    print("Received response 2 by gpt agent.")

    # 2nd call to GPT assistant API: Using prompt 2a
    prompt2a = prompts_gpt_assistant[1].replace("XXXagent2XXX", response2)
    print("Sending prompt 2a to gpt assistant...")
    response2 = query_gpt_assistant(prompt2a)
    print("Received response 2a by gpt assistant.")
    messages_history.append({"role": "assistant", "content": response2})

    # 3rd call to GPT agent API: Using prompt 3
    prompt3 = prompts_gpt_agent[2].replace("XXXresponse2XXX", response2)
    print("Sending prompt 3 to gpt agent...")
    response3 = query_gpt_agent(prompt3, messages_history)
    print("Received response 3 by gpt agent.")

    # 3rd call to GPT assistant API: Using prompt 3a
    prompt3a = prompts_gpt_assistant[2].replace("XXXagent3XXX", response3)
    print("Sending prompt 3a to gpt assistant...")
    response3 = query_gpt_assistant(prompt3a)
    print("Received response 3a by gpt assistant.")
    messages_history.append({"role": "assistant", "content": response3})
    skgc_topics = response3.split(',')
    topics = [topic.strip() for topic in skgc_topics]
    return topics


def response_to_integer(response: str) -> int:
    """
    Converts a string to an integer.
    If the string doesn't contain integer (or additional content), 0 is returned.

    Args:
    response (str): The response to be converted to an integer.

    Returns:
    int: The converted integer if successful, or 0 if an error occurred.
    """
    try:
        count = int(response)
        return count
    except ValueError:
        print("Error: The Evaluation procedure was partly not successful."
              "The evaluation count did not work as expected.")
        return 0


def eval_one(publication: Dict[str, Any], messages_history: List[Dict[str, str]]):
    """
            Evaluates the topic extraction result for the given publication and compares it to the result of the State
            Of the Art (SOTA) for topic extraction in the field of computer science, the Computer Science Ontology
            Classifier (CSOC): https://github.com/angelosalatino/cso-classifier
            The evaluation is done via four calls to the GPT agent API. The first two calls refer to the evaluation of
            the SKGC (this program) result, the last two calls refer to the evaluation of the CSOC result that was read
            in from the JSON file or the user input. For each SKGC and CSOC the metrics precision, recall and F1 for
            their evaluation with the provided gold standard (human expert annotation) are calculated.
            The four calls to the GPT agent refer to the following tasks:

            1. Comparison between the SKGC result and the gold standard via ordering the keyword lists according to the
            similarity of the keywords. Matching keywords will be put at the beginning of each list, followed by
            similar keywords. Keywords that cannot be matched are put at the end of each list. This intermediate step
            (instead of directly calculating the evaluation metrics) was found to improve the accuracy of the evaluation
            metrics in experiments conducted during the elaboration of the present pipeline. Furthermore, the output of
            the ordered lists helps in manually checking the evaluation results of the GPT agent.
            2. Count of the matching and similar keywords of the two lists SKGC result and gold standard.
            3. Comparison between the CSOC result and the gold standard via ordering the keyword lists according to the
            similarity of the keywords. Matching keywords will be put at the beginning of each list, followed by
            similar keywords. Keywords that cannot be matched are put at the end of each list.
            4. Count of the matching and similar keywords of the two lists CSOC result and gold standard.

            After each call to the GPT agent, there is a call to the GPT assistant that is only used to check the
            correctness of the output format of the GPT agent's answers.
            The final evaluation results (csoc_topics_ordered, gold_standard_ordered1, gold_standard_ordered2,
            skgc_topics_ordered, skgc_precision, skgc_recall, skgc_F1, csoc_precision, csoc_recall, csoc_F1) are stored
            in the publication dictionary.

            Args:
                publication (Dict[str, Any]): Publication of which the SKGC and CSOC results should be evaluated.
                messages_history (List[Dict[str, str]]): GPT agent messages history regarding the topic extraction of
            the current publication. The messages history contains only the messages with the GPT agent, not with
            the GPT assistant that is only used to check the correctness of the output format of the GPT agent's
            answers. The Open AI API currently doesn't store the history itself
            ( https://platform.openai.com/docs/guides/text-generation/chat-completions-api).
            Returns:
                None
            """

    prompts_gpt_agent_eval = load_prompts_from_yaml("prompts_gpt_agent_eval.yaml")
    prompts_gpt_assistant_eval = load_prompts_from_yaml("prompts_gpt_assistant_eval.yaml")

    # 1st call to GPT agent API: Using prompt 1
    prompt1 = prompts_gpt_agent_eval[0].replace("XXXskgc_topicsXXX", list_to_comma_separated_string
                                                (publication["skgc_topics"]))   # use of templates and placeholders
    prompt1 = prompt1.replace("XXXgold_standardXXX", list_to_comma_separated_string(publication["gold_standard"]))
    print("Sending prompt 1 to gpt agent...")  # In the terminal the user can see the progress of the extraction
    # and evaluation process
    response1 = query_gpt_agent(prompt1, messages_history)
    print("Received response 1 by gpt agent.")

    # 1st call to GPT assistant API: Using prompt 1a
    prompt1a = prompts_gpt_assistant_eval[0].replace("XXXagent4XXX", response1)  # "XXXagent4XXX" is the placeholder
    # for the GPT agent response that will be checked by the assistant
    print("Sending prompt 1a to gpt assistant...")
    response1 = query_gpt_assistant(prompt1a)
    print("Received response 1a by gpt assistant.")
    your_result = response1.split('Your result:')[1].split('Human expert result:')[0].strip()
    skgc_topics_ordered = your_result.split(',')
    human_expert_result = response1.split('Human expert result:')[1].strip()
    gold_standard_ordered1 = human_expert_result.split(',')  # gold standard is ordered twice: First in comparison with
    # the SKGC result (gold_standard_ordered1) and second in comparison with the CSOC result (gold_standard_ordered2)
    publication["skgc_topics_ordered"] = [topic.strip() for topic in skgc_topics_ordered]
    publication["gold_standard_ordered1"] = [topic.strip() for topic in gold_standard_ordered1]
    messages_history.append({"role": "assistant", "content": response1})  # the messages history records all the
    # of the conversation about one publication. Instead of storing the direct responses of the GPT agent, the checked
    # responses of the GPT assistant are stored.

    # 2nd call to GPT agent API: Using prompt 2
    prompt2 = prompts_gpt_agent_eval[1].replace("XXXskgc_topics_orderedXXX", list_to_comma_separated_string
                                                (skgc_topics_ordered))
    prompt2 = prompt2.replace("XXXgold_standard_orderedXXX", list_to_comma_separated_string
                              (gold_standard_ordered1))
    print("Sending prompt 2 to gpt agent...")
    response2 = query_gpt_agent(prompt2, messages_history)
    print("Received response 2 by gpt agent.")

    # 2nd call to GPT assistant API: Using prompt 2a
    prompt2a = prompts_gpt_assistant_eval[1].replace("XXXagent5XXX", response2)
    print("Sending prompt 2a to gpt assistant...")
    response2 = query_gpt_assistant(prompt2a)
    print("Received response 2a by gpt assistant.")
    matching_topics = response_to_integer(response2)
    if len(skgc_topics_ordered) == 0:
        precision = -1
        print("Error: Evaluation partly failed. Precision value of -1 means incorrect evaluation")
    else:
        precision = float(matching_topics)/len(skgc_topics_ordered)
    if len(gold_standard_ordered1) == 0:
        recall = -1
        print("Error: Evaluation partly failed. Recall value of -1 means incorrect evaluation")
    else:
        recall = float(matching_topics)/len(gold_standard_ordered1)
    if precision + recall == 0:
        f1 = -1
        print("Error: Evaluation partly failed. F1 value of -1 means incorrect evaluation")
    else:
        f1 = hmean([precision, recall])
    publication["skgc_precision"] = precision
    publication["skgc_recall"] = recall
    publication["skgc_f1"] = f1
    messages_history.append({"role": "assistant", "content": response2})

    # 3rd call to GPT agent API: Using prompt 3
    prompt3 = prompts_gpt_agent_eval[2].replace("XXXcsoc_topicsXXX", list_to_comma_separated_string
                                                (publication["csoc_result"]))
    prompt3 = prompt3.replace("XXXgold_standardXXX", list_to_comma_separated_string(publication["gold_standard"]))
    print("Sending prompt 3 to gpt agent...")
    response3 = query_gpt_agent(prompt3, messages_history)
    print("Received response 3 by gpt agent.")

    # 3rd call to GPT assistant API: Using prompt 3a
    prompt3a = prompts_gpt_assistant_eval[2].replace("XXXagent6XXX", response3)
    print("Sending prompt 3a to gpt assistant...")
    response3 = query_gpt_assistant(prompt3a)
    print("Received response 3a by gpt assistant.")
    csoc_result = response3.split('CSOC result:')[1].split('Human expert result:')[0].strip()
    csoc_topics_ordered = csoc_result.split(',')
    human_expert_result = response3.split('Human expert result:')[1].strip()
    gold_standard_ordered2 = human_expert_result.split(',')  # gold standard is ordered twice: First in comparison with
    # the SKGC result (gold_standard_ordered1) and second in comparison with the CSOC result (gold_standard_ordered2)
    publication["csoc_topics_ordered"] = [topic.strip() for topic in csoc_topics_ordered]
    publication["gold_standard_ordered2"] = [topic.strip() for topic in gold_standard_ordered2]
    messages_history.append({"role": "assistant", "content": response3})

    # 4th call to GPT agent API: Using prompt 4
    prompt4 = prompts_gpt_agent_eval[3].replace("XXXcsoc_topics_orderedXXX", list_to_comma_separated_string
                                                (csoc_topics_ordered))
    prompt4 = prompt4.replace("XXXgold_standard_orderedXXX", list_to_comma_separated_string
                              (gold_standard_ordered2))
    print("Sending prompt 4 to gpt agent...")
    response4 = query_gpt_agent(prompt4, messages_history)
    print("Received response 4 by gpt agent.")

    # 4th call to GPT assistant API: Using prompt 4a
    prompt4a = prompts_gpt_assistant_eval[3].replace("XXXagent7XXX", response4)
    print("Sending prompt 4a to gpt assistant...")
    response4 = query_gpt_assistant(prompt4a)
    print("Received response 4a by gpt assistant.")
    matching_topics = response_to_integer(response4)
    if len(csoc_topics_ordered) == 0:
        precision = -1
        print("Error: Evaluation partly failed. Precision value of -1 means incorrect evaluation")
    else:
        precision = float(matching_topics) / len(csoc_topics_ordered)
    if len(gold_standard_ordered2) == 0:
        recall = -1
        print("Error: Evaluation partly failed. Recall value of -1 means incorrect evaluation")
    else:
        recall = float(matching_topics) / len(gold_standard_ordered2)
    if precision + recall == 0:
        f1 = -1
        print("Error: Evaluation partly failed. F1 value of -1 means incorrect evaluation")
    else:
        f1 = hmean([precision, recall])
    publication["csoc_precision"] = precision
    publication["csoc_recall"] = recall
    publication["csoc_f1"] = f1
    messages_history.append({"role": "assistant", "content": response4})


def extract_topics_all(publications: List[Dict[str, Any]], messages_history_all: List[List[Dict[str, str]]]) -> List[
                        Dict[str, Any]]:
    """
    Extract topics from publications and add them to the dictionary of each publication as "SKGC topics".

    Args:
        publications (List[Dict[str, Any]]): List of publications.
        messages_history_all (List[List[Dict[str, str]]]): List of all conversations about all the publications selected
    Returns:
        List[Dict[str, Any]]: List of publications with "SKGC topics" added.

    """
    count = 1
    print("-" * 160)
    print("-" * 160)
    print("Topic extraction and evaluation process")  # In the terminal the user can see the progress of the extraction
    # and evaluation process
    for publication in publications:
        print("-" * 160)
        print("-" * 160)
        print(f"Publication {str(count)} of {str(len(publications))}: SKGC Topic Extraction")
        print("-" * 160)
        messages_history = list()
        skgc_topics = extract_topics_one(publication, messages_history)
        publication["skgc_topics"] = skgc_topics
        messages_history_all.append(messages_history)
        count += 1
    return publications


def eval_all(publications_and_topics: List[Dict[str, Any]], messages_history_all: List[List[Dict[str, str]]]) -> List[
             Dict[str, Any]]:
    """
    Evaluate all produced results (i.e. the extracted SKGC topics).
    Evaluation against Gold Standard and comparison with CSOC result.

    Args:
        publications_and_topics (List[Dict[str, Any]]): List of publications with extracted SKGC topics.
        messages_history_all (List[List[Dict[str, str]]]): List of all conversations about all the publications selected

    Returns:
        List[Dict[str, Any]]: List of publications with SKGC topics and evaluation results (csoc_topics_ordered,
        gold_standard_ordered1, gold_standard_ordered2, skgc_topics_ordered, skgc_precision, skgc_recall, skgc_F1,
        csoc_precision, csoc_recall, csoc_F1) added.

    """
    count = 0
    for publication in publications_and_topics:
        print("-" * 160)
        print("-" * 160)
        print(f"Publication {str(count+1)} of {str(len(publications_and_topics))}: Evaluation")  # In the terminal the
        # user can see the progress of the extraction and evaluation process
        print("-" * 160)
        eval_one(publication, messages_history_all[count])
        count += 1
    return publications_and_topics


def print_eval_details(publications_and_topics: List[Dict[str, Any]]):
    """
        Prints evaluation details to console, and - if requested by the user - additionally stores the evaluation
        in a txt-file.

        Args:
            publications_and_topics (List[Dict[str, Any]]): List of publications with the extraction and evaluation
            results.

        Returns:
            None
        """
    print("Would you like to save the evaluation details to a txt-file?")
    print("A: Yes")
    print("B: No")
    while True:
        choice = input("Please enter A or B: ")
        if choice.upper() == "A":
            file_bool = True
            while True:
                file_name = input("Please enter a file name to store the evaluation details (ending with .txt): ")
                if file_name.endswith(".txt"):
                    break
                else:
                    print("Invalid file name. Please ensure it ends with .txt.")
            break
        elif choice.upper() == "B":
            file_bool = False
            file_name = "file_name.txt"
            break
        else:
            print("Invalid input. Please enter 'A' or 'B'")

    with open(file_name, 'w') as file:
        if file_bool:
            dual_output = DualOutput(file)
            sys.stdout = dual_output  # "print" function will output the given strings both to the console and to
            # the given file
        count = 1
        print("-" * 160)
        print("Evaluation details:")
        for publication in publications_and_topics:
            print("-" * 160)
            print(f"Publication {str(count)} of {str(len(publications_and_topics))}:")
            count += 1
            print()
            print(f"Publication title: {publication['title']}")
            print()
            skgc_topics = publication["skgc_topics_ordered"]
            gold_standard_order1 = publication["gold_standard_ordered1"]  # gold standard is ordered twice: First in
            # comparison with the SKGC result (gold_standard_ordered1) and second in comparison with the CSOC result
            # (gold_standard_ordered2)
            # Both ordered lists are used in the output table to check the accuracy of the order of the SKGC results
            # and the CSOC results respectively
            gold_standard_order2 = publication["gold_standard_ordered2"]
            csoc_result = publication["csoc_topics_ordered"]
            skgc_precision = publication["skgc_precision"]
            skgc_recall = publication["skgc_recall"]
            skgc_f1 = publication["skgc_f1"]
            csoc_precision = publication["csoc_precision"]
            csoc_recall = publication["csoc_recall"]
            csoc_f1 = publication["csoc_f1"]
            max_len = max(len(skgc_topics), len(gold_standard_order1), len(gold_standard_order2), len(csoc_result))
            print(
                f"{'SKGC Topics':<40} {'Gold Standard (Order 1)':<40} {'Gold Standard (Order 2)':<40}"
                f" {'CSOC Topics':<40}")  # table format with 4 columns of 40 characters length

            print("-" * 160)

            for i in range(max_len):
                skgc_topic = skgc_topics[i] if i < len(skgc_topics) else ''
                gold_standard_topic1 = gold_standard_order1[i] if i < len(gold_standard_order1) else ''
                gold_standard_topic2 = gold_standard_order2[i] if i < len(gold_standard_order2) else ''
                csoc_topic = csoc_result[i] if i < len(csoc_result) else ''
                print(f"{skgc_topic:<40} {gold_standard_topic1:<40} {gold_standard_topic2:<40} {csoc_topic:<40}")
                print("-" * 160)
            print(f"{f'Precision: {skgc_precision}':<40} {'':<40} {'':<40} {f'Precision: {csoc_precision}':<40}")
            print("-" * 160)
            print(f"{f'Recall: {skgc_recall}':<40} {'':<40} {'':<40} {f'Recall: {csoc_recall}':<40}")
            print("-" * 160)
            print(f"{f'F1: {skgc_f1}':<40} {'':<40} {'':<40} {f'F1: {csoc_f1}':<40}")
        # Calculation of the overall mean of the precision, recall and F1 mean of each SKGC and CSOC
        print()
        print("-" * 160)
        print("Overall comparison between SKGC and CSOC: Precision mean comparison")
        precision_list_skgc = list()
        precision_list_csoc = list()
        for publication in publications_and_topics:
            precision_list_skgc.append(publication["skgc_precision"])
            precision_list_csoc.append(publication["csoc_precision"])
        precision_mean_skgc = hmean(precision_list_skgc)  # average of means
        precision_mean_csoc = hmean(precision_list_csoc)  # average of means
        print(f"Precision mean of SKGC appproach: {str(precision_mean_skgc)}")
        print(f"Precision mean of CSOC appproach: {str(precision_mean_csoc)}")
        print("-" * 160)
        print("Overall comparison between SKGC and CSOC: Recall mean comparison")
        recall_list_skgc = list()
        recall_list_csoc = list()
        for publication in publications_and_topics:
            recall_list_skgc.append(publication["skgc_recall"])
            recall_list_csoc.append(publication["csoc_recall"])
        recall_mean_skgc = hmean(recall_list_skgc)  # average of means
        recall_mean_csoc = hmean(recall_list_csoc)  # average of means
        print(f"Recall mean of SKGC appproach: {str(recall_mean_skgc)}")
        print(f"Recall mean of CSOC appproach: {str(recall_mean_csoc)}")
        print("-" * 160)
        print("Overall comparison between SKGC and CSOC: F1 mean comparison")
        f1_list_skgc = list()
        f1_list_csoc = list()
        for publication in publications_and_topics:
            f1_list_skgc.append(publication["skgc_f1"])
            f1_list_csoc.append(publication["csoc_f1"])

        if all(x >= 0 for x in f1_list_skgc):
            f1_mean_skgc = hmean(f1_list_skgc)  # average of means
        else:
            f1_mean_skgc = -1
            print("Error: Overall evaluation partly failed. F1 value of -1 means incorrect evaluation")
        if all(x >= 0 for x in f1_list_csoc):
            f1_mean_csoc = hmean(f1_list_csoc)  # average of means
        else:
            f1_mean_csoc = -1
            print("Error: Overall evaluation partly failed. F1 value of -1 means incorrect evaluation")
        print(f"F1 mean of SKGC appproach: {str(f1_mean_skgc)}")
        print(f"F1 mean of CSOC appproach: {str(f1_mean_csoc)}")
        print("-" * 160)
        if f1_mean_skgc > f1_mean_csoc:  # Depending on the F1 overall mean comparison, a final conclusion statement is
            # printed
            print("According to the automatic evaluation, the SKGC approach yielded better results than the CSOC"
                  " approach.")
        elif f1_mean_skgc < f1_mean_csoc:
            print("According to the automatic evaluation, the CSOC approach yielded better results than the SKGC"
                  " approach.")
        else:
            print("According to the automatic evaluation, the SKGC and the CSOC approach performed overall"
                  " identically.")
        if file_bool:
            # Restore the original stdout
            sys.stdout = dual_output.console


def skgc_topics_and_eval_to_json(publications_and_topics: List[Dict[str, Any]]):
    """
    Write publications_and_topics (with the evaluation metrics) to a JSON file.

    Args:
        publications_and_topics (List[Dict[str, Any]]): List of publications with the extraction and evaluation results.

    Returns:
        None
    """
    while True:
        file_name = input("Please enter the file name to save the JSON data (ending with .json): ")
        try:
            with open(file_name, 'w') as file:
                json.dump(publications_and_topics, file, indent=4)
                print(f"Data saved to {file_name}")
                break
        except Exception as e:
            print(f"Error writing to file {file_name}: {e}")


def query_gpt_agent(prompt: str, messages_history: List[Dict[str, str]]) -> str:
    """
    Queries the GPT Agent API (OpenAI model gpt-4o) with the provided prompt.

    Args:
        prompt (str): The prompt to send to the API.
        messages_history (List[Dict[str, str]]): GPT agent messages history regarding the topic extraction and
        evaluation of the current publication. The messages history contains only the messages with the GPT agent, not
        with the GPT assistant that is only used to check the correctness of the output format of the GPT agent's
        answers. The Open AI API currently doesn't store the history itself
        ( https://platform.openai.com/docs/guides/text-generation/chat-completions-api).

    Returns:
        response (str): The response from the API.
    """
    time.sleep(10)  # waiting time was added to not surpass the TPM (tokens per minute) limit of the OpenAI API.
    # 10s waiting time was calculated based on the TPM limit for usage tier 1 of the GPT-4o model as of May 31st 2024,
    # which was 30,000. The message_history at the end of the extraction and evaluation pipeline in the present use case
    # is 4,500-5,500 tokens, former message_history objects are smaller, which is why 10s waiting time was chosen as a
    # safe limit. As the API usage tiers are assigned by OpenAI on organization level according to the API use, the
    # number of seconds can be adjusted according to the usage tier of the user's organization. Usage tier 2 for example
    # allows for 450,000 TPM which would enable to reduce the waiting time to 1s to be on the safe side.

    api_key = os.getenv("API_KEY_AGENT")  # the API_KEY_AGENT must be stored in the .env file.
    # For this implementation it was decided to use two OpenAI keys, one for the agent role that accomplishes the
    # extraction and evaluation tasks, and one for the assistant role that checks the output of the agent for formal
    # correctness. As per current status, the conversation history is not stored using the OpenAI API (differently than
    # using ChatGPT). However, in case of future changes and also for clear delineation between the agent and assistant
    # role, two different API keys and functions are used.
    if not api_key:
        raise ValueError("API key not found. Please set the API_KEY_AGENT environment variable.")

    organization = os.getenv("ORGANIZATION")  # the ORGANIZATION ID must be stored in the .env file.
    if not organization:
        raise ValueError("Organization for OpenAI access not found. Please set the ORGANIZATION environment variable.")

    client = OpenAI(
        organization=organization,
        api_key=api_key
    )

    messages_history.append({"role": "user", "content": prompt})  # GPT agent messages history regarding the topic
    # extraction and evaluation of the current publication. The messages history contains only the messages with the GPT
    # agent, not with the GPT assistant that is only used to check the correctness of the output format of the GPT
    # agent's answers. The Open AI API currently doesn't store the history itself
    # ( https://platform.openai.com/docs/guides/text-generation/chat-completions-api).
    # messages format according to the OpenAI API reference: https://platform.openai.com/docs/api-reference/
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # GPT-4o was chosen as the most recently published model at the time of creating of this
            # script (mid to end of May 2024). According to the announcement of OpenAI, GPT-4o "achieves
            # GPT-4 Turbo-level performance on text, reasoning, and coding intelligence", is faster and cheaper in use
            # than GPT-4 Turbo.

            messages=messages_history,
            temperature=0,  # temperature was set to 0 according to OpenAI guidelines for minimizing non-determinism
            # (https://platform.openai.com/docs/guides/text-generation/reproducible-outputs and
            # https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api)
            seed=4  # use of seed parameter to promote reproducible outputs
            # (https://platform.openai.com/docs/guides/text-generation/reproducible-outputs)
        )
        response = response.choices[0].message.content.strip()
        return response

    except Exception as e:
        print(f"Error querying OpenAI API: {e}")
        return ""


def query_gpt_assistant(prompt: str) -> str:
    """
    Queries the GPT Assistant API (OpenAI model gpt-4o) with the provided prompt.

    Args:
        prompt (str): The prompt to send to the API.

    Returns:
        response (str): The response from the API.
    """
    time.sleep(10)  # waiting time was added to not surpass the TPM (tokens per minute) limit of the OpenAI API.
    # 10s waiting time was calculated based on the TPM limit for usage tier 1 of the GPT-4o model as of May 31st 2024,
    # which was 30,000. The message_history at the end of the extraction and evaluation pipeline in the present use case
    # is 4,500-5,500 tokens, former message_history objects are smaller, which is why 10s waiting time was chosen as a
    # safe limit. As the API usage tiers are assigned by OpenAI on organization level according to the API use, the
    # number of seconds can be adjusted according to the usage tier of the user's organization. Usage tier 2 for example
    # allows for 450,000 TPM which would enable to reduce the waiting time to 1s to be on the safe side.

    api_key = os.getenv("API_KEY_ASSISTANT")  # the API_KEY_ASSISTANT must be stored in the .env file.
    # For this implementation it was decided to use two OpenAI keys, one for the agent role that accomplishes the
    # extraction and evaluation tasks, and one for the assistant role that checks the output of the agent for formal
    # correctness. As per current status, the conversation history is not stored using the OpenAI API (differently than
    # using ChatGPT). However, in case of future changes and also for clear delineation between the agent and assistant
    # role, two different API keys and functions are used.

    if not api_key:
        raise ValueError("API key not found. Please set the API_KEY_ASSISTANT environment variable.")

    organization = os.getenv("ORGANIZATION")  # the ORGANIZATION ID must be stored in the .env file.
    if not organization:
        raise ValueError("Organization for OpenAI access not found. Please set the ORGANIZATION environment variable.")

    client = OpenAI(
        organization=organization,
        api_key=api_key
    )

    messages = list()
    messages.append({"role": "system", "content": "Hello GPT, you are my very helpful and intelligent assistant for"
                                                  "checking the answers of another GPT sister of yourself."})
    messages.append({"role": "user", "content": prompt})
    # messages format according to the OpenAI API reference: https://platform.openai.com/docs/api-reference/

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # GPT-4o was chosen as the most recently published model at the time of creating of this
            # script (mid to end of May 2024). According to the announcement of OpenAI, GPT-4o "achieves
            # GPT-4 Turbo-level performance on text, reasoning, and coding intelligence", is faster and cheaper in use
            # than GPT-4 Turbo.
            messages=messages,
            temperature=0,  # temperature was set to 0 according to OpenAI guidelines for minimizing non-determinism
            # (https://platform.openai.com/docs/guides/text-generation/reproducible-outputs and
            # https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api)
            seed=4  # use of seed parameter to promote reproducible outputs
            # (https://platform.openai.com/docs/guides/text-generation/reproducible-outputs)
        )
        response = response.choices[0].message.content.strip()
        return response
    except Exception as e:
        print(f"Error querying OpenAI API: {e}")
        return ""


def load_prompts_from_yaml(file_name: str) -> List[str]:
    """
    Load prompts from a YAML file.

    Args:
        file_name (str): The name of the YAML file containing the prompts.

    Returns:
        prompts (List [str]): A list of prompts.
    """
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    prompts = list()
    try:
        with open(file_path, 'r') as file:
            prompts_yaml = yaml.safe_load(file)
            prompts = [prompt.strip() for prompt in prompts_yaml]
        return prompts

    except FileNotFoundError as fnf_error:
        print(f"Error: The file '{file_name}' was not found. {fnf_error}")
        return prompts

    except yaml.YAMLError as yaml_error:
        print(f"Error: Failed to parse YAML file '{file_name}'. {yaml_error}")
        return prompts

    except Exception as e:
        print(f"An unexpected error while handling the yaml file {file_name} occurred: {e}")
        return prompts


def print_messages_history(messages_history_all: List[List[Dict[str, str]]]):
    """
        Print all conversations about all selected publications

        Args:
            messages_history_all (List[List[Dict[str, str]]]): List of all conversations about all selected publications

        Returns:
            None
        """
    print("-" * 160)
    print("-" * 160)
    print("Conversation with GPT agent: Topic extraction and evaluation")
    count = 1
    for messages_history in messages_history_all:
        print("-" * 160)
        print(f"Publication {str(count)} of {str(len(messages_history_all))}")
        for message in messages_history:
            for key, value in message.items():
                print(f"{key.capitalize()}: {value}")
                print("-" * 160)
        count += 1


def main():
    load_dotenv()
    publications = get_publications_data()
    if not publications:
        print(
            f"An unexpected error while reading the data occurred."
            f" Please restart the program.")
        return

    if len(publications) == 1:
        print("The following data was successfully read in:")
        print("-" * 160)
        for key, value in publications[0].items():
            if isinstance(value, list):
                print(f"{key.capitalize()}: {list_to_comma_separated_string(value)}")
            else:
                print(f"{key.capitalize()}: {value}")
        print("The publication is being processed now.")
    else:
        publications = select_publications(publications)
        print("The following data was successfully read in:")
        count = 1
        for publication in publications:
            print("-" * 160)
            print(f"Publication {str(count)} of {str(len(publications))}:")
            count += 1
            for key, value in publication.items():
                if isinstance(value, list):
                    print(f"{key.capitalize()}: {list_to_comma_separated_string(value)}")
                else:
                    print(f"{key.capitalize()}: {value}")
            print()
        print("-" * 160)
        print("-" * 160)
        print("The selected publications are being processed now.")

    messages_history_all = list()

    # step 1: topic extraction
    publications_and_topics = extract_topics_all(publications, messages_history_all)

    # step 2: evaluation
    eval_all(publications_and_topics, messages_history_all)

    # step 3: output results
    # 3.1: print all conversations about all processed publications
    print_messages_history(messages_history_all)

    # 3.2: print evaluation details and store them in a txt-file (on user request)
    print_eval_details(publications_and_topics)

    # 3.3: store the final result (SKGC topics and the evaluation metrics for the SKGC and CSOC approach) in a json file
    skgc_topics_and_eval_to_json(publications_and_topics)


if __name__ == "__main__":
    main()
