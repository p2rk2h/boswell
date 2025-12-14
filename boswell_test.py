#!/usr/bin/env python3
"""
Boswell Test - LLM Comparative Analysis Tool

This script automates the process of:
1. Prompting multiple LLM models with an essay question
2. Having each model grade every other model's essay
3. Saving and analyzing the results

This implementation uses concurrent processing to improve performance.
"""

import os
import sys
import json
import time
import re
import argparse
import importlib
import requests
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from statistics import median
import concurrent.futures
from threading import Lock

from LetterGrade import LetterGrade	#####

# Configuration
API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Check for API key
if not API_KEY:
    print("Error: OPENROUTER_API_KEY environment variable is not set.")
    print("Please set it with: export OPENROUTER_API_KEY='your-api-key'")
    sys.exit(1)

# Model configurations - customize as needed
import requests
# import pandas as pd


def get_free_openrouter_models():
    url = "https://openrouter.ai/api/v1/models"
    headers = {
        "Authorization": "Bearer YOUR_OPENROUTER_API_KEY", # Optional for listing, but good practice
        "Content-Type": "application/json"
    }
    try :
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        all_models = response.json().get('data', [])	#	200 OK
        # Filter for models where prompt and completion costs are 0
        free_models = [
            model for model in all_models
            if float(model['pricing']['prompt']) == 0 and float(model['pricing']['completion']) == 0 ]

        modelsF = [ ]
        for model in free_models:
            model_id = model.get('id')
            name = model.get('name')
            #  Extract provider from model_id
            provider = model_id.split('/')[0].title() if model_id and '/' in model_id else "N/A"
            modelsF.append({'name': name, 'model_id': model_id, 'provider': provider})
#            print(f"Name: {model['name']} | ID: {model['id']}")

        models , provider_key = [ ] , [ ]
        for model in modelsF :
            provider = model[ 'name' ].split( ':' )[ 0 ].strip( )
            if ( provider not in provider_key ) :
                provider_key.append( provider )
                models.append( model )
#                print( "FREE:" , provider , model )
        return ( models )
    except requests.exceptions.RequestException as e :
        print( f"Error fetching free models from OpenRouter API: {e}" )
        return ( [ ] )

def getTopModels( free : bool , modelimit = 100 ) :
    # The OpenRouter API endpoint to list all models	#  https://gemini.google.com/app/f10c6613f816b435
    API_URL = "https://openrouter.ai/api/v1/models"

    if ( free ) :
        free = min( modelimit , MAX_FREE_MODELS )
        return ( get_free_openrouter_models( ) )
    try :
        SCRAPING_URL = "https://openrouter.ai/models?sort=usage&sort_order=desc&type=chat&limit=" + str( modelimit ) if modelimit > 0 else '50'
        print( f"Attempting to retrieve model list from API endpoint: {API_URL}" )
        response = requests.get( API_URL )
        response.raise_for_status( )  # Raise HTTPError for bad (4xx or 5xx)
        data = response.json( )
    except requests.exceptions.RequestException as e :
        print( f"Error fetching data from OpenRouter API : {e}" )
        #  Fallback to known-good, table-formatted scraping URL if API fails
        print( f"Falling back to scraping URL: {SCRAPING_URL}" )
        # ⚠️ NOTE: For scraping fallback, introduce robust Selenium code
        data = { "data" : [ ] }
    models100 = [ ]  #  process data
    for model_info in data.get( 'data' , [ ] ) :
        model_id = model_info.get( 'id' )
        name = model_info.get( 'name' )
        #  Extract provider from model_id (e.g., "openai/gpt-4" -> "openai")
        if ( model_id and '/' in model_id ) :
            provider = model_id.split( '/' )[ 0 ].title( )
#            print( "TOP:" , model_id , name , provider )
        else :
            provider = "N/A"
        models100.append( { 'name' : name , 'model_id' : model_id , 'provider' : provider } )   #  Add other useful info; e.g., 'context_length' : model_info.get('context_length')
    return ( models100 )

# Only models verified to work with OpenRouter are included
'''
MODELS = [
    {"name": "GPT-4o", "model_id": "openai/gpt-4o"},	#####	20251020
    {"name": "Claude-3-Opus-20240229", "model_id": "anthropic/claude-3-opus-20240229"},	#####	20251020
    {"name": "Claude-3.5-Sonnet", "model_id": "anthropic/claude-3.5-sonnet-20240620"},	#####	20251020
    {"name": "Claude-3-haiku", "model_id": "anthropic/claude-3-haiku-20240307"},	#####	20251020
    {"name": "Llama-3-8B", "model_id": "meta-llama/llama-3-8b-instruct"},	#####	20251020
    {"name": "GPT-3.5-Turbo", "model_id": "openai/gpt-3-5-turbo"},	#####	20251020
    {"name": "o1-mini", "model_id": "openai/o1-mini"},	#####	20251020
    {"name": "o1-preview", "model_id": "openai/o1-preview"},	#####	20251020
    {"name": "DeepSeek-V2", "model_id": "deepseek/deepseek-v2"},	#####	20251020
    {"name": "DeepSeek-Coder-V2", "model_id": "deepseek/deepseek-coder-v2"},	#####	20251020
    {"name": "grok2", "model_id": "x-ai/grok-2"},	#####	20251020
    {"name": "grok-beta", "model_id": "x-ai/grok-beta"},	#####	20251020
    {"name": "Qwen-vl-8b", "model_id": "qwen/qwen3-vl8b-instruct"},	#####	20251020
    {"name": "Gemini Flash 1.5", "model_id": "google/gemini-1-5-flash"},	#####	20251020
    {"name": "Mistral-large-2", "model_id": "mistralai/mistral-large-2"},	#####	20251020
    {"name": "Mistral-nemo", "model_id": "mistralai/mistral-nemo"},	#####	20251020
    {"name": "Cohere-command-r-plus", "model_id": "cohere/command-r-plus"},	#####	20251020
    {"name": "Microsoft-phi-3-medium", "model_id": "microsoft/phi-3-medium"}	#####	20251020
]

models100 = getTopModels( free = True )

#	use only well-known providers	#####
prvdr = [ 'OpenAI' , 'Anthropic' , 'Google' , 'Mistral' , 'Meta' , 'Qwen' , 'xAI' , 'DeepSeek' , 'Perplexity' , 'Cohere' , 'NVIDIA' , 'Baidu' ]	#####
MODELS , mdls , i , j = [ ] , { } , 1 , 0
for d in models100 :
    k = d[ 'name' ].find( ':' )
    if ( d[ 'name' ][ : k ] not in mdls.keys( ) and d[ 'name' ][ : k ] in prvdr ) :
        d0 = dict( )
        d0[ 'name' ] = d[ 'name' ]
        d0[ 'model_id' ] = d[ 'model_id' ]
        mdls[ d[ 'name' ][ : k ] ] = d0
        MODELS.append( d0 )
        i += 1
    j += 1
'''

# Use a set for O(1) provider lookup speed
ALLOWED_PROVIDERS = {'OpenAI', 'Anthropic', 'Google', 'Mistral', 'Meta', 
                     'Qwen', 'xAI', 'DeepSeek', 'Perplexity', 'Cohere', 
                     'NVIDIA', 'Baidu'}

# store the result, keyed by provider name,
# select only the first model encountered for each provider.
MAX_FREE_MODELS = 16
MODELS_BY_PROVIDER = {} 
MODELS = [ ]  #  final_models = []

'''
for model_data in models100:
    # Safely extract the root model name (e.g., 'Google' from 'Google: Gemini Pro')
    # Using .split(':') is safer than .find(':')
    provider_key = model_data['name'].split(':')[0].strip()

    # 1. Check if provider is allowed
    if provider_key in ALLOWED_PROVIDERS:
        
        # 2. Check if we have already stored a model for this provider (Ensures one model per provider)
        if provider_key not in MODELS_BY_PROVIDER:
            MODELS_BY_PROVIDER[provider_key] = model_data
            MODELS.append(model_data)  #  final_models

# The list of selected models is now in final_models
# Example: final_models
# print( MODELS )
# Additional models to try (commented out as they're not currently available)
# ADDITIONAL_MODELS = [
#     {"name": "Claude-3.5-Sonnet", "model_id": "anthropic/claude-3-5-sonnet-20240620"},
# ]
'''

# Available domains
AVAILABLE_DOMAINS = {
#    "pol_sci_1": "Political Science - Level 1: AI Policy Analysis",
    "sample_boswell_query": "Political Science - Level 0: AI Policy Analysis",	#####	allow composite grades 'A-/B+', 'B-/C+', 'C-/D+', 'F'
    "pol_sci_2": "Political Science - Level 2: AI Governance Analysis",
    "comp_sci_1": "Computer Science - Level 1: Algorithm Analysis",
    "comp_sci_2": "Computer Science - Level 2: System Design"
}


def calculate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string.
    
    This is a rough estimation using a simple rule of thumb:
    - Approximately 4 characters per token for English text
    """
    return max(1, len(text) // 4)


def call_openrouter_api(model_id: str, prompt: str) -> Dict[str, Any]:
    """Call OpenRouter API with the specified model and prompt.
    
    Returns:
        JSON response with added cost information
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/alanwilhelm/botwell",  # Update with your actual repo
        "X-Title": "Boswell Test"
    }
    
    data = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    start_time = time.time()
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=data
    )
    end_time = time.time()
    
    if response.status_code != 200:
        print(f"Error calling API: {response.status_code}")
        print(response.text)
        return {"error": response.text, "cost": 0.0, "duration": 0.0}
    
    result = response.json()
    
    # Calculate token counts if not provided
    if "usage" not in result:
        prompt_tokens = calculate_tokens(prompt)
        completion_tokens = 0
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0].get("message", {})
            completion_text = message.get("content", "")
            completion_tokens = calculate_tokens(completion_text)
        
        result["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    
    # Estimate cost (not all models provide accurate pricing info)
    # These are rough estimates, actual pricing varies by model
    input_cost_per_thousand = 0.0005  # $0.0005 per 1K input tokens (conservative)
    output_cost_per_thousand = 0.0015  # $0.0015 per 1K output tokens (conservative)
    
    # Try to get model-specific pricing from the response
    model_info = result.get("model", {})
    if isinstance(model_info, dict):
        pricing = model_info.get("pricing", {})
        if pricing:
            input_cost_per_thousand = pricing.get("input", input_cost_per_thousand)
            output_cost_per_thousand = pricing.get("output", output_cost_per_thousand)
    
    input_cost = (result["usage"]["prompt_tokens"] / 1000) * input_cost_per_thousand
    output_cost = (result["usage"]["completion_tokens"] / 1000) * output_cost_per_thousand
    total_cost = input_cost + output_cost
    
    # Add cost and duration information to the result
    result["cost_info"] = {
        "input_tokens": result["usage"]["prompt_tokens"],
        "output_tokens": result["usage"]["completion_tokens"],
        "input_cost": input_cost,
        "output_cost": output_cost, 
        "total_cost": total_cost,
        "duration": end_time - start_time
    }
    
    return result


def fetch_available_openrouter_models() -> List[Dict[str, Any]]:
    """Fetch the list of available models from OpenRouter."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    print("Fetching available models from OpenRouter...")
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers
        )
        
        if response.status_code != 200:
            print(f"Error fetching models: {response.status_code}")
            print(response.text)
            return []
        
        models_data = response.json()
        # Process and format the models using a thread pool for faster processing
        formatted_models = []
        model_data_list = models_data.get("data", [])
        
        # Define a function to process each model
        def process_model(model_data):
            model_id = model_data.get("id")
            if not model_id:
                return None
                
            model_name = model_data.get("name", model_id.split("/")[-1])
            return {
                "name": model_name,
                "model_id": model_id,
                "context_length": model_data.get("context_length"),
                "pricing": model_data.get("pricing", {}),
                "description": model_data.get("description", "")
            }
            
        # Process models concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            for model in executor.map(process_model, model_data_list):
                if model:
                    formatted_models.append(model)
        
        return formatted_models
        
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []


def update_models_file(output_file: str = "openrouter_models.json") -> None:
    """Update the local models file with the latest available models from OpenRouter."""
    models = fetch_available_openrouter_models()
    print( "fetch available models:" , models )
    if not models:
        print("No models fetched. Check your API key and connection.")
        return
    
    # Save the models to a JSON file
    try:
        with open(output_file, 'w') as f:
            json.dump({
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "models": models
            }, f, indent=2)
        
        print(f"Successfully updated models file: {output_file}")
        print(f"Found {len(models)} available models")
        
        # Print a sample of model IDs for convenience
        print("\nSample model IDs for use in boswell_test.py:")
        for i, model in enumerate(models[:10]):  # Show first 10 models
            print(f"  {model['name']}: \"{model['model_id']}\"")
        
        if len(models) > 10:
            print(f"  ... and {len(models) - 10} more (see {output_file} for complete list)")
            
    except Exception as e:
        print(f"Error saving models file: {e}")


def extract_grade(feedback: str) -> str:
    """Extract letter grade from grading feedback."""
    # First try to match the exact format we requested
    match = re.search(r"Grade:\s*([A-C][+-]?)", feedback)
    if match:
        return match.group(1)
    
    # Fall back to more flexible patterns if the exact format isn't found
    # Look for letter grades in various contexts
    grade_patterns = [
#        r"([A-C][+-]?)\s*grade",  # "A+ grade" or "B- grade"
        r"(([A-F][+-]?(/[A-F][+-]?)?))\s*grade",  # "A+/A grade" or "B- grade"
#        r"grade\s*(?:of|is|:)?\s*([A-C][+-]?)",  # "grade of A" or "grade: B+/B"
        r"grade\s*(?:of|is|:)?\s*([A-F][+-]?(/[A-F][+-]?)?)",  # "grade of A" or "grade: B+"
#        r"grade\s*[\"']([A-C][+-]?)[\"']",  # grade "A-" or grade 'B'
        r"grade\s*[\"']([A-F][+-]?(/[A-F][+-]?)?)[\"']",  # grade "A-" or grade 'B/C'
        r"([A-F][+-]?(/[A-F][+-]?)?)$"  # Just the grade at the end of a line
#         r"^(\S+)\s+gave\s+(\S+)\s+a grade of\s+([A-F][+-]?(/[A-F][+-]?)?)$"
    ]	#	sample output to parse:	"o1-mini gave GPT-4o-mini a grade of A-/B+"	==>	"o1-mini" , "GPT-4o-mini" , "A-/B+"
    
    for pattern in grade_patterns:
        for line in feedback.split('\n'):
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group(1)
    
    # If we've reached here, no grade was found
    print(f"  Warning: Could not extract grade from feedback. Using N/A.")
    return "N/A"  # Grade not found


def grade_to_numeric(grade: str) -> float:
    """Convert letter grade to numeric value for calculating statistics."""
    grade_map = {
        "A+": 4.25, "A": 4.0, "A-": 3.75,
        "B+": 3.25, "B": 3.0, "B-": 2.75,
        "C+": 2.25, "C": 2.0, "C-": 1.75,
        "D+": 1.25, "D": 1.0, "D-": 0.75,
        "F": 0.0,
        "N/A": 0.0
    }
#    return grade_map.get(grade, 0.0)
    return ( ( LetterGrade( grade ) ).numeric_value )  #####


def grade_to_percentage(numeric_grade: float) -> int:
    """Convert numeric grade to percentage (0-100 scale).
    
    This creates a more granular scale that spreads the grades across
    a wider range for better visualization and comparison:
    
    A+ (4.25) -> 97-100
    A  (4.0) -> 93-96
    A- (3.75) -> 90-92
    B+ (3.25) -> 87-89
    B  (3.0) -> 83-86
    B- (2.75) -> 80-82
    C+ (2.25) -> 77-79
    C  (2.0) -> 73-76
    C- (1.75) -> 70-72
    D+ (1.25) -> 67-69
    D  (1.0) -> 63-66
    D- (0.75) -> 60-62
    F  (0.0) -> 0-59
    """
    '''	#####
    if numeric_grade >= 4.5:  # A+
        return 100
    elif numeric_grade >= 4.0:  # A
        return 93 + int((numeric_grade - 4.0) / 0.3 * 4)
    elif numeric_grade >= 3.75:  # A-
        return 90 + int((numeric_grade - 3.7) / 0.3 * 3)
    elif numeric_grade >= 3.25:  # B+
        return 87 + int((numeric_grade - 3.3) / 0.4 * 3)
    elif numeric_grade >= 3.0:  # B
        return 83 + int((numeric_grade - 3.0) / 0.3 * 4)
    elif numeric_grade >= 2.75:  # B-
        return 80 + int((numeric_grade - 2.7) / 0.3 * 3)
    elif numeric_grade >= 2.25:  # C+
        return 77 + int((numeric_grade - 2.3) / 0.4 * 3)
    elif numeric_grade >= 2.0:  # C
        return 73 + int((numeric_grade - 2.0) / 0.3 * 4)
    elif numeric_grade >= 1.75:  # C-
        return 70 + int((numeric_grade - 1.7) / 0.3 * 3)
    elif numeric_grade >= 1.25:  # D+
        return 67 + int((numeric_grade - 1.3) / 0.4 * 3)
    elif numeric_grade >= 1.0:  # D
        return 63 + int((numeric_grade - 1.0) / 0.3 * 4)
    elif numeric_grade >= 0.75:  # D-
        return 60 + int((numeric_grade - 0.7) / 0.3 * 3)
    else:  # F
        # Scale F grade from 0 to 59 based on the numeric value
        return max(0, int(numeric_grade * 59 / 0.7))
    '''
    return ( int( ( LetterGrade( numeric_grade ).numeric_value/4.25 + 1 ) * 50 ) )  #  (A+, F) => (100, 50)	#####  

def percentage_to_letter_grade(percentage: int) -> str:
    """Convert percentage (0-100 scale) to letter grade."""
    '''	#####
    if percentage > 97:
        return "A+"
    elif percentage >= 95:
        return "A"
    elif percentage >= 90:
        return "A-"
    elif percentage >= 87:
        return "B+"
    elif percentage >= 83:
        return "B" 
    elif percentage >= 80:
        return "B-"
    elif percentage >= 77:
        return "C+"
    elif percentage >= 73:
        return "C"
    elif percentage >= 70:
        return "C-"
    elif percentage >= 67:
        return "D+"
    elif percentage >= 63:
        return "D"
    elif percentage >= 60:
        return "D-"
    else:
        return "F"
    '''
    return ( str( LetterGrade( ( percentage * 0.02 - 1 ) * 4.25 ) ) )	#####

def load_domain(domain_name: str) -> Tuple[str, str, Dict[str, str]]:
    """Load the specified domain module and return its prompts."""
    try:
        domain_module = importlib.import_module(f"domains.{domain_name}")
        return (
            domain_module.ESSAY_PROMPT, 
            domain_module.GRADING_PROMPT,
            domain_module.DOMAIN_INFO
        )
    except (ImportError, AttributeError) as e:
        print(f"Error loading domain '{domain_name}': {e}")
        sys.exit(1)


def create_results_directory(domain_name: str, timestamp: str) -> Tuple[str, str]:
    """Create a timestamped directory for results."""
    # Create base results directory if it doesn't exist
    if not os.path.exists("results"):
        os.makedirs("results")
    
    # Create timestamped directory for this run
    run_dir = f"results/{timestamp}-{domain_name}"
    essays_dir = f"{run_dir}/essays"
    
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(essays_dir, exist_ok=True)
    
    return run_dir, essays_dir


def save_essay_with_grades(author: str, essay: str, grades: Dict[str, Dict[str, Any]], essays_dir: str) -> str:
    """Save an essay to a text file with grading feedback appended."""
    filename = f"{essays_dir}/{author}.md"
    
    with open(filename, 'w') as f:
        # Write essay header
        f.write(f"# Essay by {author}\n\n")
        f.write(essay)
        
        # Write grading feedback
        f.write("\n\n---\n\n")
        f.write("# Grading Feedback\n\n")
        
        for grader_name, grade_info in grades.items():
            f.write(f"## Graded by: {grader_name}\n\n")
            f.write(grade_info["feedback"])
            f.write(f"\n\n**Letter Grade:** {grade_info['grade']}\n")
#            f.write(f"**Numeric Grade:** {grade_info['numeric_grade']:.1f}\n\n")
            f.write(f"**Numeric Grade:** {grade_info['numeric_grade']:.2f}\n\n")
            f.write("---\n\n")
    
    return filename


def verify_model(model: Dict[str, str]) -> Tuple[bool, Dict[str, str]]:
    """Verify a single model is available on OpenRouter.
    
    This function is designed to be run concurrently.
    
    Returns:
        Tuple of (success, model) where success is a boolean indicating if model is available
    """
    model_name = model["name"]
    model_id = model["model_id"]
    
    print(f"  Testing model: {model_name}...")
    # Using a minimal prompt just to check if the model is available
    test_prompt = "Hello, please respond with a single word."
    response = call_openrouter_api(model_id, test_prompt)
    
    if "error" in response:
        print(f"  ❌ Model {model_name} ({model_id}) is not available: {response.get('error', {}).get('message', 'Unknown error')}")
        return (False, model)
        
    print(f"  ✅ Model {model_name} verified and available")
    return (True, model)


def verify_available_models() -> List[Dict[str, str]]:
    """Verify which models are available on OpenRouter and return only valid ones.
    Verifies models concurrently to speed up the process.
    
    Returns:
        List of verified model dictionaries
    """
    print("Verifying available models...")
    verified_models = []
    
    # Use ThreadPoolExecutor to verify models concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(MODELS))) as executor:
        # Submit all verification tasks
        future_to_model = {
            executor.submit(verify_model, model): model["name"]
            for model in MODELS
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                success, model = future.result()
                if success:
                    verified_models.append(model)
            except Exception as e:
                print(f"  Error verifying model {model_name}: {e}")
    
    if not verified_models:
        print("Error: No valid models available on OpenRouter.")
        sys.exit(1)
        
    return verified_models


def grade_essay(grader: Dict[str, str], author: str, essay: str, grading_prompt: str, max_retries: int) -> Dict[str, Any]:

    """Grade an essay with retry logic.
    
    This function is designed to be run concurrently.
    
    Returns:
        Dict containing grading results and metadata
    """
    grader_name = grader["name"]
    grader_id = grader["model_id"]
    
    result = {
        "grader_name": grader_name,
        "author": author,
        "feedback": None,
        "grade": None,
        "numeric_grade": 0.0,
        "response": None,
        "error": None,
        "duration": 0.0,
        "start_time": time.time()
    }
    
    print(f"  {grader_name} grading {author}'s essay...")
    formatted_grading_prompt = grading_prompt.format(essay=essay)
    
    for retry in range(max_retries):
        try:
            response = call_openrouter_api(grader_id, formatted_grading_prompt)

            if "error" in response:
                if retry < max_retries - 1:
                    print(f"  Error getting feedback from {grader_name}, retrying ({retry+1}/{max_retries})...")
                    time.sleep(2)
                    continue
                else:
                    result["error"] = f"Error after {max_retries} attempts: {response.get('error')}"
                    print(f"  Error getting feedback from {grader_name} after {max_retries} attempts, skipping.")
                    break
            
            feedback = response.get("choices", [{}])[0].get("message", {}).get("content", "")

            if not feedback:
                if retry < max_retries - 1:
                    print(f"  Received empty feedback from {grader_name}, retrying...")
                    continue
                else:
                    result["error"] = "Empty feedback after all retries"
                    break
            
            # Extract grade from feedback
            grade = extract_grade(feedback)
#            grade , grader , mdl = extract_grade(feedback)	#	grader and mdl should be equal to grader-name and author, respectively
            
            # Store results
            result["feedback"] = feedback
            result["grade"] = grade
#            result["numeric_grade"] = grade_to_numeric(grade)
            result["numeric_grade"] = LetterGrade(grade).numeric_value	#####
            result["response"] = response
            result["duration"] = time.time() - result["start_time"]
            
            # Format for display
            cost_display = ""
            if "cost_info" in response:
                cost_info = response["cost_info"]
                cost_display = f"(${cost_info['total_cost']:.4f})"
                
            time_display = f"in {result['duration']:.2f}s"
            print(f"  {grader_name} gave {author} a grade of {grade} {cost_display} {time_display}")
            
            return result
            
        except Exception as e:
            if retry < max_retries - 1:
                print(f"  Exception when getting feedback from {grader_name}: {e}, retrying ({retry+1}/{max_retries})...")
                time.sleep(2)
            else:
                result["error"] = f"Exception after {max_retries} attempts: {str(e)}"
                print(f"  Failed to get feedback from {grader_name} after {max_retries} attempts: {e}")

    # If we get here with no feedback, we failed after all retries
    result["duration"] = time.time() - result["start_time"]
    return result


def get_essay_from_model(model: Dict[str, str], essay_prompt: str, max_retries: int) -> Dict[str, Any]:
    """Get an essay from a single model with retry logic.
    
    This function is designed to be run concurrently.
    
    Returns:
        Dict containing the model name, essay content, response info, and timing data
    """
    model_name = model["name"]
    model_id = model["model_id"]
    result = {
        "model_name": model_name,
        "essay": None,
        "response": None,
        "error": None,
        "duration": 0.0,
        "start_time": time.time()
    }
    
    print(f"  Requesting essay from {model_name}...")
    
    for retry in range(max_retries):
        try:
            response = call_openrouter_api(model_id, essay_prompt)
            
            if "error" in response:
                if retry < max_retries - 1:
                    print(f"  Error getting essay from {model_name}, retrying ({retry+1}/{max_retries})...")
                    time.sleep(2)  # Wait longer between retries
                    continue
                else:
                    result["error"] = f"Error after {max_retries} attempts: {response.get('error')}"
                    print(f"  Error getting essay from {model_name} after {max_retries} attempts, skipping.")
                    break
            
            essay = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not essay:
                if retry < max_retries - 1:
                    print(f"  Received empty response from {model_name}, retrying...")
                    continue
                else:
                    result["error"] = "Empty response after all retries"
                    break
            
            # Success - store results
            result["essay"] = essay
            result["response"] = response
            result["duration"] = time.time() - result["start_time"]
            
            # Format for display
            cost_display = ""
            if "cost_info" in response:
                cost_info = response["cost_info"]
                cost_display = f"(${cost_info['total_cost']:.4f}, {cost_info['input_tokens'] + cost_info['output_tokens']} tokens)"
            
            time_display = f"in {result['duration']:.2f}s"
            print(f"  Received essay from {model_name} ({len(essay)} characters) {cost_display} {time_display}")
            
            return result
            
        except Exception as e:
            if retry < max_retries - 1:
                print(f"  Exception when getting essay from {model_name}: {e}, retrying ({retry+1}/{max_retries})...")
                time.sleep(2)
            else:
                result["error"] = f"Exception after {max_retries} attempts: {str(e)}"
                print(f"  Failed to get essay from {model_name} after {max_retries} attempts: {e}")
    
    # If we get here with no essay, we failed after all retries
    result["duration"] = time.time() - result["start_time"]
    return result


def run_boswell_test(domain_name: str, output_file: str, selected_models: List[str] = None, 
                 skip_verification: bool = False, max_retries: int = 2) -> Dict[str, Any]:
    """Run the full Boswell Test and return results."""
    # Start timing the entire process
    run_start_time = time.time()
    
    # Load domain
    essay_prompt, grading_prompt, domain_info = load_domain(domain_name)
    
    # Get verified models or use all models if verification is skipped
    if skip_verification:
        print("Skipping model verification (using all configured models)...")
        verified_models = MODELS
    else:
        verified_models = verify_available_models()
    
    # Filter models if needed
    models_to_use = verified_models
#    print( [ m for m in verified_models ] )
    print( 'total verified Models for this run =\t' , len( verified_models ) , '\n' , [ m for m in verified_models ] )	#####

    if selected_models:
        models_to_use = [m for m in verified_models if m["name"] in selected_models]
        if not models_to_use:
            print("Error: No valid models selected.")
            sys.exit(1)
    
    # Create timestamp for this run - used for both timing and directory naming
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Initialize cost and timing tracking
    cost_tracking = {
        "total_cost": 0.0,
        "total_tokens": 0,
        "total_duration": 0.0,
        "essay_costs": {},
        "grading_costs": {}
    }
    
    timing_tracking = {
        "start_time": timestamp,
        "step_durations": {
            "essay_generation": 0.0,
            "grading": 0.0,
            "analysis": 0.0,
            "file_generation": 0.0,
            "total": 0.0
        },
        "model_timing": {
            "essay": {},
            "grading": {}
        }
    }
    
    # Create a results lock to prevent race conditions when updating shared data
    results_lock = Lock()
    
    results = {
        "domain": domain_info,
        "essays": {},
        "grades": {},
        "summary": {},
        "essay_files": {},  # Track essay files
        "available_models": [m["name"] for m in verified_models],
        "selected_models": [m["name"] for m in models_to_use],
        "run_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cost": cost_tracking,  # Add cost tracking
        "timing": timing_tracking  # Add timing tracking
    }
    
    # Step 1: Get essays from each model concurrently
    print(f"\nStep 1: Collecting essays from each model for domain '{domain_info['name']}'...")
    essay_start_time = time.time()
    
    # Use ThreadPoolExecutor to run API calls concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(models_to_use))) as executor:
        # Start all essay generation tasks
        future_to_model = {
            executor.submit(get_essay_from_model, model, essay_prompt, max_retries): model["name"]
            for model in models_to_use
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                essay_result = future.result()
                
                # Skip models with errors
                if essay_result["error"] or not essay_result["essay"]:
                    continue
                
                # Thread-safe update of the results dictionary
                with results_lock:
                    # Store the essay
                    results["essays"][model_name] = essay_result["essay"]
                    
                    # Track timing
                    results["timing"]["model_timing"]["essay"][model_name] = essay_result["duration"]
                    
                    # Track costs if available
                    if "response" in essay_result and essay_result["response"] and "cost_info" in essay_result["response"]:
                        cost_info = essay_result["response"]["cost_info"]
                        results["cost"]["essay_costs"][model_name] = cost_info
                        results["cost"]["total_cost"] += cost_info["total_cost"]
                        results["cost"]["total_tokens"] += cost_info["input_tokens"] + cost_info["output_tokens"]
                        results["cost"]["total_duration"] += cost_info["duration"]
                
            except Exception as e:
                print(f"  Error processing results from {model_name}: {e}")
    
    # Record total time for essay generation
    results["timing"]["step_durations"]["essay_generation"] = time.time() - essay_start_time
    
    # Check if we have enough essays to continue
    if len(results["essays"]) < 2:
        print("Error: Not enough valid essays received to perform grading (minimum 2 required).")
        print("Saving partial results and exiting.")
        save_results(results, output_file)
        return results
    
    # Step 2: Grade each essay with each model concurrently
    print("\nStep 2: Grading essays...")
    grading_start_time = time.time()
    
    # Prepare grading tasks
    grading_tasks = []
    for grader in models_to_use:
        grader_name = grader["name"]
        
        # Skip models that didn't generate essays
        if grader_name not in results["essays"]:
            print(f"  Skipping grader {grader_name} as it did not generate an essay.")
            continue
        
        # Initialize structures for this grader
        with results_lock:
            if grader_name not in results["timing"]["model_timing"]["grading"]:
                results["timing"]["model_timing"]["grading"][grader_name] = {}
            
            results["grades"][grader_name] = {}
        
        # Create tasks for each author-grader pair (excluding self-grading)
        for author, essay in results["essays"].items():
#            if grader_name != author:  # Skip self-grading	#####	add self-grading back
                grading_tasks.append((grader, author, essay))
    
    # Use ThreadPoolExecutor to run grading concurrently
    # Limit to 10 concurrent tasks to avoid overwhelming the API
    max_concurrent_gradings = min(10, len(grading_tasks))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_gradings) as executor:
        # Start all grading tasks
        future_to_task = {
            executor.submit(grade_essay, grader, author, essay, grading_prompt, max_retries): (grader["name"], author)
            for grader, author, essay in grading_tasks
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_task):
            grader_name, author = future_to_task[future]
            try:
                grading_result = future.result()
                
                # Skip if there was an error
                if grading_result["error"] or not grading_result["feedback"]:
                    continue

#                print( '#####' , grader_name , author , grading_result[ 'feedback' ][ -11 : ] , grading_result[ 'grade' ] , grading_result[ 'numeric_grade' ] , '#####' )
                
                # Thread-safe update of the results dictionary
                with results_lock:
                    # Store grading results
                    results["grades"][grader_name][author] = {
                        "feedback": grading_result["feedback"],
                        "grade": grading_result["grade"],
                        "numeric_grade": grading_result["numeric_grade"],
                        "cost_info": grading_result["response"].get("cost_info", {})
                    }
                    
                    # Track timing
                    results["timing"]["model_timing"]["grading"][grader_name][author] = grading_result["duration"]
                    
                    # Track costs if available
                    if "response" in grading_result and grading_result["response"] and "cost_info" in grading_result["response"]:
                        cost_info = grading_result["response"]["cost_info"]
                        
                        # Initialize grading costs for this grader if not already there
                        if grader_name not in results["cost"]["grading_costs"]:
                            results["cost"]["grading_costs"][grader_name] = {}
                            
                        # Add this specific grading cost
                        results["cost"]["grading_costs"][grader_name][author] = cost_info
                        
                        # Update totals
                        results["cost"]["total_cost"] += cost_info["total_cost"]
                        results["cost"]["total_tokens"] += cost_info["input_tokens"] + cost_info["output_tokens"]
                        results["cost"]["total_duration"] += cost_info["duration"]
                
            except Exception as e:
                print(f"  Error processing grading results from {grader_name} for {author}: {e}")
    
    # Record total time for grading
    results["timing"]["step_durations"]["grading"] = time.time() - grading_start_time
    
    # Step 3: Calculate summary statistics
    print("\nStep 3: Calculating summary statistics...")
    analysis_start_time = time.time()
    
    for author in results["essays"].keys():
        numeric_grades = [
            results["grades"][grader][author]["numeric_grade"]
            for grader in results["grades"]
            if author in results["grades"][grader]
        ]
        
        if numeric_grades:
            results["summary"][author] = {
#                "median_numeric": median(numeric_grades),
                "median_numeric": LetterGrade.median(numeric_grades),
                "grades_received": [
                    results["grades"][grader][author]["grade"]
                    for grader in results["grades"]
                    if author in results["grades"][grader]
                ]
            }
    
    # Record time for analysis
    results["timing"]["step_durations"]["analysis"] = time.time() - analysis_start_time
    
    # Step 4: Create results directory and save essays with grading feedback
    print("\nStep 4: Creating results directory and saving artifacts...")
    file_gen_start_time = time.time()
    
    # Use the timestamp from the beginning of the run
    run_dir, essays_dir = create_results_directory(domain_name, timestamp)
    
    for author in results["essays"].keys():
        # Get all grades for this author
        author_grades = {
            grader: results["grades"][grader][author]
            for grader in results["grades"]
            if author in results["grades"][grader]
        }
        
        # Save essay with grades
        filename = save_essay_with_grades(
            author, 
            results["essays"][author], 
            author_grades,
            essays_dir
        )
        
        results["essay_files"][author] = filename
        print(f"  Saved essay and feedback for {author} to {filename}")
    
    # Save tables and other formats
    print("  Generating formatted tables...")

#    print( "##### results : #####" )
#    for k1 , v1 in results[ 'grades' ].items( ) :  #  grader_name
#        for k2 , v2 in v1. items( ) :  #  author : feedback , grade , numeric_grade
#                    print( k1 , ':' , k2 , ':' , v2[ 'feedback' ][ -11 : ] , ';' , v2[ 'grade' ] , ';' , v2[ 'numeric_grade' ] )

    generate_grade_tables(results, run_dir)
    
    # Save consolidated results
    save_results(results, f"{run_dir}/full_results.json")
    print(f"  All results saved to {run_dir} directory")
    
    # Also save to original output file location if specified
    if output_file != "boswell_results.json":
        save_results(results, output_file)
    
    # Record time for file generation
    results["timing"]["step_durations"]["file_generation"] = time.time() - file_gen_start_time
    
    # Calculate and record total run time
    total_run_time = time.time() - run_start_time
    results["timing"]["step_durations"]["total"] = total_run_time
    
    # Return path to results directory in results
    results["results_dir"] = run_dir
    
    # We'll format the durations when needed rather than storing them
    # This avoids dependency issues when generating reports
    # The formatted values can still be calculated as needed from the step_durations
    
    return results


def save_results(results: Dict[str, Any], filename: str) -> None:
    """Save results to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filename}")


def calculate_grading_bias(results: Dict[str, Any], models: List[str]) -> Dict[str, Any]:
    """Calculate grading bias for each model."""
    # For each grader, collect all grades they gave
    grader_grades = {}
    for grader in models:
        if grader not in results["grades"]:
            continue
            
        # Convert letter grades to numeric for easier calculation
        numeric_grades = []
        for author, grade_info in results["grades"][grader].items():
            numeric_grades.append(grade_info["numeric_grade"])
            
        if numeric_grades:
            grader_grades[grader] = {
                "numeric_grades": numeric_grades,
#                "median_given": median(numeric_grades),
                "median_given": LetterGrade.median(numeric_grades),
#                "mean_given": sum(numeric_grades) / len(numeric_grades),
                "mean_given": LetterGrade.mean(numeric_grades),
                "count": len(numeric_grades)
            }

    # Calculate overall median of all grades
    all_numeric_grades = []
    for grader in grader_grades:
        all_numeric_grades.extend(grader_grades[grader]["numeric_grades"])
    
#    overall_median = median(all_numeric_grades)
    overall_median = LetterGrade.median(all_numeric_grades)
#    overall_mean = sum(all_numeric_grades) / len(all_numeric_grades)
    overall_mean = LetterGrade.mean(all_numeric_grades)
    
#    print( '#####' , grader , grader_grades[ grader ] , overall_median , overall_mean , all_numeric_grades , '#####' )

    # Calculate bias (difference from overall median)
    bias_results = {
        "overall_median": overall_median,
        "overall_mean": overall_mean,
        "grader_bias": {}
    }
    
    for grader, stats in grader_grades.items():
        # Positive bias means more lenient, negative means stricter
#        median_bias = stats["median_given"] - overall_median
        median_bias = LetterGrade( stats["median_given"] ).numeric_value - LetterGrade( overall_median ).numeric_value
#        mean_bias = stats["mean_given"] - overall_mean
        mean_bias = LetterGrade( stats["mean_given"] ).numeric_value - LetterGrade( overall_mean ).numeric_value

#        print( '#####' , grader , median_bias , stats[ 'median_given' ] , mean_bias , stats[ 'mean_given' ] , '#####' )
        
        # Convert numeric bias back to letter grade offset
        letter_bias = "Neutral"
        if abs(median_bias) < 0.15:
            letter_bias = "Neutral"
        elif median_bias > 0:
            if median_bias > 0.6:
                letter_bias = "Very Lenient (+2 grades)"
            elif median_bias > 0.3:
                letter_bias = "Lenient (+1 grade)"
            else:
                letter_bias = "Slightly Lenient (+1/3 grade)"
        else:
            if median_bias < -0.6:
                letter_bias = "Very Strict (-2 grades)"
            elif median_bias < -0.3:
                letter_bias = "Strict (-1 grade)"
            else:
                letter_bias = "Slightly Strict (-1/3 grade)"
        
        bias_results["grader_bias"][grader] = {
            "median_given": stats["median_given"],
            "median_bias": median_bias,
            "mean_bias": mean_bias,
            "letter_bias": letter_bias,
            "count": stats["count"]
        }
    
    return bias_results


def generate_bias_ascii_table(bias_results: Dict[str, Any]) -> str:
    """Generate an ASCII table showing grading bias for each model."""
    # Calculate column widths
    model_width = max(len(grader) for grader in bias_results["grader_bias"].keys()) + 2
    bias_width = 20  # Enough for bias description
    grade_width = 5   # For letter grade
    score_width = 5   # For percentage score
    
    # Build header
    header = f"{'Model':{model_width}} | {'Grade':{grade_width}} | {'Score':{score_width}} | {'Grading Bias':{bias_width}} |"
    
    # Build separator
    separator = "-" * model_width + "+" + "-" * (grade_width + 2) + "+" + "-" * (score_width + 2) + "+" + "-" * (bias_width + 2) + "+"
    
    # Build rows
    rows = []
    for grader, stats in bias_results["grader_bias"].items():
        # Convert numeric grade to letter and percentage
        median_grade = stats["median_given"]
        percentage = grade_to_percentage(median_grade)
        median_letter = percentage_to_letter_grade(percentage)
                
        row = f"{grader:{model_width}} | {median_letter:^{grade_width}} | {percentage:^{score_width}} | {stats['letter_bias']:{bias_width}} |"
        rows.append(row)
    
    # Add overall median row
    overall_median = bias_results["overall_median"]
    overall_percentage = grade_to_percentage(overall_median)
    overall_median_letter = percentage_to_letter_grade(overall_percentage)
            
    overall_row = f"{'OVERALL':{model_width}} | {overall_median_letter:^{grade_width}} | {overall_percentage:^{score_width}} | {'Baseline':{bias_width}} |"
    
    # Combine everything
    return f"{header}\n{separator}\n" + "\n".join(rows) + f"\n{separator}\n{overall_row}"


def generate_bias_markdown_table(bias_results: Dict[str, Any]) -> str:
    """Generate a Markdown table showing grading bias for each model."""
    # Build header
    header = "| Model | Grade | Score | Grading Bias | Numeric Bias |"
    
    # Build separator
    separator = "|------|-------|-------|-------------|-------------|"
    
    # Build rows
    rows = []
    for grader, stats in bias_results["grader_bias"].items():
        # Convert numeric grade to letter and percentage
        median_grade = stats["median_given"]
        percentage = grade_to_percentage(median_grade)
        median_letter = percentage_to_letter_grade(percentage)
                
        row = f"| {grader} | {median_letter} | {percentage} | {stats['letter_bias']} | {stats['median_bias']:.2f} |"
        rows.append(row)
    
    # Add overall median row
    overall_median = bias_results["overall_median"]
    overall_percentage = grade_to_percentage(overall_median)
    overall_median_letter = percentage_to_letter_grade(overall_percentage)
            
    overall_row = f"| **OVERALL** | {overall_median_letter} | {overall_percentage} | **Baseline** | 0.00 |"
    
    # Combine everything
    return f"{header}\n{separator}\n" + "\n".join(rows) + f"\n{overall_row}"


def generate_cost_report(results: Dict[str, Any]) -> str:
    """Generate a detailed cost report for the run."""
    cost_data = results["cost"]
    
    # Format total cost and duration
    total_cost = cost_data["total_cost"]
    total_duration = cost_data["total_duration"]
    total_tokens = cost_data["total_tokens"]
    
    report = []
    report.append("# Boswell Test Cost Report\n")
    report.append(f"Run timestamp: {results['run_timestamp']}\n")
    report.append(f"Domain: {results['domain']['name']}\n")
    report.append(f"## Summary\n")
    report.append(f"- **Total cost**: ${total_cost:.4f}")
    report.append(f"- **Total tokens**: {total_tokens:,}")
    report.append(f"- **Total duration**: {total_duration:.2f} seconds\n")
    
    # Essay Generation Costs
    report.append(f"## Essay Generation Costs\n")
    report.append("| Model | Tokens | Cost | Duration (s) |")
    report.append("|-------|--------|------|--------------|")
    
    essay_total_cost = 0
    for model_name, cost_info in cost_data["essay_costs"].items():
        tokens = cost_info["input_tokens"] + cost_info["output_tokens"]
        cost = cost_info["total_cost"]
        duration = cost_info["duration"]
        essay_total_cost += cost
        
        report.append(f"| {model_name} | {tokens:,} | ${cost:.4f} | {duration:.2f} |")
    
    report.append(f"| **TOTAL** | | **${essay_total_cost:.4f}** | |\n")
    
    # Grading Costs
    report.append(f"## Grading Costs\n")
    report.append("| Grader | Essays Graded | Total Tokens | Total Cost | Avg. Cost per Essay |")
    report.append("|--------|---------------|--------------|------------|---------------------|")
    
    grading_total_cost = 0
    for grader_name, gradings in cost_data["grading_costs"].items():
        essays_graded = len(gradings)
        grader_tokens = sum(info["input_tokens"] + info["output_tokens"] for info in gradings.values())
        grader_cost = sum(info["total_cost"] for info in gradings.values())
        avg_cost = grader_cost / essays_graded if essays_graded > 0 else 0
        grading_total_cost += grader_cost
        
        report.append(f"| {grader_name} | {essays_graded} | {grader_tokens:,} | ${grader_cost:.4f} | ${avg_cost:.4f} |")
    
    report.append(f"| **TOTAL** | | | **${grading_total_cost:.4f}** | |\n")
    
    # Cost breakdown by phase
    report.append(f"## Cost Breakdown by Phase\n")
    report.append("| Phase | Cost | Percentage |")
    report.append("|-------|------|------------|")
    
    essay_percent = (essay_total_cost / total_cost) * 100 if total_cost > 0 else 0
    grading_percent = (grading_total_cost / total_cost) * 100 if total_cost > 0 else 0
    
    report.append(f"| Essay Generation | ${essay_total_cost:.4f} | {essay_percent:.1f}% |")
    report.append(f"| Grading | ${grading_total_cost:.4f} | {grading_percent:.1f}% |")
    report.append(f"| **TOTAL** | **${total_cost:.4f}** | **100%** |")
    
    return "\n".join(report)


def generate_timing_report(results: Dict[str, Any]) -> str:
    """Generate a detailed timing report for the run."""
    timing_data = results["timing"]
    
    report = []
    report.append("# Boswell Test Timing Report\n")
    report.append(f"Run timestamp: {results['run_timestamp']}\n")
    report.append(f"Domain: {results['domain']['name']}\n")
    
    report.append("## Summary\n")
    
    # Format step durations
    durations = timing_data["step_durations"]
    total = durations["total"]
    
    # Handle case where total might be zero
    if total <= 0:
        # If total is zero, just show the durations without percentages
#        report.append(f"- **Total run time**: 0.0s")
        report.append(f"- **Total run time**: {total:.2f}s")	#####
        report.append(f"- **Essay generation**: {durations['essay_generation']:.1f}s")
        report.append(f"- **Grading**: {durations['grading']:.1f}s")
        report.append(f"- **Analysis**: {durations['analysis']:.1f}s")
        report.append(f"- **File generation**: {durations['file_generation']:.1f}s\n")
    else:
        # Format durations manually - don't rely on 'formatted' which might not exist yet
        minutes, seconds = divmod(total, 60)
        formatted_total = f"{int(minutes)}m {seconds:.1f}s"
        
        report.append(f"- **Total run time**: {formatted_total}")
        report.append(f"- **Essay generation**: {durations['essay_generation']:.1f}s ({durations['essay_generation']/total*100:.1f}%)")
        report.append(f"- **Grading**: {durations['grading']:.1f}s ({durations['grading']/total*100:.1f}%)")
        report.append(f"- **Analysis**: {durations['analysis']:.1f}s ({durations['analysis']/total*100:.1f}%)")
        report.append(f"- **File generation**: {durations['file_generation']:.1f}s ({durations['file_generation']/total*100:.1f}%)\n")
    
    # Essay generation timing by model
    report.append("## Essay Generation Timing\n")
    report.append("| Model | Duration (s) |")
    report.append("|-------|-------------|")
    
    for model, duration in timing_data["model_timing"]["essay"].items():
        report.append(f"| {model} | {duration:.2f} |")
    
    # Calculate average essay generation time
    if timing_data["model_timing"]["essay"]:
        avg_essay_time = sum(timing_data["model_timing"]["essay"].values()) / len(timing_data["model_timing"]["essay"])
        report.append(f"| **Average** | **{avg_essay_time:.2f}** |\n")
    
    # Grading timing by model
    report.append("## Grading Timing\n")
    report.append("| Grader | Essays Graded | Total Duration (s) | Avg. Seconds per Essay |")
    report.append("|--------|---------------|-------------------|------------------------|")
    
    for grader, gradings in timing_data["model_timing"]["grading"].items():
        essays_graded = len(gradings)
        total_time = sum(gradings.values())
        avg_time = total_time / essays_graded if essays_graded > 0 else 0
        
        report.append(f"| {grader} | {essays_graded} | {total_time:.2f} | {avg_time:.2f} |")
    
    return "\n".join(report)


def generate_visualizations(results: Dict[str, Any], run_dir: str) -> None:
    """Generate data visualizations of the test results."""
    models = list(results["essays"].keys())
    
    # Create a charts directory
    charts_dir = f"{run_dir}/charts"
    os.makedirs(charts_dir, exist_ok=True)
    
    # 1. Chart: Grading bias visualization
    plt.figure(figsize=(12, 8))
    
    bias_data = results.get("bias_analysis", {}).get("grader_bias", {})
    if bias_data:
        graders = list(bias_data.keys())
        median_bias = [bias_data[grader]["median_bias"] for grader in graders]
        
        # Sort by bias value for better visualization
        sorted_indices = np.argsort(median_bias)
        sorted_graders = [graders[i] for i in sorted_indices]
        sorted_bias = [median_bias[i] for i in sorted_indices]
        
        # Color bars based on bias (red for negative/strict, green for positive/lenient)
        colors = ['#FF6B6B' if bias < 0 else '#4ECDC4' for bias in sorted_bias]
        
        plt.bar(sorted_graders, sorted_bias, color=colors)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Grading Bias by Model')
        plt.xlabel('Model')
        plt.ylabel('Bias (Negative = Stricter, Positive = More Lenient)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{charts_dir}/grading_bias.png", dpi=300)
        plt.close()
    
    # 2. Chart: Grade distribution
    plt.figure(figsize=(12, 8))
    
    # Collect all numeric grades
    numeric_grades = {}
    for model in models:
        grades = []
        for grader, grades_given in results.get("grades", {}).items():
            if model in grades_given:
                grades.append(grades_given[model]["numeric_grade"])
        if grades:
            numeric_grades[model] = grades

#    print( '##### nmrcGrds : ' , models , ':' , numeric_grades , '#####' )
    
    if numeric_grades:
        # Create boxplot of grades received by each model
        plt.boxplot([numeric_grades[model] for model in models], 
                   tick_labels=models, 
                   patch_artist=True,
                   boxprops=dict(facecolor='#A8DADC'))
        
        plt.title('Distribution of Grades Received by Each Model')
        plt.ylabel('Numeric Grade')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{charts_dir}/grade_distribution.png", dpi=300)
        plt.close()
    
    # 3. Chart: Timing comparison
    plt.figure(figsize=(12, 8))
    
    timing_data = results.get("timing", {}).get("model_timing", {}).get("essay", {})
    if timing_data:
        model_names = list(timing_data.keys())
        times = list(timing_data.values())
        
        # Sort by time for better visualization
        sorted_indices = np.argsort(times)
        sorted_models = [model_names[i] for i in sorted_indices]
        sorted_times = [times[i] for i in sorted_indices]
        
        plt.barh(sorted_models, sorted_times, color='#457B9D')
        plt.title('Time to Generate Essay by Model')
        plt.xlabel('Time (seconds)')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{charts_dir}/essay_generation_time.png", dpi=300)
        plt.close()
    
    # 4. Chart: Grading time comparison
    plt.figure(figsize=(12, 8))
    
    grading_data = results.get("timing", {}).get("model_timing", {}).get("grading", {})
    if grading_data:
        # Calculate average grading time for each model
        avg_grading_times = {}
        for grader, times in grading_data.items():
            if times:
                avg_grading_times[grader] = sum(times.values()) / len(times)
        
        if avg_grading_times:
            graders = list(avg_grading_times.keys())
            avg_times = list(avg_grading_times.values())
            
            # Sort by time for better visualization
            sorted_indices = np.argsort(avg_times)
            sorted_graders = [graders[i] for i in sorted_indices]
            sorted_avg_times = [avg_times[i] for i in sorted_indices]
            
            plt.barh(sorted_graders, sorted_avg_times, color='#1D3557')
            plt.title('Average Grading Time by Model')
            plt.xlabel('Time (seconds)')
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f"{charts_dir}/average_grading_time.png", dpi=300)
            plt.close()
    
    # 5. Chart: Cost breakdown
    plt.figure(figsize=(12, 8))
    
    # Essay costs
    essay_costs = {}
    for model, cost_info in results.get("cost", {}).get("essay_costs", {}).items():
        essay_costs[model] = cost_info.get("total_cost", 0)
    
    # Grading costs
    grading_costs = {}
    for grader, gradings in results.get("cost", {}).get("grading_costs", {}).items():
        grading_costs[grader] = sum(info.get("total_cost", 0) for info in gradings.values())
    
    if essay_costs and grading_costs:
        # Prepare data for stacked bar chart
        models_with_both = [m for m in models if m in essay_costs and m in grading_costs]
        essay_cost_values = [essay_costs.get(m, 0) for m in models_with_both]
        grading_cost_values = [grading_costs.get(m, 0) for m in models_with_both]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bar_width = 0.5
        
        # Create stacked bars
        ax.bar(models_with_both, essay_cost_values, bar_width, 
               label='Essay Generation', color='#E9C46A')
        ax.bar(models_with_both, grading_cost_values, bar_width,
               bottom=essay_cost_values, label='Grading', color='#F4A261')
        
        ax.set_title('Cost Breakdown by Model')
        ax.set_xlabel('Model')
        ax.set_ylabel('Cost ($)')
        ax.legend()
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{charts_dir}/cost_breakdown.png", dpi=300)
        plt.close()
    
    # 6. Chart: Process timing breakdown
    plt.figure(figsize=(10, 6))
    
    durations = results.get("timing", {}).get("step_durations", {})
    if durations:
        steps = ['essay_generation', 'grading', 'analysis', 'file_generation']
        step_names = ['Essay Generation', 'Grading', 'Analysis', 'File Generation']
        times = [durations.get(step, 0) for step in steps]
        
        # Check if we have any non-zero times to create a meaningful pie chart
        if sum(times) > 0:
            # Create pie chart
            plt.pie(times, labels=step_names, autopct='%1.1f%%', 
                    startangle=90, shadow=False, 
                    colors=['#264653', '#2A9D8F', '#E9C46A', '#F4A261'])
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            plt.title('Time Breakdown by Process Phase')
        else:
            # If all times are zero, create a placeholder message
            plt.text(0.5, 0.5, 'No timing data available', 
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=14)
            plt.title('Time Breakdown by Process Phase')
            plt.axis('off')  # Hide axes
        
        plt.tight_layout()
        plt.savefig(f"{charts_dir}/time_breakdown.png", dpi=300)
        plt.close()
    
    # 7. Chart: Boswell Quotient visualization
    if "boswell_quotient" in results and "model_scores" in results["boswell_quotient"]:
        plt.figure(figsize=(14, 8))
        
        quotient_data = results["boswell_quotient"]["model_scores"]
        
        # Sort models by Boswell Quotient score (descending)
        sorted_models = sorted(
            quotient_data.keys(),
            key=lambda m: quotient_data[m]["boswell_quotient"],
            reverse=True
        )
        
        # Prepare data for visualization
        scores = [quotient_data[model]["boswell_quotient"] for model in sorted_models]
        
        # Create bar chart for overall Boswell Quotient
        plt.barh(sorted_models, scores, color='#6A5ACD')
        plt.title('Boswell Quotient by Model', fontsize=16)
        plt.xlabel('Boswell Quotient (0-100)', fontsize=12)
        plt.yticks(fontsize=10)
        plt.xlim(0, 100)  # Scale from 0 to 100
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{charts_dir}/boswell_quotient.png", dpi=300)
        plt.close()
        
        # 8. Chart: Boswell Quotient component breakdown
        plt.figure(figsize=(14, 10))
        
        # Prepare component data
        components = ["performance", "evaluation", "efficiency"]
        component_data = {component: [] for component in components}
        
        for model in sorted_models:
            model_components = quotient_data[model].get("components", {})
            for component in components:
                component_data[component].append(model_components.get(component, 0))
        
        # Set up the bar chart
        bar_width = 0.25
        index = np.arange(len(sorted_models))
        
        # Plot each component
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Set colors for components
        colors = ['#4CAF50', '#2196F3', '#FF9800']  # Green, Blue, Orange
        
        # Plot bars for each component
        for i, component in enumerate(components):
            ax.barh(index + i*bar_width, component_data[component], 
                   bar_width, label=component.capitalize(), color=colors[i])
        
        # Configure axes and labels
        ax.set_yticks(index + bar_width)
        ax.set_yticklabels(sorted_models)
        ax.set_xlabel('Component Score (0-100)')
        ax.set_title('Boswell Quotient Component Breakdown by Model')
        ax.set_xlim(0, 100)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f"{charts_dir}/boswell_quotient_components.png", dpi=300)
        plt.close()
    
    print(f"  Visualizations saved to {charts_dir}")

def calculate_boswell_quotient(results: Dict[str, Any], models: List[str]) -> Dict[str, Any]:
    """Calculate the Boswell Quotient for each model.
    
    The Boswell Quotient is a composite score (0-100) based on:
    1. Performance Score (50%): Based on grades received from other models
    2. Evaluation Score (30%): Based on grading accuracy and consistency
    3. Efficiency Score (20%): Based on timing and resource usage
    
    Returns:
        Dictionary with Boswell Quotient scores and component scores for each model
    """
    quotient_results = {
        "model_scores": {},
        "component_weights": {
            "performance": 0.50,  # 50% weight for performance
            "evaluation": 0.30,   # 30% weight for evaluation capability
            "efficiency": 0.20    # 20% weight for efficiency
        }
    }
    
    # Collect all performance scores (grades received)
    performance_scores = {}
#    max_performance_score = 4.3  # A+ = 4.3
    max_performance_score = 4.25  # A+ = 4.25
    
    for model in models:
        if model in results["summary"]:
            # Get median numeric grade received (on 0-4.3 scale)
            median_grade = results["summary"][model]["median_numeric"]
            # Convert to 0-100 scale for the performance component
#            performance_score = (median_grade / max_performance_score) * 100
            performance_score = 50 * ( 1 + LetterGrade( median_grade ).numeric_value / 4.25 )	#	50 - 100
            performance_scores[model] = performance_score
    
    # Collect evaluation scores (based on grading accuracy and bias)
    evaluation_scores = {}
    if "bias_analysis" in results and "grader_bias" in results["bias_analysis"]:
        bias_data = results["bias_analysis"]["grader_bias"]

#        print( '##### : bias_data' , bias_data , '#####' )
        
        for model in models:
            if model in bias_data:
                # Calculate how close the model's grading is to the consensus
                # A bias of 0 is perfect, higher absolute bias reduces score
                bias_magnitude = abs(bias_data[model]["median_bias"])
                
                # Convert bias to a score (0-100)
                # Perfect score for no bias, decreasing as bias increases
                # Bias of 0.7 (very lenient/strict) gives ~40% evaluation score
                # Bias of 0.3 (slightly lenient/strict) gives ~70% evaluation score
                evaluation_score = max(0, 100 - (bias_magnitude * 100))
                evaluation_scores[model] = evaluation_score
    
    # Collect efficiency scores (based on timing and tokens)
    efficiency_scores = {}
    timing_data = results.get("timing", {}).get("model_timing", {})
    cost_data = results.get("cost", {})
    
    if timing_data and "essay" in timing_data and "grading" in timing_data:
        # Calculate max response times to normalize
        essay_times = [timing_data["essay"].get(model, 0) for model in models if model in timing_data["essay"]]
        max_essay_time = max(essay_times) if essay_times else 1
        
        # Calculate average grading times
        avg_grading_times = {}
        for model in models:
            if model in timing_data["grading"]:
                times = timing_data["grading"][model]
                if times:
                    avg_grading_times[model] = sum(times.values()) / len(times)
        
        max_grading_time = max(avg_grading_times.values()) if avg_grading_times else 1
        
        # Calculate efficiency scores
        for model in models:
            essay_time_score = 0
            grading_time_score = 0
            
            # Essay generation efficiency (faster is better)
            if model in timing_data["essay"]:
                # Inverse relationship - faster times get higher scores
                essay_time = timing_data["essay"][model]
                essay_time_score = max(0, 100 * (1 - (essay_time / max_essay_time)))
            
            # Grading efficiency
            if model in avg_grading_times:
                # Inverse relationship - faster times get higher scores
                grading_time = avg_grading_times[model]
                grading_time_score = max(0, 100 * (1 - (grading_time / max_grading_time)))
            
            # Combine essay and grading efficiency (equal weight)
            if essay_time_score > 0 or grading_time_score > 0:
                weights = []
                scores = []
                
                if essay_time_score > 0:
                    weights.append(0.5)
                    scores.append(essay_time_score)
                
                if grading_time_score > 0:
                    weights.append(0.5)
                    scores.append(grading_time_score)
                
                total_weight = sum(weights)
                if total_weight > 0:
                    efficiency_scores[model] = sum(w * s for w, s in zip(weights, scores)) / total_weight
    
    # Calculate the final Boswell Quotient for each model
    for model in models:
        components = {}
        weighted_scores = []
        weights = []
        
        # Add performance component if available
        if model in performance_scores:
            performance = performance_scores[model]
            components["performance"] = performance
            weighted_scores.append(performance * quotient_results["component_weights"]["performance"])
            weights.append(quotient_results["component_weights"]["performance"])
        
        # Add evaluation component if available
        if model in evaluation_scores:
            evaluation = evaluation_scores[model]
            components["evaluation"] = evaluation
            weighted_scores.append(evaluation * quotient_results["component_weights"]["evaluation"])
            weights.append(quotient_results["component_weights"]["evaluation"])
        
        # Add efficiency component if available
        if model in efficiency_scores:
            efficiency = efficiency_scores[model]
            components["efficiency"] = efficiency
            weighted_scores.append(efficiency * quotient_results["component_weights"]["efficiency"])
            weights.append(quotient_results["component_weights"]["efficiency"])
        
        # Calculate final score if we have any components
        if weighted_scores:
            total_weight = sum(weights)
            if total_weight > 0:
                # Normalize by actual weights used
                final_score = sum(weighted_scores) / total_weight
                
                # Store results
                quotient_results["model_scores"][model] = {
                    "boswell_quotient": round(final_score, 1),
                    "components": components,
                    "rank": 0  # Will be filled in later
                }
    
    # Rank models by Boswell Quotient
    ranked_models = sorted(
        quotient_results["model_scores"].keys(),
        key=lambda m: quotient_results["model_scores"][m]["boswell_quotient"],
        reverse=True
    )
    
    # Assign ranks (1-based)
    for i, model in enumerate(ranked_models):
        quotient_results["model_scores"][model]["rank"] = i + 1
    
    return quotient_results


def generate_boswell_quotient_table(quotient_results: Dict[str, Any]) -> str:
    """Generate a Markdown table showing the Boswell Quotient for each model."""
    # Build header
    header = "| Rank | Model | Boswell Quotient | Grade | Performance | Evaluation | Efficiency |"
    
    # Build separator
    separator = "|------|-------|-----------------|-------|------------|------------|------------|"
    
    # Build rows
    rows = []
    sorted_models = sorted(
        quotient_results["model_scores"].keys(),
        key=lambda m: quotient_results["model_scores"][m]["rank"]
    )
    
    for model in sorted_models:
        score_data = quotient_results["model_scores"][model]
        components = score_data["components"]
        
        rank = score_data["rank"]
        quotient = score_data["boswell_quotient"]
        
        # Add letter grade equivalent of the Boswell Quotient
        letter_grade = percentage_to_letter_grade(int(quotient))
        
        performance = components.get("performance", "N/A")
        if performance != "N/A":
            performance = f"{performance:.1f}"
        
        evaluation = components.get("evaluation", "N/A")
        if evaluation != "N/A":
            evaluation = f"{evaluation:.1f}"
            
        efficiency = components.get("efficiency", "N/A")
        if efficiency != "N/A":
            efficiency = f"{efficiency:.1f}"
        
        rows.append(f"| {rank} | {model} | {quotient:.1f} | {letter_grade} | {performance} | {evaluation} | {efficiency} |")
    
    # Combine everything
    return f"{header}\n{separator}\n" + "\n".join(rows)


def generate_boswell_report(results: Dict[str, Any], quotient_results: Dict[str, Any]) -> str:
    """Generate a detailed report on the Boswell Quotient results.
    
    The report includes:
    1. Introduction to the Boswell Quotient
    2. Overall ranking of models
    3. Component breakdown and analysis
    4. Insights and observations
    """
    domain_info = results["domain"]
    timestamp = results["run_timestamp"]
    
    # Calculate some statistics for the report
    model_scores = quotient_results["model_scores"]
    sorted_models = sorted(
        model_scores.keys(),
        key=lambda m: model_scores[m]["rank"]
    )
    
    # Get top and bottom 3 models (or fewer if less than 6 models total)
    top_models = sorted_models[:min(3, len(sorted_models))]
    bottom_models = sorted_models[max(0, len(sorted_models)-3):]
    
    # Find performance leaders in each component
    component_leaders = {
        "performance": {"model": "", "score": 0},
        "evaluation": {"model": "", "score": 0},
        "efficiency": {"model": "", "score": 0}
    }
    
    for model in sorted_models:
        components = model_scores[model]["components"]
        for component in component_leaders:
            if component in components:
                score = components[component]
                if score > component_leaders[component]["score"]:
                    component_leaders[component]["model"] = model
                    component_leaders[component]["score"] = score
    
    # Generate the report
    lines = []
    
    # Title and introduction
    lines.append("# Boswell Quotient Analysis Report")
    lines.append(f"\nRun timestamp: {timestamp}")
    lines.append(f"Domain: {domain_info['name']}")
    
    # Introduction section
    lines.append("\n## Introduction to the Boswell Quotient")
    lines.append(
        "The Boswell Quotient is a comprehensive metric designed to evaluate LLM capabilities across multiple dimensions. "
        "Named after James Boswell, who was known for his deep understanding and insightful observations of Samuel Johnson, "
        "this quotient aims to measure how well an AI model can serve as an indispensable companion in academic and analytical tasks."
    )
    lines.append("\nThe Boswell Quotient (0-100) is calculated from three key components:")
    lines.append("1. **Performance (50%)**: How well the model performs in generating content, based on grades received from peers")
    lines.append("2. **Evaluation (30%)**: How accurately and consistently the model evaluates others' work")
    lines.append("3. **Efficiency (20%)**: How quickly and resource-efficiently the model completes tasks")
    
    # Overall ranking section
    lines.append("\n## Overall Model Rankings")
    lines.append("The table below shows all models ranked by their Boswell Quotient:")
    
    # Add the full table
    lines.append("\n" + generate_boswell_quotient_table(quotient_results))
    
    # Top performers section
    lines.append("\n## Top Performing Models")
    if top_models:
        top_model = top_models[0]
        top_score = model_scores[top_model]["boswell_quotient"]
        lines.append(f"\n**{top_model}** achieved the highest Boswell Quotient of **{top_score:.1f}**, ")
        
        # Add description of why the top model performed well
        components = model_scores[top_model]["components"]
        strengths = []
        
        if "performance" in components and components["performance"] > 85:
            strengths.append("exceptional content quality")
        elif "performance" in components and components["performance"] > 75:
            strengths.append("strong content quality")
            
        if "evaluation" in components and components["evaluation"] > 85:
            strengths.append("excellent evaluation capabilities")
        elif "evaluation" in components and components["evaluation"] > 75:
            strengths.append("good evaluation consistency")
            
        if "efficiency" in components and components["efficiency"] > 85:
            strengths.append("outstanding efficiency")
        elif "efficiency" in components and components["efficiency"] > 75:
            strengths.append("good response time")
            
        if strengths:
            lines.append(f"demonstrating {', '.join(strengths)}.")
        else:
            lines.append("with balanced performance across all components.")
        
        # Mention other top performers
        if len(top_models) > 1:
            other_tops = [f"**{m}** ({model_scores[m]['boswell_quotient']:.1f})" for m in top_models[1:]]
            lines.append(f"\nOther top performers include {', '.join(other_tops)}.")
    
    # Component analysis section
    lines.append("\n## Component Analysis")
    
    # Performance component
    lines.append("\n### Performance Component")
    performance_leader = component_leaders["performance"]["model"]
    performance_score = component_leaders["performance"]["score"]
    
    if performance_leader:
        lines.append(
            f"**{performance_leader}** leads in the Performance component with a score of **{performance_score:.1f}**. "
            f"This reflects the model's ability to generate high-quality content that receives favorable evaluations from peers."
        )
    
    # Evaluation component
    lines.append("\n### Evaluation Component")
    evaluation_leader = component_leaders["evaluation"]["model"]
    evaluation_score = component_leaders["evaluation"]["score"]
    
    if evaluation_leader:
        lines.append(
            f"**{evaluation_leader}** excels in the Evaluation component with a score of **{evaluation_score:.1f}**. "
            f"This indicates the model's ability to provide consistent and fair assessments that align with the consensus."
        )
    
    # Efficiency component
    lines.append("\n### Efficiency Component")
    efficiency_leader = component_leaders["efficiency"]["model"]
    efficiency_score = component_leaders["efficiency"]["score"]
    
    if efficiency_leader:
        lines.append(
            f"**{efficiency_leader}** demonstrates superior Efficiency with a score of **{efficiency_score:.1f}**. "
            f"This reflects the model's ability to generate responses and evaluations quickly while using resources effectively."
        )
    
    # Observations and insights
    lines.append("\n## Observations and Insights")
    
    # Look for models with balanced performance
    balanced_models = []
    for model in sorted_models:
        components = model_scores[model]["components"]
        if len(components) >= 3:  # Has all three components
            scores = [components.get(c, 0) for c in ["performance", "evaluation", "efficiency"]]
            if scores[0] > 0 and scores[1] > 0 and scores[2] > 0:
                # Check if all scores are within 15 points of each other
                if max(scores) - min(scores) <= 15:
                    balanced_models.append((model, model_scores[model]["boswell_quotient"]))
    
    if balanced_models:
        balanced_models.sort(key=lambda x: x[1], reverse=True)
        balanced_list = [f"**{m}** ({s:.1f})" for m, s in balanced_models[:3]]
        lines.append(
            f"The most balanced models across all components were {', '.join(balanced_list)}. "
            f"These models demonstrate well-rounded capabilities essential for being a reliable AI assistant."
        )
    
    # Look for interesting patterns or outliers
    performance_vs_evaluation = []
    for model in sorted_models:
        components = model_scores[model]["components"]
        if "performance" in components and "evaluation" in components:
            perf = components["performance"]
            eval_score = components["evaluation"]
            if abs(perf - eval_score) > 20:  # Significant difference
                performance_vs_evaluation.append((model, perf, eval_score))
    
    if performance_vs_evaluation:
        # Models that are better at performance than evaluation
        better_performers = [(m, p, e) for m, p, e in performance_vs_evaluation if p > e]
        if better_performers:
            example = better_performers[0]
            lines.append(
                f"\n**{example[0]}** and similar models demonstrate significantly higher content quality "
                f"({example[1]:.1f}) than evaluation consistency ({example[2]:.1f}). "
                f"This suggests these models excel at generating content but may need improvement in "
                f"their critical assessment capabilities."
            )
        
        # Models that are better at evaluation than performance
        better_evaluators = [(m, p, e) for m, p, e in performance_vs_evaluation if e > p]
        if better_evaluators:
            example = better_evaluators[0]
            lines.append(
                f"\n**{example[0]}** and similar models show stronger evaluation capabilities "
                f"({example[2]:.1f}) compared to content generation ({example[1]:.1f}). "
                f"These models may excel as critics or reviewers rather than primary content generators."
            )
    
    # Conclusion
    lines.append("\n## Conclusion")
    lines.append(
        "The Boswell Quotient provides a multi-dimensional assessment of AI models' capabilities, "
        "helping identify which models are best suited for different use cases. A high Boswell Quotient "
        "indicates a model that can effectively serve as a comprehensive AI assistant - generating high-quality "
        "content, providing accurate evaluations, and doing so efficiently."
    )
    
    if top_models:
        top_three = [f"**{m}**" for m in top_models[:min(3, len(top_models))]]
        lines.append(f"\nBased on this analysis, {', '.join(top_three)} stand out as the most capable AI assistants in this domain.")
    
    # Return the full report
    return "\n".join(lines)


def generate_grade_tables(results: Dict[str, Any], run_dir: str) -> None:
    """Generate formatted tables of grades in ASCII, Markdown, and CSV formats."""
    models = list(results["essays"].keys())
    
    # Calculate grading bias
    bias_results = calculate_grading_bias(results, models)
    
    # Add bias results to the main results
    results["bias_analysis"] = bias_results
    
    # Calculate Boswell Quotient
    quotient_results = calculate_boswell_quotient(results, models)
    
    # Add Boswell Quotient results to the main results
    results["boswell_quotient"] = quotient_results
    
    # Generate ASCII table
    ascii_table = generate_ascii_table(results, models)
    with open(f"{run_dir}/grades_table.txt", 'w') as f:
        f.write(ascii_table)
    
    # Generate Markdown table
    markdown_table = generate_markdown_table(results, models)
    with open(f"{run_dir}/grades_table.md", 'w') as f:
        f.write(markdown_table)
    
    # Generate CSV table
    csv_table = generate_csv_table(results, models)
    with open(f"{run_dir}/grades_table.csv", 'w') as f:
        f.write(csv_table)
    
    # Generate bias tables
    bias_ascii = generate_bias_ascii_table(bias_results)
    with open(f"{run_dir}/grading_bias.txt", 'w') as f:
        f.write(bias_ascii)
        
    bias_markdown = generate_bias_markdown_table(bias_results)
    with open(f"{run_dir}/grading_bias.md", 'w') as f:
        f.write(bias_markdown)
    
    # Generate Boswell Quotient table and report
    boswell_quotient_table = generate_boswell_quotient_table(quotient_results)
    boswell_report = generate_boswell_report(results, quotient_results)
    
    with open(f"{run_dir}/boswell_quotient.md", 'w') as f:
        f.write(boswell_quotient_table)
    
    with open(f"{run_dir}/boswell_report.md", 'w') as f:
        f.write(boswell_report)
    
    # Generate cost report
    cost_report = generate_cost_report(results)
    with open(f"{run_dir}/cost_report.md", 'w') as f:
        f.write(cost_report)
        
    # Generate timing report
    timing_report = generate_timing_report(results)
    with open(f"{run_dir}/timing_report.md", 'w') as f:
        f.write(timing_report)
    
    # Generate visualizations
    generate_visualizations(results, run_dir)
    
    # Generate JSON grades file (just the grades and summary)
    grades_json = {
        "domain": results["domain"],
        "grades": results["grades"],
        "summary": results["summary"],
        "bias_analysis": bias_results,
        "boswell_quotient": quotient_results,
        "cost": results["cost"],
        "run_timestamp": results["run_timestamp"]
    }
    with open(f"{run_dir}/grades.json", 'w') as f:
        json.dump(grades_json, f, indent=2)


def generate_ascii_table(results: Dict[str, Any], models: List[str]) -> str:
    """Generate an ASCII table of grades with percentage scores."""
    # Calculate column widths
    model_width = max(len(model) for model in models) + 2
    grade_width = 10  # Enough for "A+ (97)" format
    median_width = 7  # Just for the letter grade
    pct_width = 5     # For percentage
    
    # Build header
    header = f"{'Model':{model_width}} |"
    for grader in models:
        header += f" {grader[:grade_width-1]:{grade_width-1}} |"
    header += f" {'Median':{median_width}} | {'Pct':{pct_width}} |"
    
    # Build separator
    separator = "-" * model_width + "+"
    for _ in models:
        separator += "-" * (grade_width + 1) + "+"
    separator += "-" * (median_width + 2) + "+" + "-" * (pct_width + 2) + "+"
    
    # Build rows
    rows = []
    for author in models:
        row = f"{author:{model_width}} |"
        
        for grader in models:
            if grader == author:
                # Self-grade
                grade_display = "---"
            elif grader in results["grades"] and author in results["grades"][grader]:
                grade_data = results["grades"][grader][author]
                letter_grade = grade_data["grade"]
                numeric_grade = grade_data["numeric_grade"]
                percentage = grade_to_percentage(numeric_grade)
                grade_display = f"{letter_grade} ({percentage})"
            else:
                grade_display = "N/A"
            
            # Truncate if too long
            if len(grade_display) > grade_width - 1:
                grade_display = grade_display[:grade_width-2] + "…"
                
            row += f" {grade_display:{grade_width-1}} |"
        
        # Add median and percentage
        if author in results["summary"]:
            median_num = results["summary"][author]["median_numeric"]
            
            # Calculate percentage
            percentage = grade_to_percentage(median_num)
            
            # Convert numeric back to letter grade for display
            median_letter = percentage_to_letter_grade(percentage)
        else:
            median_letter = "N/A"
            percentage = "N/A"
            
        row += f" {median_letter:^{median_width}} | {percentage:^{pct_width}} |"
        rows.append(row)
    
    # Combine everything
    return f"{header}\n{separator}\n" + "\n".join(rows)


def generate_markdown_table(results: Dict[str, Any], models: List[str]) -> str:
    """Generate a Markdown table of grades with percentage scores."""
    # Build header
    header = "| Model | " + " | ".join(models) + " | Median Grade | Percentage |"
    
    # Build separator
    separator = "|------|" + "|".join(["---" for _ in models]) + "|-------------|-----------|"
    
    # Build rows
    rows = []
    for author in models:
        grades = []
        
        for grader in models:
            if grader == author:
                # Self-grade
                grades.append("---")
            elif grader in results["grades"] and author in results["grades"][grader]:
                grade_data = results["grades"][grader][author]
                letter_grade = grade_data["grade"]
                numeric_grade = grade_data["numeric_grade"]
                percentage = grade_to_percentage(numeric_grade)
                # Add percentage in parentheses
                grades.append(f"{letter_grade} ({percentage})")
            else:
                grades.append("N/A")
        
        # Add median
        if author in results["summary"]:
            median_num = results["summary"][author]["median_numeric"]
            
            # Calculate percentage
            percentage = grade_to_percentage(median_num)
            
            # Convert numeric back to letter grade for display
            median_letter = percentage_to_letter_grade(percentage)
            
            median_display = f"{median_letter} | {percentage}"
        else:
            median_display = "N/A | N/A"
            
        row = f"| {author} | " + " | ".join(grades) + f" | {median_display} |"
        rows.append(row)
    
    # Combine everything
    return f"{header}\n{separator}\n" + "\n".join(rows)


def generate_csv_table(results: Dict[str, Any], models: List[str]) -> str:
    """Generate a CSV table of grades with percentage scores."""
    # Build header
    header = "Model," + ",".join(models) + ",Median Grade,Percentage"
    
    # Build rows
    rows = []
    for author in models:
        grades = []
        
        for grader in models:
#            if grader == author:	#####	add self-grade
#                # Self-grade		#####	add self-grade
#                grades.append("---")	#####	add self-grade
#            elif grader in results["grades"] and author in results["grades"][grader]:	#####	add self-grade
            if grader in results["grades"] and author in results["grades"][grader]:
                grade_data = results["grades"][grader][author]
                letter_grade = grade_data["grade"]
                numeric_grade = grade_data["numeric_grade"]
                percentage = grade_to_percentage(numeric_grade)
                # Add percentage in parentheses for CSV
#                grades.append(f"{letter_grade} ({percentage})")
                grades.append(f"{letter_grade}")	#####
            else:
                grades.append("N/A")
        
        # Add median
        if author in results["summary"]:
            median_num = results["summary"][author]["median_numeric"]
            
            # Calculate percentage
            percentage = grade_to_percentage(median_num)
            
            # Convert numeric back to letter grade for display
            median_letter = percentage_to_letter_grade(percentage)
        else:
            median_letter = "N/A"
            percentage = "N/A"
            
        row = f"{author}," + ",".join(grades) + f",{median_letter},{percentage}"
        rows.append(row)
    
    # Combine everything
    return f"{header}\n" + "\n".join(rows)


def print_summary(results: Dict[str, Any]) -> None:
    """Print a summary table of results."""
    print("\n=== SUMMARY OF RESULTS ===")
    print(f"{'Model':<20} | {'Grade':<5} | {'Score':<5} | {'Grades Received'}")
    print("-" * 80)
    
    for author, stats in results["summary"].items():
        median_num = stats["median_numeric"]
        grades = ", ".join(stats["grades_received"])
        
        # Calculate percentage score
        percentage = grade_to_percentage(median_num)
        
        # Convert numeric back to letter grade for display
        median_letter = percentage_to_letter_grade(percentage)
                
        print(f"{author:<20} | {median_letter:<5} | {percentage:<5} | {grades}")
    
    # Print Boswell Quotient if available
    if "boswell_quotient" in results and "model_scores" in results["boswell_quotient"]:
        print("\n=== BOSWELL QUOTIENT ===")
        print(f"{'Rank':<5} | {'Model':<20} | {'Quotient':<8} | {'Grade':<5} | {'Performance':<11} | {'Evaluation':<11} | {'Efficiency':<11}")
        print("-" * 100)
        
        # Sort models by rank
        quotient_data = results["boswell_quotient"]["model_scores"]
        sorted_models = sorted(
            quotient_data.keys(),
            key=lambda m: quotient_data[m]["rank"]
        )
        
        for model in sorted_models:
            score_data = quotient_data[model]
            components = score_data["components"]
            
            rank = score_data["rank"]
            quotient = score_data["boswell_quotient"]
            
            # Convert to letter grade
            letter_grade = percentage_to_letter_grade(int(quotient))
            
            performance = components.get("performance", 0)
            performance_str = f"{performance:.1f}" if performance else "N/A"
            
            evaluation = components.get("evaluation", 0)
            evaluation_str = f"{evaluation:.1f}" if evaluation else "N/A"
            
            efficiency = components.get("efficiency", 0)
            efficiency_str = f"{efficiency:.1f}" if efficiency else "N/A"
            
            print(f"{rank:<5} | {model:<20} | {quotient:<8.1f} | {letter_grade:<5} | {performance_str:<11} | {evaluation_str:<11} | {efficiency_str:<11}")
    
    # Print grading bias if available
    if "bias_analysis" in results:
        print("\n=== GRADING BIAS ANALYSIS ===")
        
        # Get overall median
        overall_median = results["bias_analysis"]["overall_median"]
        overall_percentage = grade_to_percentage(overall_median)
        overall_median_letter = percentage_to_letter_grade(overall_percentage)
        
        print(f"Overall Median Grade: {overall_median_letter} ({overall_percentage})")
        print(f"{'Model':<20} | {'Grade':<5} | {'Score':<5} | {'Grading Bias':<25} | {'Bias Value'}")
        print("-" * 85)
        
        for grader, stats in results["bias_analysis"]["grader_bias"].items():
            # Convert numeric grade to percentage and letter
            median_grade = stats["median_given"]
            percentage = grade_to_percentage(median_grade)
            median_letter = percentage_to_letter_grade(percentage)
                    
            print(f"{grader:<20} | {median_letter:<5} | {percentage:<5} | {stats['letter_bias']:<25} | {stats['median_bias']:.2f}")
    
    # Print cost summary if available
    if "cost" in results:
        cost_data = results["cost"]
        total_cost = cost_data["total_cost"]
        total_duration = cost_data["total_duration"]
        total_tokens = cost_data["total_tokens"]
        
        print("\n=== COST SUMMARY ===")
        print(f"Total Cost: ${total_cost:.4f}")
        print(f"Total Tokens: {total_tokens:,}")
        print(f"Total Duration: {total_duration:.2f} seconds")
        
        # Calculate cost breakdown
        essay_cost = sum(info["total_cost"] for info in cost_data["essay_costs"].values())
        grading_cost = 0
        for grader_costs in cost_data["grading_costs"].values():
            grading_cost += sum(info["total_cost"] for info in grader_costs.values())
            
        print(f"Essay Generation: ${essay_cost:.4f} ({essay_cost/total_cost*100:.1f}%)")
        print(f"Grading: ${grading_cost:.4f} ({grading_cost/total_cost*100:.1f}%)")
        print(f"See cost_report.md for detailed breakdown")
    
    # Print timing information if available
    if "timing" in results and "step_durations" in results["timing"]:
        print("\n=== TIMING INFORMATION ===")
        
        # Format durations manually
        durations = results["timing"]["step_durations"]
        total = durations["total"]
        
        # Handle case where total might be zero
        if total <= 0:
            print(f"Total Run Time: 0.0s")
        else:
            minutes, seconds = divmod(total, 60)
            formatted_total = f"{int(minutes)}m {seconds:.1f}s"
            print(f"Total Run Time: {formatted_total}")
            
        print(f"Essay Generation: {durations['essay_generation']:.1f}s")
        print(f"Grading: {durations['grading']:.1f}s")
        print(f"Analysis: {durations['analysis']:.1f}s")
        print(f"File Generation: {durations['file_generation']:.1f}s")
    
    # Print essay file locations
    if "essay_files" in results:
        print("\n=== SAVED ESSAY FILES ===")
        for author, filename in results["essay_files"].items():
            print(f"{author:<20} | {filename}")
            
    # Print visualizations location
    if "results_dir" in results:
        print("\n=== VISUALIZATIONS ===")
        print(f"Data visualizations available in: {results['results_dir']}/charts/")
        print("- Grading bias chart")
        print("- Grade distribution boxplot")
        print("- Essay generation timing comparison")
        print("- Grading time comparison")
        print("- Cost breakdown")
        print("- Process timing breakdown")
        if "boswell_quotient" in results:
            print("- Boswell Quotient chart")
            print("- Boswell Quotient component breakdown")


def list_domains() -> None:
    """Display available domains."""
    print("\nAvailable domains:")
    print("-" * 60)
    for domain_id, description in AVAILABLE_DOMAINS.items():
        print(f"{domain_id:<15} | {description}")


def list_models() -> None:
    """Display available models."""
    print("\nAvailable models:")
    print("-" * 60)
    for model in MODELS:
        print(f"{model['name']:<25} | {model['model_id']}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Boswell Test - LLM Comparative Analysis")

    # --- ADD FREE FLAG HERE ---  ##########
    parser.add_argument( '-f', '--free', type = int, nargs = '?', const = MAX_FREE_MODELS, default = 0, help='Use only the top free models from OpenRouter. Optionally spedify a limit (def: 16).')  ##########
    # -----------------------
    
    # Main operation modes
    domain_group = parser.add_mutually_exclusive_group()
#    domain_group.add_argument("--domain", type=str, choices=AVAILABLE_DOMAINS.keys(),
    domain_group.add_argument("-d", "--domain", nargs='*', type=str, choices=AVAILABLE_DOMAINS.keys(),
#                        default="pol_sci_1",
#                        help="Domain to test (default: pol_sci_1)")
                        default="sample_boswell_query",  #####
                        help="Domain to test (default: sample_boswell_query)")  #####
    domain_group.add_argument("--all-domains", action="store_true",
                        help="Run tests on all available domains")
    
    parser.add_argument("--output", type=str, default="boswell_results.json",
                        help="Output file name (default: boswell_results.json)")
    
    parser.add_argument("--models", type=str, nargs="+",
                        help="Specific models to test (default: all models)")
    
    parser.add_argument("--skip-verification", action="store_true",
                        help="Skip the model verification step (faster but less reliable)")
    
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Maximum number of retries for API calls (default: 3)")
    
    # Information modes
    parser.add_argument("--list-domains", action="store_true",
                        help="List available domains and exit")
    
    parser.add_argument("--list-models", action="store_true",
                        help="List available models and exit")
    
    # Model management
    parser.add_argument("--update-models", action="store_true",
                        help="Update the local models file with currently available OpenRouter models and exit")
    
    parser.add_argument("--models-file", type=str, default="openrouter_models.json",
                        help="File to save/load OpenRouter models (default: openrouter_models.json)")
    print( 'parse_args : ' , parser.parse_args( ) )
    return parser.parse_args()


def run_all_domains(args) -> None:
    """Run Boswell Test on all available domains."""
    print(f"Starting Boswell Test for ALL available domains ({len(AVAILABLE_DOMAINS)} domains)...")
    
    # If models specified, use them, otherwise use all
    if args.models:
        print(f"Using {len(args.models)} specified models: {', '.join(args.models)}")
        models_to_use = args.models
    else:
        print(f"Using all available models")
        models_to_use = None
    
    all_results = {}
    
    # Run tests for each domain sequentially
    for domain_name, domain_description in AVAILABLE_DOMAINS.items():
        print(f"\n{'='*70}")
        print(f"=== Running test for domain: {domain_description} ===")
        print(f"{'='*70}\n")
        
        # Create a domain-specific output file
        domain_output = f"boswell_results_{domain_name}.json"
        
        try:
            # Run the test with all options
            results = run_boswell_test(
                domain_name=domain_name,
                output_file=domain_output,
                selected_models=models_to_use,
                skip_verification=args.skip_verification,
                max_retries=args.max_retries
            )
            
            # Store results
            all_results[domain_name] = {
                "directory": results.get("results_dir", ""),
                "domain": domain_description,
                "boswell_quotient": results.get("boswell_quotient", {}),
                "run_timestamp": results.get("run_timestamp", "")
            }
            
            # Print summary
            print_summary(results)
            
            print(f"\nTest for domain '{domain_name}' completed!")
            print(f"  - Full results directory: {results['results_dir']}")
            print(f"  - Tables directory: {results['results_dir']}/grades_table.*")
            print(f"  - Essays directory: {results['results_dir']}/essays/")
            print(f"  - Visualizations: {results['results_dir']}/charts/")
            
        except Exception as e:
            print(f"Error running test for domain '{domain_name}': {e}")
            print("Continuing with next domain...")
    
    # Print final summary
    print(f"\n{'='*70}")
    print(f"=== ALL DOMAIN TESTS COMPLETED ===")
    print(f"{'='*70}")
    print(f"Summary of all domain test results:")
    
    for domain_name, info in all_results.items():
        print(f"  - {info['domain']}: {info['directory']}")
    
    # Generate aggregate Boswell Quotient if we have multiple domains
    if len(all_results) > 1:
        print("\nGenerating aggregate Boswell Quotient report across all domains...")
        
        # Create timestamp for aggregate results
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Create aggregate results directory
        aggregate_dir = f"results/{timestamp}-aggregate"
        os.makedirs(aggregate_dir, exist_ok=True)
        
        # Generate aggregate data
        aggregated_data = aggregate_boswell_quotient(all_results)
        
        # Create mapping of domain IDs to descriptions
        domain_descriptions = {name: desc for name, desc in AVAILABLE_DOMAINS.items()}
        
        # Generate aggregate report
        aggregate_report = generate_aggregate_boswell_report(aggregated_data, domain_descriptions)
        
        # Save aggregate report
        with open(f"{aggregate_dir}/aggregate_boswell_report.md", 'w') as f:
            f.write(aggregate_report)
        
        # Generate aggregate visualizations
        charts_dir = f"{aggregate_dir}/charts"
        os.makedirs(charts_dir, exist_ok=True)
        
        # Create aggregate Boswell Quotient chart
        plt.figure(figsize=(14, 8))
        
        # Prepare data
        sorted_models = sorted(
            aggregated_data["model_scores"].keys(),
            key=lambda m: aggregated_data["model_scores"][m]["average_boswell_quotient"],
            reverse=True
        )
        
        scores = [aggregated_data["model_scores"][m]["average_boswell_quotient"] for m in sorted_models]
        
        # Create bar chart
        plt.barh(sorted_models, scores, color='#6A5ACD')
        plt.title('Aggregate Boswell Quotient by Model (All Domains)', fontsize=16)
        plt.xlabel('Boswell Quotient (0-100)', fontsize=12)
        plt.yticks(fontsize=10)
        plt.xlim(0, 100)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save chart
        plt.savefig(f"{charts_dir}/aggregate_boswell_quotient.png", dpi=300)
        plt.close()
        
        # Create domain comparison chart for top models
        # Only include models that appear in all domains
        all_domain_models = [m for m in sorted_models 
                           if len(aggregated_data["model_scores"][m]["domain_scores"]) == len(all_results)]
        
        # Take top 5 models or fewer if less available
        top_models = all_domain_models[:min(5, len(all_domain_models))]
        
        if top_models:
            plt.figure(figsize=(14, 8))
            
            # Set up domain names and positions
            domains = list(aggregated_data["domains_analyzed"])
            domain_names = [domain_descriptions.get(d, d) for d in domains]
            
            # Set up bar positions
            bar_width = 0.15
            positions = np.arange(len(domains))
            
            # Plot bars for each model
            for i, model in enumerate(top_models):
                domain_scores = [aggregated_data["model_scores"][model]["domain_scores"].get(d, 0) for d in domains]
                plt.bar(positions + i*bar_width, domain_scores, bar_width, label=model)
            
            # Configure chart
            plt.title('Boswell Quotient Comparison Across Domains (Top Models)', fontsize=16)
            plt.xlabel('Domain', fontsize=12)
            plt.ylabel('Boswell Quotient', fontsize=12)
            plt.xticks(positions + bar_width * (len(top_models)-1)/2, domain_names, rotation=45, ha='right')
            plt.ylim(0, 100)
            plt.legend(title='Model')
            plt.tight_layout()
            
            # Save chart
            plt.savefig(f"{charts_dir}/domain_comparison.png", dpi=300)
            plt.close()
        
        # Save aggregate data as JSON
        with open(f"{aggregate_dir}/aggregate_boswell_quotient.json", 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "aggregated_data": aggregated_data,
                "source_domains": [d for d in aggregated_data["domains_analyzed"]],
                "domain_descriptions": domain_descriptions
            }, f, indent=2)
        
        print(f"Aggregate results saved to: {aggregate_dir}")
        print(f"  - Aggregate report: {aggregate_dir}/aggregate_boswell_report.md")
        print(f"  - Visualizations: {aggregate_dir}/charts/")
    
    print("\nBoswell Test for all domains has been completed!")

def aggregate_boswell_quotient(domain_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate Boswell Quotient scores across multiple domains.
    
    Args:
        domain_results: Dictionary of domain names to result dictionaries
        
    Returns:
        Dictionary containing aggregated Boswell Quotient data
    """
    # Initialize data structure for aggregated results
    aggregated = {
        "model_scores": {},
        "domains_analyzed": list(domain_results.keys()),
        "component_weights": {
            "performance": 0.50,
            "evaluation": 0.30,
            "efficiency": 0.20
        }
    }
    
    # Track which models appear in which domains
    model_domains = {}
    
    # Collect all Boswell Quotient data by model across domains
    for domain_name, results in domain_results.items():
        if "boswell_quotient" in results and "model_scores" in results["boswell_quotient"]:
            model_scores = results["boswell_quotient"]["model_scores"]
            
            for model, score_data in model_scores.items():
                # Initialize model entry if not exists
                if model not in aggregated["model_scores"]:
                    aggregated["model_scores"][model] = {
                        "domain_scores": {},
                        "domain_components": {},
                        "average_boswell_quotient": 0.0,
                        "rank": 0
                    }
                
                # Add domain-specific data
                aggregated["model_scores"][model]["domain_scores"][domain_name] = score_data["boswell_quotient"]
                aggregated["model_scores"][model]["domain_components"][domain_name] = score_data.get("components", {})
                
                # Track which domains this model appears in
                if model not in model_domains:
                    model_domains[model] = []
                model_domains[model].append(domain_name)
    
    # Calculate aggregate scores
    for model, data in aggregated["model_scores"].items():
        # Calculate average Boswell Quotient across all domains where the model appears
        if data["domain_scores"]:
            data["average_boswell_quotient"] = sum(data["domain_scores"].values()) / len(data["domain_scores"])
            
            # Calculate aggregated component scores
            all_components = {
                "performance": [],
                "evaluation": [],
                "efficiency": []
            }
            
            # Collect component scores from all domains
            for domain, components in data["domain_components"].items():
                for component in all_components:
                    if component in components:
                        all_components[component].append(components[component])
            
            # Calculate average for each component
            data["aggregated_components"] = {}
            for component, scores in all_components.items():
                if scores:
                    data["aggregated_components"][component] = sum(scores) / len(scores)
    
    # Rank models by average Boswell Quotient
    sorted_models = sorted(
        aggregated["model_scores"].keys(),
        key=lambda m: aggregated["model_scores"][m]["average_boswell_quotient"],
        reverse=True
    )
    
    # Assign ranks
    for i, model in enumerate(sorted_models):
        aggregated["model_scores"][model]["rank"] = i + 1
    
    # Add consistency metrics - how consistent are models across domains?
    for model, data in aggregated["model_scores"].items():
        if len(data["domain_scores"]) > 1:
            scores = list(data["domain_scores"].values())
            max_score = max(scores)
            min_score = min(scores)
            
            # Calculate consistency as 100 - (max-min)
            # Higher values mean more consistent performance across domains
            data["consistency"] = 100 - (max_score - min_score)
            
            # Add domain count
            data["domain_count"] = len(data["domain_scores"])
            
            # Identify best and worst domains
            best_domain = max(data["domain_scores"].items(), key=lambda x: x[1])[0]
            worst_domain = min(data["domain_scores"].items(), key=lambda x: x[1])[0]
            
            data["best_domain"] = best_domain
            data["worst_domain"] = worst_domain
    
    return aggregated


def generate_aggregate_boswell_report(aggregated_data: Dict[str, Any], domain_descriptions: Dict[str, str]) -> str:
    """Generate a comprehensive report of aggregate Boswell Quotient results.
    
    Args:
        aggregated_data: Aggregated Boswell Quotient data
        domain_descriptions: Dictionary mapping domain IDs to descriptions
        
    Returns:
        Markdown report as a string
    """
    lines = []
    
    # Title and introduction
    lines.append("# Aggregate Boswell Quotient Analysis Across Domains")
    lines.append(f"\nThis report aggregates model performance across {len(aggregated_data['domains_analyzed'])} domains:")
    
    # List the domains included
    for domain_id in aggregated_data['domains_analyzed']:
        domain_desc = domain_descriptions.get(domain_id, domain_id)
        lines.append(f"- **{domain_desc}**")
    
    # Overall model rankings table
    lines.append("\n## Overall Model Rankings")
    lines.append("\nThe table below shows models ranked by their average Boswell Quotient across all domains:")
    lines.append("\n| Rank | Model | Boswell Quotient | Grade | Domain Count | Consistency | Best Domain | Worst Domain |")
    lines.append("|------|-------|-----------------|-------|--------------|-------------|------------|-------------|")
    
    # Sort models by rank
    sorted_models = sorted(
        aggregated_data["model_scores"].keys(),
        key=lambda m: aggregated_data["model_scores"][m]["rank"]
    )
    
    # Add each model's data
    for model in sorted_models:
        data = aggregated_data["model_scores"][model]
        
        rank = data["rank"]
        avg_score = data["average_boswell_quotient"]
        
        # Convert to letter grade
        letter_grade = percentage_to_letter_grade(int(avg_score))
        
        domain_count = data.get("domain_count", len(data["domain_scores"]))
        
        # Format consistency if available
        consistency = data.get("consistency", "N/A")
        if consistency != "N/A":
            consistency = f"{consistency:.1f}"
            
        # Get best and worst domains with descriptions
        best_domain = data.get("best_domain", list(data["domain_scores"].keys())[0] if data["domain_scores"] else "N/A")
        worst_domain = data.get("worst_domain", best_domain)
        
        best_domain_desc = domain_descriptions.get(best_domain, best_domain)
        worst_domain_desc = domain_descriptions.get(worst_domain, worst_domain)
        
        # Format row
        lines.append(f"| {rank} | {model} | {avg_score:.1f} | {letter_grade} | {domain_count} | {consistency} | {best_domain_desc} | {worst_domain_desc} |")
    
    # Top performers section
    if sorted_models:
        lines.append("\n## Top Performing Models")
        top_model = sorted_models[0]
        top_data = aggregated_data["model_scores"][top_model]
        
        lines.append(f"\n**{top_model}** achieved the highest average Boswell Quotient of **{top_data['average_boswell_quotient']:.1f}** " +
                    f"across {len(top_data['domain_scores'])} domains.")
        
        # Add information about consistency for top model
        if "consistency" in top_data:
            if top_data["consistency"] > 90:
                lines.append(f"With a consistency score of {top_data['consistency']:.1f}, it demonstrated remarkably stable performance across all domains.")
            elif top_data["consistency"] > 75:
                lines.append(f"With a consistency score of {top_data['consistency']:.1f}, it performed well across most domains with some variation.")
            else:
                lines.append(f"With a consistency score of {top_data['consistency']:.1f}, it showed significant performance variation across domains.")
                
                # Mention best and worst domains
                best = domain_descriptions.get(top_data["best_domain"], top_data["best_domain"])
                worst = domain_descriptions.get(top_data["worst_domain"], top_data["worst_domain"])
                lines.append(f"It performed best in **{best}** and had more difficulty with **{worst}**.")
        
        # Mention other top performers
        if len(sorted_models) > 1:
            runner_ups = []
            for model in sorted_models[1:4]:  # Get up to next 3 models
                if model in aggregated_data["model_scores"]:
                    score = aggregated_data["model_scores"][model]["average_boswell_quotient"]
                    runner_ups.append(f"**{model}** ({score:.1f})")
            
            if runner_ups:
                lines.append(f"\nOther strong performers include {', '.join(runner_ups)}.")
    
    # Domain-specific leaders
    lines.append("\n## Domain-Specific Leaders")
    lines.append("\nThe table below shows which models performed best in each domain:")
    lines.append("\n| Domain | Top Model | Boswell Quotient | Grade |")
    lines.append("|--------|-----------|------------------|-------|")
    
    # Find top model for each domain
    for domain_id in aggregated_data["domains_analyzed"]:
        domain_desc = domain_descriptions.get(domain_id, domain_id)
        
        # Find the best model for this domain
        best_model = ""
        best_score = 0
        
        for model, data in aggregated_data["model_scores"].items():
            if domain_id in data["domain_scores"] and data["domain_scores"][domain_id] > best_score:
                best_model = model
                best_score = data["domain_scores"][domain_id]
        
        if best_model:
            # Convert to letter grade
            letter_grade = percentage_to_letter_grade(int(best_score))
            lines.append(f"| {domain_desc} | {best_model} | {best_score:.1f} | {letter_grade} |")
    
    # Overall insights
    lines.append("\n## Key Insights")
    
    # Find models that perform consistently across domains
    consistent_models = []
    for model, data in aggregated_data["model_scores"].items():
        if "consistency" in data and data["consistency"] > 85 and len(data["domain_scores"]) > 1:
            consistent_models.append((model, data["consistency"], data["average_boswell_quotient"]))
    
    if consistent_models:
        # Sort by consistency
        consistent_models.sort(key=lambda x: x[1], reverse=True)
        
        # Take up to top 3
        top_consistent = consistent_models[:min(3, len(consistent_models))]
        model_list = [f"**{m}** (consistency: {c:.1f})" for m, c, _ in top_consistent]
        
        lines.append("\n### Most Consistent Performers")
        lines.append(f"These models maintained the most consistent performance across different domains: {', '.join(model_list)}.")
        lines.append("High consistency suggests these models have broad capabilities that transfer well between different subject areas.")
    
    # Find specialists (high performance in specific domains)
    specialists = []
    for model, data in aggregated_data["model_scores"].items():
        if len(data["domain_scores"]) > 1:
            scores = list(data["domain_scores"].values())
            max_score = max(scores)
            avg_other = sum(s for s in scores if s != max_score) / (len(scores) - 1)
            
            # If top domain is significantly better than average of others
            if max_score > avg_other + 10:
                best_domain = max(data["domain_scores"].items(), key=lambda x: x[1])[0]
                specialists.append((model, best_domain, max_score, max_score - avg_other))
    
    if specialists:
        # Sort by specialization gap
        specialists.sort(key=lambda x: x[3], reverse=True)
        
        lines.append("\n### Domain Specialists")
        lines.append("These models showed significantly stronger performance in specific domains:")
        
        for model, domain, score, gap in specialists[:min(3, len(specialists))]:
            domain_desc = domain_descriptions.get(domain, domain)
            lines.append(f"- **{model}** excels in **{domain_desc}** (score: {score:.1f}, {gap:.1f} points above its average in other domains)")
    
    # Conclusion
    lines.append("\n## Conclusion")
    lines.append(
        "The aggregated Boswell Quotient provides a comprehensive view of model capabilities across multiple domains, "
        "helping identify both generalist models that perform consistently well across diverse tasks and specialist models "
        "that excel in particular areas."
    )
    
    if sorted_models:
        lines.append(
            f"\nBased on this cross-domain analysis, **{sorted_models[0]}** emerges as the most capable overall AI assistant, "
            f"with strong performance across the tested domains."
        )
    
    return "\n".join(lines)


def get_provider_key(model_data):
    """Safely extracts the clean provider name from the model's 'name' field."""
    return model_data['name'].split(':')[0].strip()


def main() -> None:
    """Main entry point for the script."""
    global MODELS  #  declare as global to modify its earlier scope declaration

    args = parse_arguments()

    # Determine which models to fetch
    free_limit = args.free  ##########

    if ( free_limit < 0 ) :
        raise argparse.ArgumentTypeError(f"limit must be zero or a positive number (received {free_limit})")

    is_free_mode = free_limit > 0
    print( "Fetching f'FREE (Limit: {free_limit}' if is_free_mode else 'TOP'} models from OpenRouter API...")
    models_raw_list = getTopModels( is_free_mode , free_limit )  ##########

    MODELS_BY_PROVIDER = {}
    # Apply Provider Filtering Logic (your original logic, simplified)
    for model_data in models_raw_list:
        provider_key = get_provider_key(model_data)

        if ( provider_key not in MODELS_BY_PROVIDER ) :
            if not args.free and provider_key not in ALLOWED_PROVIDERS:
                continue  #  Skip if not free and not in the allowed list
            MODELS_BY_PROVIDER[provider_key] = model_data
            MODELS.append(model_data)

    if ( is_free_mode ) :  #  args.free  ##########
        nmodels = len(MODELS)
        if ( MAX_FREE_MODELS < nmodels ) :
            print(f"Warning: Found {nmodels} free models. Winnowing to {MAX_FREE_MODELS}...")
            for i in range(nmodels - 1, -1, -1):
                provider_key = get_provider_key(MODELS[i])
            
                # This is the criteria for deletion: not in ALLOWED_PROVIDERS
                if provider_key not in ALLOWED_PROVIDERS:
#                    print( 'Excluding ' , provider_key , ' to max 12 free' )
                    del MODELS[i]
                    nmodels -= 1
                
                    if nmodels == MAX_FREE_MODELS:
                        break

    if not MODELS:
        print("Error: Could not find any valid models after filtering. Check API or PROVIDERS.")
        sys.exit(1)
        
    print(f"Successfully loaded {len(MODELS)} models for testing.")
    print( 'FREE:' if args.free else 'TOP:' , MODELS )
    # ====================================================================

    # Handle information modes
    if args.list_domains:
        list_domains()
        return
    
    if args.list_models:
        list_models()
        return

    # Handle model management
    if args.update_models:
        update_models_file(args.models_file)
        return
    
    # Run tests on all domains
    if args.all_domains:
        run_all_domains(args)
        return
    
    # Run the test on a single domain
    print(f"Starting Boswell Test for domain '{args.domain}'...")
    
    # If models specified, use them, otherwise use all
    if args.models:
        print(f"Using {len(args.models)} specified models: {', '.join(args.models)}")
        models_to_use = args.models
    else:
        print(f"Using all available models")
        models_to_use = None
    
    # Run the test with all options
    results = run_boswell_test(
        domain_name=args.domain,
        output_file=args.output,
        selected_models=models_to_use,
        skip_verification=args.skip_verification,
        max_retries=args.max_retries
    )
    
    # Print summary
    print_summary(results)
    
    print(f"\nBoswell Test completed!")
    print(f"  - Full results directory: {results['results_dir']}")
    print(f"  - Tables directory: {results['results_dir']}/grades_table.*")
    print(f"  - Essays directory: {results['results_dir']}/essays/")
    print(f"  - Visualizations: {results['results_dir']}/charts/")


if __name__ == "__main__":
    main()
