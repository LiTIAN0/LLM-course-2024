from llama_index.llms.ollama import Ollama
import dspy
import pandas as pd
import random
import re

llm = Ollama(model="tinyllama", request_timeout=600.0)

# you can use DSPY (https://github.com/stanfordnlp/dspy), but you can also choose another method of interacting with an LLM
dspy.settings.configure(lm=llm)

# Task: implement a method, that will take a query string as input and produce N misspelling variants of the query.
# These variants with typos will be used to test a search engine quality.
# Example
# Query: machine learning applications
# Possible Misspellings:
# "machin learning applications" (missing "e" in "machine")
# "mashine learning applications" (phonetically similar spelling of "machine")
# "machine lerning aplications" (missing "a" in "learning" and "p" in "applications")
# "machin lerning aplications" (combining multiple typos)
# "mahcine learing aplication" (transposed letters in "machine" and typos in "learning" and "applications")
#
# Questions:
# 1. Does the search engine produce the same results for all the variants?
# 2. Do all variants make sense?
# 3. How to improve robustness of the method, for example, skip known abbreviations, like JFK or NBC.
# 4. Can you test multiple LLMs and figure out which one is the best?
# 5. Do the misspellings capture a variety of error types (phonetic, omission, transposition, repetition)?


class QueryMisspeller:
    def __init__(self, llm):
        self.llm = llm
        self.abbreviations = {'JFK', 'NBC', 'US', 'UK'}

    def generate_misspellings_with_llm(self, query: str, n: int = 3):
    	"""Generate misspellings using the LLM"""
    	prompt = f"""
    	Generate {n} different misspelled versions of this search query: "{query}"

    	Rules for generating misspellings:
    	1. Keep abbreviations like JFK, NBC, US unchanged
    	2. Include different types of errors:
       	- Omission (Missing letters)
       	- Transposition (Swapped letters)
       	- Phonetic mistakes (f/ph, k/c, etc.)
       	- Repetition
       	- Multiple typos in one variant
    	3. Each variant should still be readable and recognizable
    	4. Output format must be EXACTLY like this example:
    
    	Original Query: machine learning applications
    	- machin learning applications (missing "e" in "machine")
    	- mashine learning applications (phonetically similar spelling of "machine")
    	- machine lerning aplications (missing "a" in "learning" and "p" in "applications")
    	- machin lerning aplications (combining multiple typos)
    	- mahcine learing aplication (transposed letters in "machine" and typos)

    	Generate misspellings for: "{query}"
    	"""
    
    	try:
        	response = self.llm.complete(prompt).text
        	# Only get lines starting with "-" and extract the misspelling part
        	variants = [
            	line.split('(')[0].strip() 
            	for line in response.split('\n') 
            	if line.strip().startswith('-')
        	]
        	# Filter out empty strings and the original query
        	variants = [v for v in variants if v and v != query][:n]
        	return variants
    	except Exception as e:
        	print(f"Error generating misspellings: {e}")
        	return []



def main():

	# Initialize misspeller with LLM
	misspeller = QueryMisspeller(llm)

	csv_file = 'web_search_queries.csv'

	df = pd.read_csv(csv_file, header=0) # Explicitly specify header row

	query_index = random.randint(0, 44)


	query = df.loc[query_index, 'Query'] 


	print(f"\nOriginal: {query}")

	print("Misspellings:")

    

	# Get misspellings using LLM

	misspellings = misspeller.generate_misspellings_with_llm(query)

	for i, variant in enumerate(misspellings, 1):

		print(f"{i}. {variant}")

if __name__ == "__main__":
    main()
