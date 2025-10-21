import os
import json
from typing import List, Optional
from dotenv import load_dotenv

# Pydantic models for structured output
from pydantic import BaseModel, Field

# LangChain components
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# --- 1. DEFINE PYDANTIC MODELS FOR PREMISE DETAILS ---

class PremiseLineItem(BaseModel):
    """A single line item charge (e.g., 'Energy Charge Summer' or 'City Fees')."""
    description: str = Field(description="The description of the charge as printed, e.g., 'Basic Service Chg' or 'State Tax'.")
    usageUnits: Optional[str] = Field(description="The usage and units, e.g., '117 kWh' or '0 therms'.", default=None)
    rate: Optional[str] = Field(description="The rate applied, e.g., '$0.130690' or '6.000%'.", default=None)
    amount: float = Field(description="The final dollar amount for this line item.")

class ServiceBreakdown(BaseModel):
    """Details for a single service (Electricity or Gas) at a premise."""
    InvoiceNumber: str = Field(description="The invoice number for this service.")
    serviceType: str = Field(description="The type of service, e.g., 'ELECTRICITY' or 'NATURAL GAS'.")
    serviceAddress: str = Field(description="The full service address for this premise.")
    meterNumber: Optional[str] = Field(description="The meter number for this service.", default=None)
    readPeriod: Optional[str] = Field(description="The billing period for these charges, e.g., '08/25/25 - 09/25/25'.", default=None)
    lineItems: List[PremiseLineItem] = Field(description="A list of all detailed charges, including taxes.")
    total: float = Field(description="The total amount for this service (e.g., 'Total' for Electricity).")

class PremiseDetails(BaseModel):
    """All extracted details for a single, unique premise."""
    premisesNumber: str = Field(description="The unique identifier for the premise, e.g., '304679358'.")
    services: List[ServiceBreakdown] = Field(description="A list of service breakdowns (e.g., one for Electricity, one for Gas).")
    premisesTotal: float = Field(description="The final 'Premises Total' amount, which sums all services for this premise.")

class PremisesList(BaseModel):
    """A simple model to hold the list of all premise numbers found."""
    premise_numbers: List[str] = Field(description="A list of all unique premise numbers found in the document.")

class AllPremises(BaseModel):
    """A top-level wrapper to hold the list of all extracted premise details."""
    premises: List[PremiseDetails]

# --- 2. SETUP RAG FUNCTIONS ---

load_dotenv()
CHROMA_PATH = "chroma"

def get_all_premises_numbers(vector_store: Chroma) -> List[str]:
    """
    First, query the document to find all unique premise numbers.
    This uses the summary page (image 1) to build a list.
    """
    print("🔄 Finding all unique premise numbers...")
    
    # Get all text from the document
    all_chunks = vector_store.get(include=["documents"])
    context_text = "\n\n---\n\n".join([doc for doc in all_chunks['documents']])

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_llm = llm.with_structured_output(PremisesList)
    
    prompt_template = """
    Based ONLY on the provided context, find the 'PREMISES SUMMARY' table.
    Extract *all* of the 'PREMISES NUMBER' values from that table.
    Return only the list of premise number strings.

    Context:
    {context}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | structured_llm
    
    try:
        response = chain.invoke({"context": context_text})
        print(f"✅ Found {len(response.premise_numbers)} premises.")
        return response.premise_numbers
    except Exception as e:
        print(f"❌ Error finding premise numbers: {e}")
        return []

def extract_premise_details(premise_num: str, vector_store: Chroma) -> Optional[PremiseDetails]:
    """
    For a single premise number, find its detailed breakdown pages
    and extract all information into the PremiseDetails model.
    """
    print(f"🔄 Extracting details for premise: {premise_num}...")
    
    # Use a retriever focused on this specific premise number
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 30})
    query = f"All details for premise {premise_num}, including electricity and natural gas charges, meter readings, and service address."
    relevant_chunks = retriever.invoke(query)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in relevant_chunks])

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_llm = llm.with_structured_output(PremiseDetails)
    
    prompt_template = """
    You will be given context from a utility bill that contains the detailed breakdown
    for a specific premise. Based ONLY on this context, extract the following
    information for PREMISE NUMBER: {premise_num}.
    
    1.  Find the 'ELECTRICITY SERVICE DETAILS' and 'NATURAL GAS SERVICE DETAILS'.
    2.  For each service, extract the invoice number, service address, meter number, and read period.
    3.  For each service, extract *all* line items from 'ELECTRICITY CHARGES'/'NATURAL GAS CHARGES'
        and all taxes (e.g., 'City Fees', 'State Tax').
    4.  Extract the 'Total' for each service.
    5.  Find the final 'Premises Total' for this premise (it's often at the end
        of the natural gas section).
    
    Format the output perfectly according to the JSON schema.

    Context:
    {context}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | structured_llm
    
    try:
        response = chain.invoke({"context": context_text, "premise_num": premise_num})
        print(f"✅ Success for premise: {premise_num}")
        return response
    except Exception as e:
        print(f"❌ Error during extraction for {premise_num}: {e}")
        return None

# --- 3. ORCHESTRATE THE EXTRACTION PROCESS ---

def main():
    """Main function to run the full extraction pipeline."""
    print("Loading vector database...")
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    print("Vector database loaded successfully.")

    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Get the list of all premises to process
    premise_numbers = get_all_premises_numbers(db)
    
    if not premise_numbers:
        print("No premise numbers found. Exiting.")
        return

    # Step 2: Loop through each premise and extract its details
    successful_extractions = 0
    for num in premise_numbers:  # Uncomment this loop
        details = extract_premise_details(num, db)
        if details:
            # Save each premise to its own JSON file
            output_filename = os.path.join(output_dir, f"{details.premisesNumber}.json")
            json_output = details.model_dump_json(indent=2)
            
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(json_output)
            
            print(f"✅ Saved premise {details.premisesNumber} to {output_filename}")
            successful_extractions += 1

    # Summary
    if successful_extractions > 0:
        print(f"\n\n--- ✅ Successfully extracted and saved {successful_extractions} premises to '{output_dir}/' folder ---")
    else:
        print("\n\n--- ❌ Failed to extract any premise details ---")

if __name__ == "__main__":
    main()