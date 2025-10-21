import os
import json
from typing import List, Optional
from dotenv import load_dotenv

# Pydantic is used to define the desired structured output
from pydantic import BaseModel, Field

# LangChain components for RAG
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# --- 1. DEFINE ALL PYDANTIC MODELS (COMBINED) ---

# --- Models for Bill Summary (from script 1) ---

class Provider(BaseModel):
    """Information about the utility provider."""
    name: str = Field(description="The name of the utility provider, e.g., 'Xcel Energy'.")
    country: str = Field(description="The country where the provider operates, e.g., 'USA'.")

class BillingAddress(BaseModel):
    """Represents a billing address."""
    addressType: str = Field(description="Type of address, e.g., 'FULL' or 'PARTIAL'.", default="FULL")
    streetLine1: str = Field(description="The primary street address line.")
    streetLine2: Optional[str] = Field(description="The secondary street address line (if any).", default=None)
    city: Optional[str] = Field(description="The city of the address.", default=None)
    state: Optional[str] = Field(description="The state or province of the address.", default=None)
    postalCode: Optional[str] = Field(description="The postal or ZIP code.", default=None)
    country: str = Field(description="The country of the address, e.g., 'USA'.")
    recipient: Optional[str] = Field(description="The name of the person or company receiving the bill.", default=None)

class AccountDataItem(BaseModel):
    """A single account data entry."""
    accountNumber: str = Field(description="The unique account number for the customer.")
    billingAddress: BillingAddress

class ChargeItem(BaseModel):
    """A single line item from the *summary* charges table."""
    chargeNameAsPrinted: str = Field(description="The description of the charge as printed on the bill (e.g., 'PREMISES DESCRIPTOR').")
    chargeAmount: float = Field(description="The monetary value of this specific charge.")
    chargeCurrencyCode: str = Field(description="Currency code for this charge, e.g., 'USD'.", default="USD")
    premisesNumber: Optional[str] = Field(description="The premises number associated with this charge, if available.", default=None)

class DisconnectNotice(BaseModel):
    """Information related to a disconnection notice, if present."""
    pastDueAmount: Optional[float] = Field(description="The overdue amount specified in the notice.", default=None)
    disconnectDate: Optional[str] = Field(description="The date of potential disconnection in YYYY-MM-DD format.", default=None)
    reconnectionFee: Optional[float] = Field(description="The fee to reconnect service.", default=None)
    currencyCode: Optional[str] = Field(description="Currency code for the fees, e.g., 'USD'.", default=None)

# --- Models for Premise Details (from script 2) ---

class PremiseLineItem(BaseModel):
    """A single detailed line item charge (e.g., 'Energy Charge Summer' or 'City Fees')."""
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

# --- Helper Model for finding premise numbers ---
class PremisesList(BaseModel):
    """A simple model to hold the list of all premise numbers found."""
    premise_numbers: List[str] = Field(description="A list of all unique premise numbers found in the document.")

# --- The FINAL, COMBINED UtilityBill Model ---

class UtilityBill(BaseModel):
    """The complete, structured data extracted from a utility bill."""
    type: str = Field(description="The type of document.", default="BILL")
    provider: Provider
    currencyCode: str = Field(description="The main currency code for the bill, e.g., 'USD'.", default="USD")
    statementDate: str = Field(description="The main date of the bill statement in YYYY-MM-DD format.")
    previousStatementDate: Optional[str] = Field(description="The date of the previous statement in YYYY-MM-DD format.", default=None)
    dueDate: str = Field(description="The date by which the payment is due in YYYY-MM-DD format.")
    periodStartDate: Optional[str] = Field(description="The start date of the billing period in YYYY-MM-DD format.", default=None)
    periodEndDate: Optional[str] = Field(description="The end date of the billing period in YYYY-MM-DD format.", default=None)
    
    totalCharges: float = Field(description="The total of new charges for the current period (e.g., 'Current Charges').")
    amountDue: float = Field(description="The total amount due for this billing period (e.g., 'Amount Due').")
    previousBalance: Optional[float] = Field(description="The balance carried over from the previous statement.", default=None)
    lastPaymentAmount: Optional[float] = Field(description="The amount of the last payment received.", default=None)
    lastPaymentDate: Optional[str] = Field(description="The date the last payment was received in YYYY-MM-DD format.", default=None)

    accountData: List[AccountDataItem] = Field(description="A list of account data objects, typically containing one entry.")
    
    charges: List[ChargeItem] = Field(description="A detailed list of all line item charges from the 'Premises Summary' table.")
    
    disconnectNotice: DisconnectNotice = Field(description="Disconnect notice details, if any are present on the bill.")
    
    # --- NEWLY ADDED FIELD ---
    # This will be populated manually in the main() function after the initial extraction.
    premiseDetails: Optional[List[PremiseDetails]] = Field(
        description="A list of detailed breakdowns for each individual premise.", 
        default=None
    )


# --- 2. DEFINE HELPER FUNCTIONS (from both files) ---

load_dotenv()
CHROMA_PATH = "chroma"

# --- Function 1: Extract Bill Summary (from script 1) ---
def extract_bill_summary(query: str, vector_store: Chroma) -> Optional[UtilityBill]:
    """Uses RAG to find summary info and populate the main UtilityBill model."""
    print(f"🔄 Running extraction for: {UtilityBill.__name__} (Summary)")
    
    # Retrieve more chunks to ensure the entire bill (especially all line items) is included
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 15})
    
    # For a single-page bill, retrieving all chunks is the most robust method.
    all_chunks = vector_store.get(include=["documents"])
    context_text = "\n\n---\n\n".join([doc for doc in all_chunks['documents']])

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    # We use the main UtilityBill model here. `premiseDetails` will be `None`.
    structured_llm = llm.with_structured_output(UtilityBill)
    
    prompt_template = """
    Based ONLY on the following context from a single utility bill, extract ALL requested information.
    Fill out the entire JSON schema provided.
    
    - For `totalCharges`, use the 'Current Charges' value.
    - For `amountDue`, use the 'Amount Due' value.
    - `charges` should be a list of ALL line items from the 'PREMISES SUMMARY' table.
    - Infer dates like `previousStatementDate` from context (e.g., 'As of 08/26' for previous balance).
    - If a payment was received, extract the amount. If it says 'No Payments Received', `lastPaymentAmount` should be 0.0.
    - If no information is present for a field, leave it as null (it will be handled by the schema).
    - The bill is from the USA, so the currency is 'USD' and country is 'USA'.
    - DO NOT attempt to fill in 'premiseDetails'. Leave it as null.

    Context:
    {context}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | structured_llm
    
    try:
        response = chain.invoke({"context": context_text})
        print(f"✅ Success for: {UtilityBill.__name__} (Summary)")
        return response
    except Exception as e:
        print(f"❌ Error during extraction for {UtilityBill.__name__}: {e}")
        return None

# --- Function 2: Get Premise Numbers (from script 2) ---
def get_all_premises_numbers(vector_store: Chroma) -> List[str]:
    """
    Query the document to find all unique premise numbers
    from the summary page.
    """
    print("🔄 Finding all unique premise numbers...")
    
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

# --- Function 3: Get Premise Details (from script 2) ---
def extract_premise_details(premise_num: str, vector_store: Chroma) -> Optional[PremiseDetails]:
    """
    For a single premise number, find its detailed breakdown pages
    and extract all information into the PremiseDetails model.
    """
    print(f"🔄 Extracting details for premise: {premise_num}...")
    
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


# --- 3. ORCHESTRATE THE COMBINED EXTRACTION ---

def main():
    """Main function to run the full, combined extraction pipeline."""
    
    # --- Setup ---
    print("Loading vector database...")
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    print("Vector database loaded successfully.")

    # --- Step 1: Extract the Bill Summary ---
    # This query is for the summary data
    summary_query = """
    Extract all summary information from the utility bill. This includes:
    1.  Provider name and country.
    2.  All summary details: statement date, due date, total current charges, previous balance, and total amount due.
    3.  Account number and the full billing address (recipient, street, city, state, zip).
    4.  A complete list of ALL line items from the 'PREMISES SUMMARY' table.
    5.  Any last payment information.
    6.  Any disconnect notice information.
    """
    print("\n--- 1. Extracting Bill Summary ---")
    final_bill = extract_bill_summary(summary_query, db)
    
    if not final_bill:
        print("❌ Critical error: Failed to extract bill summary. Exiting.")
        return

    # --- Step 2: Get the list of all premises to process ---
    print("\n--- 2. Finding Premise Numbers ---")
    premise_numbers = get_all_premises_numbers(db)
    
    all_details = []
    if not premise_numbers:
        print("⚠️ Warning: No premise numbers found. The final JSON will only contain summary data.")
    else:
        # --- Step 3: Loop and extract details for each premise ---
        print(f"\n--- 3. Extracting Details for {len(premise_numbers)} Premises ---")
        for num in premise_numbers:
            details = extract_premise_details(num, db)
            if details:
                all_details.append(details)
        print(f"--- ✅ Successfully extracted details for {len(all_details)} premises ---")

    # --- Step 4: Combine Summary and Details ---
    print("\n--- 4. Combining Summary and Details ---")
    final_bill.premiseDetails = all_details # Assign the list of details to the main object

    # --- Step 5: Save the Single, Combined JSON ---
    output_filename = "combined_bill.json"
    try:
        json_output = final_bill.model_dump_json(indent=2)
        
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(json_output)
            
        print(f"\n\n--- ✅ SUCCESS ---")
        print(f"All data has been extracted and saved to a single file: {output_filename}")
        
    except Exception as e:
        print(f"\n\n--- ❌ FAILED TO SAVE ---")
        print(f"An error occurred while saving the final JSON: {e}")

if __name__ == "__main__":
    main()