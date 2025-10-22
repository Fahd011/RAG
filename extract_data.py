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

# --- 1. DEFINE NEW PYDANTIC MODELS TO MATCH YOUR TARGET JSON ---

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
    """A single line item from the charges or premises summary."""
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
    
    charges: List[ChargeItem] = Field(description="A detailed list of all line item charges, often from a 'Premises Summary' or similar table.")
    
    disconnectNotice: DisconnectNotice = Field(description="Disconnect notice details, if any are present on the bill.")


# --- 2. SETUP THE RAG EXTRACTION FUNCTION ---

load_dotenv()
CHROMA_PATH = "chroma"

def extract_structured_data(query: str, pydantic_model: BaseModel, vector_store: Chroma):
    """Uses RAG to find information and populate a Pydantic model."""
    print(f"🔄 Running extraction for: {pydantic_model.__name__}")
    
    # Retrieve more chunks to ensure the entire bill (especially all line items) is included
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 15})
    
    # Use the query to find relevant chunks, but also just get all chunks to be safe
    # For a single-page bill, retrieving all chunks is the most robust method.
    all_chunks = vector_store.get(include=["documents"])
    context_text = "\n\n---\n\n".join([doc for doc in all_chunks['documents']])

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_llm = llm.with_structured_output(pydantic_model)
    
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

    Context:
    {context}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | structured_llm
    
    try:
        response = chain.invoke({"context": context_text})
        print(f"✅ Success for: {pydantic_model.__name__}")
        return response
    except Exception as e:
        print(f"❌ Error during extraction for {pydantic_model.__name__}: {e}")
        return None

# --- 3. ORCHESTRATE THE EXTRACTION PROCESS (SIMPLIFIED) ---

def main():
    """Main function to run the full extraction pipeline."""
    print("Loading vector database...")
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    print("Vector database loaded successfully.")

    # A single, comprehensive query to guide the LLM
    extraction_query = """
    Extract all information from the utility bill. This includes:
    1.  Provider name and country.
    2.  All summary details: statement date, due date, total current charges, previous balance, and total amount due.
    3.  Account number and the full billing address (recipient, street, city, state, zip).
    4.  A complete list of ALL line items from the 'PREMISES SUMMARY' table, including premises number, descriptor, and current bill amount for each.
    5.  Any last payment information ('No Payments Received' means 0.0).
    """

    # We now call the function ONCE, asking for the entire UtilityBill object
    final_bill = extract_structured_data(extraction_query, UtilityBill, db)
    
    if final_bill:
        print("\n\n--- Final Extracted JSON Data ---")
        
        # 1. Get the JSON data as a string
        json_output = final_bill.model_dump_json(indent=2)
        
        # 2. Print it to the console (like before)
        print(json_output)
        
        # 3. Define an output file name
        output_filename = "extracted_bill.json"
        
        # 4. Write the JSON string to the file
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(json_output)
            
        print(f"\n✅ Successfully saved JSON to {output_filename}")
        
    else:
        print("\n\n--- ❌ Failed to create final JSON ---")
        print("Data extraction failed.")

if __name__ == "__main__":
    main()