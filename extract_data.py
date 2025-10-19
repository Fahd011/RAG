import os
import json
from typing import List, Optional
from dotenv import load_dotenv

# Pydantic is used to define the desired structured output
# 1. UPDATED IMPORT: Directly import from Pydantic, not the v1 compatibility layer
from pydantic import BaseModel, Field

# LangChain components for RAG
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# --- 2. CORRECTED PYDANTIC MODELS ---

class Provider(BaseModel):
    """Information about the utility provider."""
    name: str = Field(description="The name of the utility provider.")
    country: str = Field(description="The country where the provider operates, e.g., 'USA'.")

class BillingAddress(BaseModel):
    """Represents a billing address."""
    recipient: Optional[str] = Field(description="The name of the person or company receiving the bill.", default=None)
    streetLine1: str = Field(description="The primary street address line.")
    city: str = Field(description="The city of the address.")
    state: str = Field(description="The state or province of the address.")
    postalCode: str = Field(description="The postal or ZIP code.")

class AccountInfo(BaseModel):
    """Key account and billing address information."""
    accountNumber: str = Field(description="The unique account number for the customer.")
    billingAddress: BillingAddress

class BillSummary(BaseModel):
    """High-level summary of the utility bill."""
    statementDate: str = Field(description="The main date of the bill statement in YYYY-MM-DD format.")
    dueDate: str = Field(description="The date by which the payment is due in YYYY-MM-DD format.")
    amountDue: float = Field(description="The total amount due for this billing period.")
    previousBalance: Optional[float] = Field(description="The balance carried over from the previous statement.", default=None)
    
class UtilityBill(BaseModel):
    """The complete, structured data extracted from a utility bill."""
    provider: Provider
    accountData: AccountInfo
    summary: BillSummary

# --- SETUP THE RAG EXTRACTION FUNCTION ---

load_dotenv()
CHROMA_PATH = "chroma"

def extract_structured_data(query: str, pydantic_model: BaseModel, vector_store: Chroma):
    """Uses RAG to find information and populate a Pydantic model."""
    print(f"🔄 Running extraction for: {pydantic_model.__name__}")
    
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    relevant_chunks = retriever.invoke(query)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in relevant_chunks])

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_llm = llm.with_structured_output(pydantic_model)
    
    prompt_template = """
    Based ONLY on the following context from a utility bill, extract the requested information.
    Format your response as a JSON object that strictly follows the provided schema. Do not add any extra text or explanations.

    Context:
    {context}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | structured_llm
    
    try:
        # 3. CORRECTED INVOCATION: Pass a dictionary with a key that matches the prompt template variable.
        response = chain.invoke({"context": context_text})
        print(f"✅ Success for: {pydantic_model.__name__}")
        return response
    except Exception as e:
        print(f"❌ Error during extraction for {pydantic_model.__name__}: {e}")
        return None

# --- ORCHESTRATE THE EXTRACTION PROCESS ---

def main():
    """Main function to run the full extraction pipeline."""
    print("Loading vector database...")
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    print("Vector database loaded successfully.")

    summary_query = "Extract the provider name, statement date, due date, amount due, and previous balance."
    summary_data = extract_structured_data(summary_query, BillSummary, db)

    account_query = "What is the account number and the full billing address including recipient, street, city, state, and zip code?"
    account_data = extract_structured_data(account_query, AccountInfo, db)

    provider_query = "What is the name and country of the utility provider?"
    provider_data = extract_structured_data(provider_query, Provider, db)
    
    if all([summary_data, account_data, provider_data]):
        final_bill = UtilityBill(
            provider=provider_data,
            accountData=account_data,
            summary=summary_data
        )
        print("\n\n--- Final Extracted JSON Data ---")
        print(final_bill.model_dump_json(indent=2)) # Use model_dump_json() for Pydantic v2
    else:
        print("\n\n--- ❌ Failed to create final JSON ---")
        print("Data extraction failed for one or more sections.")

if __name__ == "__main__":
    main()