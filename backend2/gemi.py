from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
from typing import List
import time
import json
import tempfile

from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_docling import DoclingLoader

import google.generativeai as genai

genai.configure(api_key="AIzaSyBTnYPIUOhSvGn8Rc8E9P-r2cmHE2LWys4")
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction="""
        You are a precise data extraction assistant. You must:
        - Return only in the specified format
        - Never calculate years of experience that exceed realistic bounds
        - Return 'NA' when information is unclear or uncertain
        """,
    )

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "processed"
CHECKPOINT_DIR = "checkpoints"  # Directory for saving progress
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def parse_llm_response(response: str) -> tuple:
    """Parse the LLM response to extract city and YOE."""
    try:
        parts = [p.strip() for p in response.split(',')]
        city_part = next((p for p in parts if 'city' in p.lower()), '')
        city = city_part.split(':')[1].strip() if ':' in city_part else None
        yoe_part = next((p for p in parts if 'yoe' in p.lower()), '')
        yoe = yoe_part.split(':')[1].strip() if ':' in yoe_part else None
        try:
            yoe = float(yoe) if yoe and yoe.lower() != 'null' else None
        except ValueError:
            yoe = None
        return city, yoe
    except Exception as e:
        print(f"Error parsing response: {str(e)}")
        return None, None

def save_checkpoint(df: pd.DataFrame, filename: str, checkpoint_number: int):
    """Save progress checkpoint."""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{checkpoint_number}_{filename}")
    df.to_csv(checkpoint_path, index=False)
    print(f"Checkpoint {checkpoint_number} saved: {checkpoint_path}")

def load_latest_checkpoint(filename: str) -> tuple[pd.DataFrame, int]:
    """Load the latest checkpoint if it exists."""
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(filename)]
    if not checkpoints:
        return None, 0
    
    latest = max(checkpoints, key=lambda x: int(x.split('_')[1]))
    checkpoint_number = int(latest.split('_')[1])
    df = pd.read_csv(os.path.join(CHECKPOINT_DIR, latest))
    print(f"Loaded checkpoint {checkpoint_number}: {latest}")
    return df, checkpoint_number

@app.post("/process_file")
async def process_file_and_extract_links(file: UploadFile = File(...)):
    """
    Upload CSV, extract resume links, process resumes, and create enriched CSV.
    Saves progress incrementally and can resume from checkpoints.
    """
    
    try:
        total_start_time = time.time()
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            realfile = await file.read()
            buffer.write(realfile)
        
        # Read the input file
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file_path, on_bad_lines="skip")
        elif file.filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format.")

        if "resumelink" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'resumelink' column.")

        # Check for existing checkpoint
        checkpoint_df, start_index = load_latest_checkpoint(file.filename)
        if checkpoint_df is not None:
            df = checkpoint_df
            print(f"Resuming from index {start_index}")
        else:
            start_index = 0
            df['city'] = None
            df['years_of_experience'] = None

        resume_links = df["resumelink"].dropna().tolist()
        extracted_data = []
        timing_data = []
        checkpoint_interval = 10  # Save every 10 processed resumes
       
        
        for index, link in enumerate(resume_links[start_index:], start=start_index):
            try:
                print(f"\nProcessing resume {index + 1}/{len(resume_links)}: {link}")
                start_time = time.time()
                
                # Load document
                if link.lower().endswith('.pdf'):
                    loader = PyPDFLoader(link)
                elif link.lower().endswith('.docx'):
                    loader = DoclingLoader(file_path=link)
                else:
                    continue
                
                pages = loader.load_and_split()
                loading_time = time.time() - start_time
                print(f"Loading time: {loading_time:.2f} seconds")
                
                llm_start_time = time.time()

                chat = model.start_chat()

                prompt = f"""Extract the current city and years of experience (YOE) from the Input text using these rules:

                For City:
                1. Extract the current city of residence
                2. Look for keywords like "current location", "residing in", "based in"
                3. If multiple cities, prioritize the one associated with current job/residence
                4. If no clear current city, return 'NA'

                For Years of Experience (YOE):
                1. Calculate based on explicit work history dates
                2. For freshers/recent graduates with no prior experience, return 0
                3. If work history shows less than 1 year, round to nearest 0.5
                4. Return 'NA' if:
                - Dates are unclear or conflicting
                - Calculated experience exceeds (2025 - graduation_year - 18)
                - Experience seems unrealistic (>40 years)
                5. Don't include:
                - Internships unless explicitly stated as work experience
                - Education period as experience
                - Overlapping job periods multiple times

                The current year is 2025.
                Strictly return in this format without any other text:
                city : <city_name/NA>, yoe : <number/NA>

                Input text:
                {pages}
                """

                # Setting up the initial context with system message
                chat = model.start_chat(
                    history=[
                        {
                            "role": "model", 
                            "parts": ["I am a precise data extraction assistant. I will return only in the specified format and use 'NA' when information is unclear or uncertain."]
                        }
                    ]
                )

                # Sending the actual prompt
                response = chat.send_message(prompt)
                print(response.text)

                processing_time = time.time() - llm_start_time
                print(f"LLM processing time: {processing_time:.2f} seconds")
                
                # response = chat_completion.choices[0].message.content
                # city, yoe = parse_llm_response(response)
                
                # # Update DataFrame and save immediately
                # df.loc[df['resumelink'] == link, 'city'] = city
                # df.loc[df['resumelink'] == link, 'years_of_experience'] = yoe
                
                # Save checkpoint at intervals
                # if (index + 1) % checkpoint_interval == 0:
                #     save_checkpoint(df, file.filename, index + 1)
                
                timing_info = {
                    "resume_no": index + 1,
                    "loading_time": loading_time,
                    "processing_time": processing_time,
                    "total_time": loading_time + processing_time
                }
                timing_data.append(timing_info)
                
                extracted_data.append({
                    "resume_no": index + 1,
                    # "data": response,
                    # "parsed_city": city,
                    # "parsed_yoe": yoe,
                    "timing": timing_info
                })

            except Exception as e:
                print(f"Error processing {link}: {str(e)}")
                # Save checkpoint on error
                save_checkpoint(df, file.filename, index + 1)
                extracted_data.append({"link": link, "error": str(e)})

        # Calculate timing statistics
        total_time = time.time() - total_start_time
        print(f"\nTotal processing time: {total_time:.2f} seconds")
        
        if timing_data:
            avg_loading_time = sum(t['loading_time'] for t in timing_data) / len(timing_data)
            avg_processing_time = sum(t['processing_time'] for t in timing_data) / len(timing_data)
            print(f"Average loading time: {avg_loading_time:.2f} seconds")
            print(f"Average processing time: {avg_processing_time:.2f} seconds")

        # Save final output
        output_filename = f"enriched_{os.path.splitext(file.filename)[0]}.csv"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        df.to_csv(output_path, index=False)

        return {
            "message": "Processing completed.", 
            "data": extracted_data,
            "output_file": output_filename,
            "timing_summary": {
                "total_time": total_time,
                "average_loading_time": avg_loading_time if timing_data else None,
                "average_processing_time": avg_processing_time if timing_data else None,
                "detailed_timing": timing_data
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/ask-llm")
async def ask_llm():
    print("Hello endpoint") 
    start_time = time.time()
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": f"Use of Ai"}],
        model="llama-3.2-1b-preview",
    )
    processing_time = time.time() - start_time
    print(processing_time)
    return {"data": chat_completion.choices[0].message.content}