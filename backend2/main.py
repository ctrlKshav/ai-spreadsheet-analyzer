from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
from typing import List
import time
from enum import Enum

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from langchain_community.document_loaders import PyPDFLoader
from langchain_docling import DoclingLoader
from dotenv import load_dotenv

load_dotenv()

class GeminiProvider(Enum):
    KEY_A = "A"
    KEY_B = "B"
    KEY_C = "C"
    KEY_D = "D"

# Initialize API configurations for each key
gemini_clients = {}
for key in GeminiProvider:
    env_key = f"GEMINI_API_KEY_{key.value}"
    api_key = os.getenv(env_key)
    if api_key:
        gemini_clients[key] = genai.GenerativeModel('gemini-1.5-flash')
        genai.configure(api_key=api_key)
    else:
        print(f"Warning: {env_key} not found in environment variables")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory setup
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "processed"
RESUME_DIR = "resumes"
CHECKPOINT_DIR = "checkpoints"
for directory in [UPLOAD_DIR, OUTPUT_DIR, RESUME_DIR, CHECKPOINT_DIR]:
    os.makedirs(directory, exist_ok=True)

def get_next_provider(current_index: int) -> GeminiProvider:
    """Rotate between the four Gemini API keys."""
    providers = list(GeminiProvider)
    return providers[current_index % len(providers)]

async def process_with_gemini(pages: str, provider: GeminiProvider) -> tuple[str, GeminiProvider]:
    """Process text with specified Gemini API key."""
    api_key = os.getenv(f"GEMINI_API_KEY_{provider.value}")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    Extract the current city and years of experience (YOE) from the Input text using these rules:

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
    
    try:
        response = model.generate_content(prompt , safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    })
        return response.text, provider
    except Exception as e:
        print(f"Error with key {provider.value}: {str(e)}")
        # Try next provider
        next_provider = GeminiProvider((list(GeminiProvider).index(provider) + 1) % len(GeminiProvider))
        return await process_with_gemini(pages, next_provider)

def parse_llm_response(response: str) -> tuple:
    """Parse the LLM response to extract city and YOE."""
    try:
        if not ('city' in response.lower() and 'yoe' in response.lower()):
            return "NA", "NA"
            
        parts = [p.strip() for p in response.split(',')]
        
        city_part = next((p for p in parts if 'city' in p.lower()), '')
        city = city_part.split(':')[1].strip() if ':' in city_part else "NA"
        city = "NA" if city.lower() in ['null', 'na', ''] else city
        
        yoe_part = next((p for p in parts if 'yoe' in p.lower()), '')
        yoe = yoe_part.split(':')[1].strip() if ':' in yoe_part else "NA"
        
        try:
            yoe = float(yoe) if yoe and yoe.lower() not in ['null', 'na'] else "NA"
        except ValueError:
            yoe = "NA"
            
        return city, yoe
        
    except Exception as e:
        print(f"Error parsing response: {str(e)}")
        return "NA", "NA"

def save_checkpoint(df: pd.DataFrame, filename: str, checkpoint_number: int):
    """Save progress checkpoint."""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{checkpoint_number}_{filename}")
    df.to_csv(checkpoint_path, index=False)
    print(f"Checkpoint {checkpoint_number} saved: {checkpoint_path}")

@app.post("/process_file")
async def process_file_and_extract_links(file: UploadFile = File(...)):
    """Process file with rotating Gemini API keys."""
    try:
        total_start_time = time.time()
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            realfile = await file.read()
            buffer.write(realfile)
        
        # Read input file
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file_path, on_bad_lines="skip")
        elif file.filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format.")

        if "resumelink" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'resumelink' column.")

        df['city'] = None
        df['years_of_experience'] = None

        resume_links = df["resumelink"].dropna().tolist()
        extracted_data = []
        timing_data = []
        checkpoint_interval = 10
        
        for index, link in enumerate(resume_links):
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
                
                # Write pages to a text file
                # text_file_path = os.path.join(RESUME_DIR, f"resume_{index + 1}.txt")
                # with open(text_file_path, "w") as text_file:
                #     for page in pages:
                #         text_file.write(page.page_content + "\n")
                # print(f"Pages written to {text_file_path}")
                
                # Process with rotating API keys
                provider = get_next_provider(index)
                print(provider)
                llm_start_time = time.time()
                
                response, used_provider = await process_with_gemini(str(pages), provider)
                processing_time = time.time() - llm_start_time
                
                city, yoe = parse_llm_response(response)
                print("\n", city, yoe)
                
                # Update DataFrame
                df.loc[df['resumelink'] == link, 'city'] = city
                df.loc[df['resumelink'] == link, 'years_of_experience'] = yoe
                
                if (index + 1) % checkpoint_interval == 0:
                    save_checkpoint(df, file.filename, index + 1)
                
                timing_info = {
                    "resume_no": index + 1,
                    "api_key": f"KEY_{used_provider.value}",
                    "loading_time": loading_time,
                    "processing_time": processing_time,
                    "total_time": loading_time + processing_time
                }
                timing_data.append(timing_info)
                
                extracted_data.append({
                    "resume_no": index + 1,
                    "api_key": f"KEY_{used_provider.value}",
                    "data": response,
                    "parsed_city": city,
                    "parsed_yoe": yoe,
                    "timing": timing_info
                })

            except Exception as e:
                print(f"Error processing {link}: {str(e)}")
                save_checkpoint(df, file.filename, index + 1)
                extracted_data.append({"link": link, "error": str(e)})

        # Calculate statistics
        total_time = time.time() - total_start_time
        
        if timing_data:
            avg_loading_time = sum(t['loading_time'] for t in timing_data) / len(timing_data)
            avg_processing_time = sum(t['processing_time'] for t in timing_data) / len(timing_data)
            
            # Calculate key-specific stats
            key_stats = {}
            for key in GeminiProvider:
                key_times = [t for t in timing_data if t['api_key'] == f"KEY_{key.value}"]
                if key_times:
                    key_stats[f"key_{key.value}"] = {
                        "count": len(key_times),
                        "avg_processing_time": sum(t['processing_time'] for t in key_times) / len(key_times)
                    }

        # Save output
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
                "key_statistics": key_stats if timing_data else None,
                "detailed_timing": timing_data
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)