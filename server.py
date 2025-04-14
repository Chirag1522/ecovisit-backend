from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Create OpenAI client for Groq
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predict")
def predict(city: str):
    if not city:
        return JSONResponse(status_code=400, content={"error": "City name is required."})

    start_date = datetime.now().date()
    future_dates = [start_date + timedelta(days=i) for i in range(7)]
    formatted_dates = [date.strftime("%Y-%m-%d") for date in future_dates]

    prompt = (
        f"Predict the Air Quality Index (AQI) for {city} for the next 7 days starting from {start_date}.\n"
        f"Return a valid JSON list of exactly 7 objects with the format:\n"
        f" - 'ds': date (format: YYYY-MM-DD)\n"
        f" - 'yhat': predicted AQI (0-500)\n"
        f" - 'yhat_lower': lower bound\n"
        f" - 'yhat_upper': upper bound\n"
        f"Ensure the data is realistic for {city} and do not include any explanation or extra text."
    )

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1000
        )

        content = response.choices[0].message.content.strip()

        print("Raw response from Groq:", content)

        predictions = json.loads(content)

        if not isinstance(predictions, list) or len(predictions) != 7:
            return JSONResponse(
                status_code=422,
                content={"error": "Invalid response format from AI", "raw": content}
            )

        return {
            "city": city,
            "predictions": predictions
        }

    except json.JSONDecodeError as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Invalid JSON received from Groq", "details": str(e), "raw": content}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
