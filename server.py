from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import openai  # Using Groq's OpenAI-compatible API
import json
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv
load_dotenv()


app = FastAPI()

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for dev; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set Groq API key and base
openai.api_key = os.getenv("GROQ_API_KEY")   
openai.api_base = "https://api.groq.com/openai/v1"

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
        response = openai.ChatCompletion.create(
            model="qwen-2.5-coder-32b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1000
        )

        content = response["choices"][0]["message"]["content"].strip()

        # Log raw response
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
