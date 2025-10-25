from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import re

app = FastAPI(title="Sentiment Analysis API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SentimentRequest(BaseModel):
    sentences: List[str]

class SentimentResult(BaseModel):
    sentence: str
    sentiment: str

class SentimentResponse(BaseModel):
    results: List[SentimentResult]

def analyze_sentiment(text: str) -> str:
    text_lower = text.lower().strip()
    
    # Expanded happy sentiment indicators
    happy_words = {
        'love', 'like', 'great', 'awesome', 'amazing', 'wonderful', 'fantastic',
        'excellent', 'good', 'nice', 'happy', 'joy', 'pleased', 'delighted',
        'perfect', 'brilliant', 'outstanding', 'superb', 'enjoy', 'best',
        'beautiful', 'joyful', 'ecstatic', 'thrilled', 'excited', 'glad',
        'pleasure', 'satisfied', 'content', 'grateful', 'thankful', 'blessed',
        'lucky', 'fortunate', 'positive', 'optimistic', 'hopeful', 'cheerful',
        'jolly', 'merry', 'blissful', 'euphoric', 'radiant', 'vibrant', 'bliss',
        'delight', 'elation', 'jubilation', 'rapture', 'contentment', 'serenity',
        'peaceful', 'calm', 'relaxed', 'comfortable', 'cozy', 'warm', 'friendly',
        'kind', 'generous', 'helpful', 'supportive', 'encouraging', 'inspiring',
        'motivating', 'empowering', 'uplifting', 'refreshing', 'rejuvenating',
        'revitalizing', 'energizing', 'invigorating', 'stimulating', 'exciting',
        'adventurous', 'fun', 'entertaining', 'interesting', 'fascinating',
        'captivating', 'engaging', 'absorbing', 'immersive', 'compelling'
    }
    
    # Expanded sad sentiment indicators
    sad_words = {
        'hate', 'terrible', 'awful', 'horrible', 'bad', 'sad', 'angry', 'mad',
        'upset', 'disappointed', 'frustrated', 'annoyed', 'depressed', 'miserable',
        'unhappy', 'sorrow', 'pain', 'hurt', 'suffering', 'distress', 'despair',
        'gloomy', 'dismal', 'bleak', 'melancholy', 'heartbroken', 'devastated',
        'crushed', 'defeated', 'hopeless', 'desperate', 'tragic', 'unfortunate',
        'regret', 'dislike', 'disgust', 'revulsion', 'contempt', 'scorn', 'bitter',
        'resentful', 'furious', 'outraged', 'irritated', 'aggravated', 'displeased',
        'dissatisfied', 'discontent', 'disheartened', 'discouraged', 'dispirited',
        'dejected', 'downcast', 'forlorn', 'woeful', 'dreadful', 'horrific',
        'appalling', 'disgusting', 'repulsive', 'vile', 'nasty', 'ugly', 'gross',
        'offensive', 'insulting', 'humiliating', 'embarrassing', 'shameful',
        'guilty', 'remorseful', 'ashamed', 'sorry', 'apologetic', 'regretful',
        'lonely', 'isolated', 'abandoned', 'rejected', 'excluded', 'ignored',
        'neglected', 'forgotten', 'unwanted', 'unloved', 'unappreciated',
        'worthless', 'useless', 'pointless', 'meaningless', 'hopeless', 'helpless',
        'powerless', 'weak', 'tired', 'exhausted', 'fatigued', 'weary', 'drained',
        'burned', 'stressed', 'anxious', 'worried', 'nervous', 'fearful', 'scared',
        'afraid', 'terrified', 'panicked', 'alarmed', 'shocked', 'stunned'
    }
    
    # Intensifiers that amplify sentiment
    intensifiers = {
        'very', 'really', 'extremely', 'absolutely', 'completely', 'totally',
        'utterly', 'entirely', 'fully', 'thoroughly', 'highly', 'exceptionally',
        'incredibly', 'unbelievably', 'remarkably', 'particularly', 'especially',
        'extraordinarily', 'immensely', 'intensely', 'profoundly', 'deeply'
    }
    
    # Negations that reverse sentiment
    negations = {'not', "n't", 'no', 'never', 'nothing', 'nobody', 'nowhere'}
    
    happy_count = 0
    sad_count = 0
    
    words = re.findall(r"\b[\w']+\b", text_lower)
    
    # Analyze each word with context
    i = 0
    while i < len(words):
        word = words[i]
        weight = 1  # Default weight
        
        # Check for intensifiers
        if word in intensifiers and i + 1 < len(words):
            next_word = words[i + 1]
            if next_word in happy_words or next_word in sad_words:
                weight = 2  # Double weight for intensified words
                i += 1  # Skip the intensifier
                word = next_word
        
        # Check for negations
        elif word in negations and i + 1 < len(words):
            next_word = words[i + 1]
            if next_word in happy_words:
                sad_count += 2  # Negated happy becomes strongly sad
                i += 2
                continue
            elif next_word in sad_words:
                happy_count += 2  # Negated sad becomes strongly happy
                i += 2
                continue
        
        # Count sentiment words with weights
        if word in happy_words:
            happy_count += weight
        elif word in sad_words:
            sad_count += weight
            
        i += 1
    
    # Check for emotional punctuation
    if '!' in text:
        # Exclamation often indicates strong emotion
        if happy_count > 0:
            happy_count += 1
        elif sad_count > 0:
            sad_count += 1
    
    if '?' in text and ('why' in text_lower or 'how' in text_lower):
        # Questions like "why is this happening?" often indicate sadness
        sad_count += 1
    
    # Check for common phrases with strong sentiment
    strong_happy_phrases = [
        'i love', 'i like', 'so happy', 'very happy', 'really happy',
        'so glad', 'very glad', 'really glad', 'so excited', 'very excited'
    ]
    
    strong_sad_phrases = [
        'i hate', 'i dislike', 'so sad', 'very sad', 'really sad',
        'so angry', 'very angry', 'really angry', 'so disappointed'
    ]
    
    for phrase in strong_happy_phrases:
        if phrase in text_lower:
            happy_count += 2
    
    for phrase in strong_sad_phrases:
        if phrase in text_lower:
            sad_count += 2
    
    # Determine final sentiment with adjusted thresholds
    if happy_count > 0 and happy_count > sad_count:
        return "happy"
    elif sad_count > 0 and sad_count > happy_count:
        return "sad"
    else:
        # If both are equal or both zero, check for any sentiment indicators
        if happy_count > 0:
            return "happy"
        elif sad_count > 0:
            return "sad"
        else:
            return "neutral"

@app.post("/sentiment", response_model=SentimentResponse)
async def analyze_batch_sentiment(request: SentimentRequest):
    results = []
    
    for sentence in request.sentences:
        sentiment = analyze_sentiment(sentence)
        results.append(SentimentResult(
            sentence=sentence,
            sentiment=sentiment
        ))
    
    return SentimentResponse(results=results)

@app.get("/")
async def root():
    return {"message": "Sentiment Analysis API is running", "endpoint": "POST /sentiment for sentiment analysis"}

@app.post("/")
async def root_post(request: SentimentRequest = None):
    if request and request.sentences:
        return await analyze_batch_sentiment(request)
    else:
        return {
            "message": "Sentiment Analysis API is running", 
            "instruction": "Send POST requests to /sentiment endpoint with JSON: {'sentences': ['sentence1', 'sentence2', ...]}"
        }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "endpoint": "/sentiment"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
