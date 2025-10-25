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
        'captivating', 'engaging', 'absorbing', 'immersive', 'compelling',
        'yay', 'woohoo', 'hooray', 'cool', 'sweet', 'fantastic', 'marvelous',
        'splendid', 'terrific', 'smiling', 'laughing', 'win', 'won', 'success',
        'achievement', 'accomplishment', 'victory', 'triumph', 'celebration',
        'congratulations', 'proud', 'impressed', 'admire', 'appreciate'
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
        'afraid', 'terrified', 'panicked', 'alarmed', 'shocked', 'stunned',
        'crying', 'tears', 'sobbing', 'weeping', 'grieving', 'mourning', 'loss',
        'failure', 'failed', 'mistake', 'error', 'wrong', 'broken', 'damaged',
        'ruined', 'destroyed', 'wasted', 'missed', 'lost'
    }
    
    # Intensifiers that amplify sentiment
    intensifiers = {
        'very', 'really', 'extremely', 'absolutely', 'completely', 'totally',
        'utterly', 'entirely', 'fully', 'thoroughly', 'highly', 'exceptionally',
        'incredibly', 'unbelievably', 'remarkably', 'particularly', 'especially',
        'extraordinarily', 'immensely', 'intensely', 'profoundly', 'deeply',
        'so', 'too', 'super', 'extra', 'most', 'more', 'quite', 'pretty'
    }
    
    # Negations that reverse sentiment
    negations = {'not', "n't", 'no', 'never', 'nothing', 'nobody', 'nowhere', 'none'}
    
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
            if next_word in happy_words:
                happy_count += 3  # Strong boost for intensified happy words
                i += 2
                continue
            elif next_word in sad_words:
                sad_count += 3  # Strong boost for intensified sad words
                i += 2
                continue
        
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
    
    # Check for emotional punctuation and capitalization
    if '!' in text:
        # Multiple exclamations indicate stronger emotion
        exclamation_count = text.count('!')
        if happy_count > 0:
            happy_count += exclamation_count
        elif sad_count > 0:
            sad_count += exclamation_count
        elif any(word in text_lower for word in ['yay', 'woohoo', 'hooray']):
            happy_count += 2
    
    # Check for ALL CAPS (often indicates strong emotion)
    if text.isupper():
        if happy_count > 0:
            happy_count += 2
        elif sad_count > 0:
            sad_count += 2
    
    # Check for common phrases with strong sentiment
    strong_happy_phrases = [
        'i love', 'i like', 'so happy', 'very happy', 'really happy',
        'so glad', 'very glad', 'really glad', 'so excited', 'very excited',
        'really excited', 'makes me happy', 'feel happy', 'so good', 'very good',
        'really good', 'so great', 'very great', 'really great', 'so wonderful',
        'very wonderful', 'really wonderful', 'so amazing', 'very amazing',
        'so fantastic', 'very fantastic', 'so excellent', 'very excellent',
        'so perfect', 'very perfect', 'so beautiful', 'very beautiful',
        'so nice', 'very nice', 'so cool', 'very cool', 'so sweet', 'very sweet',
        'thank you', 'thanks', 'grateful for', 'appreciate it', 'love it',
        'enjoy it', 'had fun', 'was fun', 'so much fun', 'great time',
        'good time', 'awesome time', 'amazing time', 'wonderful time'
    ]
    
    strong_sad_phrases = [
        'i hate', 'i dislike', 'so sad', 'very sad', 'really sad',
        'so angry', 'very angry', 'really angry', 'so disappointed',
        'very disappointed', 'really disappointed', 'so upset', 'very upset',
        'really upset', 'so frustrated', 'very frustrated', 'really frustrated',
        'so annoyed', 'very annoyed', 'really annoyed', 'so terrible',
        'very terrible', 'really terrible', 'so awful', 'very awful',
        'really awful', 'so horrible', 'very horrible', 'really horrible',
        'so bad', 'very bad', 'really bad', 'makes me sad', 'feel sad',
        'feel angry', 'feel upset', 'feel frustrated', 'feel annoyed',
        'hate it', 'dislike it', 'not happy', 'unhappy with', 'not good',
        'not great', 'not nice', 'bad time', 'terrible time', 'awful time',
        'horrible time', 'worst time', 'regret it', 'sorry about'
    ]
    
    for phrase in strong_happy_phrases:
        if phrase in text_lower:
            happy_count += 3
    
    for phrase in strong_sad_phrases:
        if phrase in text_lower:
            sad_count += 3
    
    # Check for positive emoji-like patterns
    if any(emoji in text for emoji in [':)', ':-)', '=)', ';)', ':-D', ':D', '<3', '❤️']):
        happy_count += 2
    
    # Check for negative emoji-like patterns  
    if any(emoji in text for emoji in [':(', ':-(', '=(', ':/', ':-/', ':\\']):
        sad_count += 2
    
    # Determine final sentiment with adjusted thresholds
    if happy_count > sad_count:
        return "happy"
    elif sad_count > happy_count:
        return "sad"
    else:
        # If both are equal, check for any sentiment indicators
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
