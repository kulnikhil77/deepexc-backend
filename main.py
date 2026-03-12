from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import hmac, hashlib, os, json, secrets
from pymongo import MongoClient
from datetime import datetime, timedelta

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MONGODB_URI = os.environ.get("MONGODB_URI")
WEBHOOK_SIGN = os.environ.get("WEBHOOK_SIGN")
HF_APP_URL = "https://kulnikhil77-deep-excavation-tool.hf.space"

def get_payments():
    client = MongoClient(MONGODB_URI)
    return client["deepexc"]["payments"]

def get_tokens():
    client = MongoClient(MONGODB_URI)
    return client["deepexc"]["tokens"]

@app.get("/")
def root():
    return {"status": "Deep Excavation Payment API running"}

@app.post("/webhook/razorpay")
async def razorpay_webhook(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Razorpay-Signature", "")

    mac = hmac.new(WEBHOOK_SIGN.encode(), body, hashlib.sha256)
    expected = mac.hexdigest()

    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=400, detail="Invalid signature")

    data = json.loads(body)
    event = data.get("event", "")

    if event == "payment_link.paid":
        payment = data["payload"]["payment"]["entity"]
        payment_id = payment["id"]
        
        # Generate one-time token
        token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(minutes=30)
        
        get_payments().insert_one({
            "payment_id": payment_id,
            "verified": True,
            "created_at": datetime.utcnow()
        })
        
        get_tokens().insert_one({
            "token": token,
            "payment_id": payment_id,
            "used": False,
            "expires_at": expires_at
        })

    return {"status": "ok"}

@app.get("/paid-redirect")
def paid_redirect():
    """Razorpay redirects here after payment — wait for webhook then redirect to HF."""
    import time
    # Wait up to 6 seconds for webhook to fire and store token
    for _ in range(3):
        time.sleep(2)
        db = get_tokens()
        record = db.find_one(
            {"used": False, "expires_at": {"$gt": datetime.utcnow()}},
            sort=[("expires_at", -1)]
        )
        if record:
            break
    return RedirectResponse(url=f"{HF_APP_URL}?paid=pending", status_code=302)

@app.get("/verify-token/{token}")
def verify_token(token: str):
    db = get_tokens()
    record = db.find_one({"token": token, "used": False})
    if not record:
        raise HTTPException(status_code=404, detail="Invalid or expired token")
    if record["expires_at"] < datetime.utcnow():
        raise HTTPException(status_code=410, detail="Token expired")
    # Mark as used
    db.update_one({"token": token}, {"$set": {"used": True}})
    return {"verified": True}

@app.get("/latest-token")
def get_latest_token():
    """Returns the latest unused token — called by HF app after redirect."""
    db = get_tokens()
    record = db.find_one(
        {"used": False, "expires_at": {"$gt": datetime.utcnow()}},
        sort=[("expires_at", -1)]
    )
    if record:
        return {"token": record["token"]}
    raise HTTPException(status_code=404, detail="No valid token found")

