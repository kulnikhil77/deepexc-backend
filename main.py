"""
Deep Excavation Tool - Payment Verification Backend
FastAPI + MongoDB + Razorpay Webhook
"""
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import hmac, hashlib, os, json
from pymongo import MongoClient
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MONGODB_URI = os.environ.get("MONGODB_URI")
RAZORPAY_WEBHOOK_KEY = os.environ.get("RAZORPAY_WEBHOOK_KEY")

def get_db():
    client = MongoClient(MONGODB_URI)
    return client["deepexc"]["payments"]

@app.get("/")
def root():
    return {"status": "Deep Excavation Payment API running"}

@app.post("/webhook/razorpay")
async def razorpay_webhook(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Razorpay-Signature", "")

    expected = hmac.new(
        RAZORPAY_WEBHOOK_KEY.encode(),
        body,
        hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=400, detail="Invalid signature")

    data = json.loads(body)
    event = data.get("event", "")

    if event == "payment_link.paid":
        payment = data["payload"]["payment"]["entity"]
        payment_id = payment["id"]
        amount = payment["amount"]
        email = payment.get("email", "")
        db = get_db()
        db.insert_one({
            "payment_id": payment_id,
            "amount": amount,
            "email": email,
            "verified": True,
            "created_at": datetime.utcnow()
        })

    return {"status": "ok"}

@app.get("/verify/{payment_id}")
def verify_payment(payment_id: str):
    db = get_db()
    record = db.find_one({"payment_id": payment_id, "verified": True})
    if record:
        return {"verified": True}
    raise HTTPException(status_code=404, detail="Payment not found")

