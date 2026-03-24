"""
Deep Excavation Tool — Payment + Report Backend
FastAPI | MongoDB | Razorpay
"""
import os, io, uuid, hmac, hashlib, json
from datetime import datetime, timedelta
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse, JSONResponse
from pymongo import MongoClient
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Env ───────────────────────────────────────────────────────────────────────
MONGODB_URI        = os.environ.get("MONGODB_URI", "")
RAZORPAY_KEY_ID    = os.environ.get("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET= os.environ.get("RAZORPAY_KEY_SECRET", "")
WEBHOOK_SIGN       = os.environ.get("WEBHOOK_SIGN", "")
HF_APP_URL         = "https://kulnikhil77-deep-excavation-tool.hf.space"
REPORT_AMOUNT_PAISE= 100   # ₹1 for testing; change to 50000 for ₹500

# ── DB ────────────────────────────────────────────────────────────────────────
def db():
    return MongoClient(MONGODB_URI)["deepexc"]

# ── Input schemas ─────────────────────────────────────────────────────────────
class LayerInput(BaseModel):
    name: str
    top: float
    bottom: float
    gamma: float
    phi: float
    c: float

class AnchorInput(BaseModel):
    level: float
    angle: float = 15.0
    spacing: float = 2.5
    free_length: float = 5.0
    fixed_length: float = 4.0
    dia: float = 32.0
    prestress: float = 0.0
    anchor_type: str = "ground"

class ProjectMeta(BaseModel):
    project_name: str = "Project"
    location: str = ""
    firm_name: str = ""
    engineer_name: str = ""
    revision: str = "R0"

class AnalysisRequest(BaseModel):
    mode: str  # "Cantilever Wall" | "Staged Excavation" | "Anchored Wall"
    layers: List[LayerInput]
    excavation_depth: float
    surcharge: float = 0.0
    gwt_behind: float = 2.0
    gwt_front: float = 1.0
    wall_section: str = "AZ 19-700"
    anchors: List[AnchorInput] = []
    wall_toe: Optional[float] = None
    wind_barrier: bool = False
    barrier_height: float = 0.0
    Vb: float = 44.0
    meta: ProjectMeta = ProjectMeta()

# ── Engine helpers ────────────────────────────────────────────────────────────
def _build_project(req: AnalysisRequest):
    from engine.models import (
        ProjectInput, SoilLayer, WaterTable, Surcharge,
        SoilType, SurchargeType
    )
    layers = []
    for lay in req.layers:
        thickness = lay.bottom - lay.top
        c = lay.c
        phi = lay.phi
        stype = (SoilType.CLAY if c > 0 and phi < 5
                 else (SoilType.SAND if c == 0 else SoilType.MIXED))
        layers.append(SoilLayer(
            name=lay.name, thickness=thickness,
            gamma=lay.gamma, gamma_sat=lay.gamma + 2.0,
            c_eff=min(c, 5.0), phi_eff=phi, c_u=c, soil_type=stype,
        ))
    wt = WaterTable(
        depth_behind_wall=req.gwt_behind,
        depth_in_excavation=req.gwt_front,
    )
    surcharges = []
    if req.surcharge > 0:
        surcharges.append(Surcharge(
            surcharge_type=SurchargeType.UNIFORM,
            magnitude=req.surcharge
        ))
    exc = req.excavation_depth
    total_soil = sum(l.thickness for l in layers)
    min_required = exc + 5.0
    if total_soil < min_required:
        extra = min_required - total_soil + 2.0
        last = layers[-1]
        layers[-1] = SoilLayer(
            name=last.name, thickness=last.thickness + extra,
            gamma=last.gamma, gamma_sat=last.gamma_sat,
            c_eff=last.c_eff, phi_eff=last.phi_eff,
            c_u=last.c_u, soil_type=last.soil_type,
        )
    return ProjectInput(
        name=req.meta.project_name,
        excavation_depth=exc,
        soil_layers=layers,
        water_table=wt,
        surcharges=surcharges,
    )

def _build_anchors(req: AnalysisRequest):
    from engine.anchored_wall import Anchor
    result = []
    for a in req.anchors:
        anc = Anchor(
            level=a.level,
            anchor_type=a.anchor_type,
            inclination=a.angle,
            horizontal_spacing=a.spacing,
            bond_stress=200.0,
            drill_diameter=0.1,
        )
        result.append(anc)
    return result

def _generate_report(req: AnalysisRequest) -> bytes:
    """Run analysis and generate report. Returns .docx bytes."""
    from engine.section_library import get_section_by_name
    project = _build_project(req)
    sec_obj = get_section_by_name(req.wall_section)
    EI = sec_obj.EI_per_m if sec_obj else 50000.0
    layers_ui = [l.dict() for l in req.layers]
    meta = req.meta

    if req.mode == "Cantilever Wall":
        from engine.cantilever_wall import (
            analyze_cantilever_fe, analyze_cantilever_blum
        )
        from reports.report_generator import generate_cantilever_report
        res_fe   = analyze_cantilever_fe(project)
        res_blum = analyze_cantilever_blum(project)
        wall_toe = (res_fe.total_wall_length if res_fe
                    else req.excavation_depth * 2.5)
        buf = generate_cantilever_report(
            project=project,
            result_fe=res_fe, result_blum=res_blum,
            wall_toe=wall_toe, EI=EI,
            wall_section_name=req.wall_section,
            layers_ui=layers_ui,
            exc_depth=req.excavation_depth,
            surcharge=req.surcharge,
            gwt_behind=req.gwt_behind,
            gwt_front=req.gwt_front,
            section_obj=sec_obj,
            project_name=meta.project_name,
            location=meta.location,
            firm_name=meta.firm_name,
            engineer_name=meta.engineer_name,
            revision=meta.revision,
        )

    elif req.mode == "Staged Excavation":
        from engine.staged_excavation import analyze_staged_excavation
        from reports.report_generator import generate_staged_report
        anchors = _build_anchors(req)
        staged_result = analyze_staged_excavation(project, anchors)
        wall_toe = req.wall_toe or req.excavation_depth * 2.0
        buf = generate_staged_report(
            project=project,
            staged_result=staged_result,
            anchors=anchors,
            wall_toe=wall_toe, EI=EI,
            wall_section_name=req.wall_section,
            layers_ui=layers_ui,
            anchors_ui=[a.dict() for a in req.anchors],
            exc_depth=req.excavation_depth,
            surcharge=req.surcharge,
            gwt_behind=req.gwt_behind,
            gwt_front=req.gwt_front,
            section_obj=sec_obj,
            project_name=meta.project_name,
            location=meta.location,
            firm_name=meta.firm_name,
            engineer_name=meta.engineer_name,
            revision=meta.revision,
        )

    elif req.mode == "Anchored Wall":
        from engine.anchored_wall import analyze_anchored_wall
        from reports.report_generator import generate_anchored_wall_report
        anchors = _build_anchors(req)
        result = analyze_anchored_wall(project, anchors)
        result_wind = None  # Wind pressure not supported in backend
        wall_toe = req.wall_toe or req.excavation_depth * 2.0
        buf = generate_anchored_wall_report(
            project=project,
            anchors=anchors,
            result=result,
            result_wind=result_wind,
            wall_toe=wall_toe, EI=EI,
            wall_section_name=req.wall_section,
            layers_ui=layers_ui,
            anchors_ui=[a.dict() for a in req.anchors],
            exc_depth=req.excavation_depth,
            surcharge=req.surcharge,
            gwt_behind=req.gwt_behind,
            gwt_front=req.gwt_front,
            section_obj=sec_obj,
            project_name=meta.project_name,
            location=meta.location,
            firm_name=meta.firm_name,
            engineer_name=meta.engineer_name,
            revision=meta.revision,
            wind_barrier=req.wind_barrier,
            barrier_height=req.barrier_height,
            Vb=req.Vb,
        )
    else:
        raise ValueError(f"Unknown mode: {req.mode}")

    return buf.getvalue() if hasattr(buf, 'getvalue') else buf

# ── API endpoints ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Deep Excavation Payment API running"}

@app.post("/create-payment")
async def create_payment(req: AnalysisRequest):
    """Generate report, store in DB, create Razorpay payment link."""
    try:
        # Generate report
        report_bytes = _generate_report(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")

    # Store in MongoDB
    report_id = str(uuid.uuid4())
    db()["reports"].insert_one({
        "report_id": report_id,
        "report_bytes": report_bytes,
        "mode": req.mode,
        "project_name": req.meta.project_name,
        "paid": False,
        "created_at": datetime.utcnow(),
        "expires_at": datetime.utcnow() + timedelta(hours=2),
    })

    # Create Razorpay payment link
    import razorpay
    client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
    callback_url = f"https://deepexc-backend.onrender.com/download/{report_id}"
    plink = client.payment_link.create({
        "amount": REPORT_AMOUNT_PAISE,
        "currency": "INR",
        "description": f"Deep Excavation Report — {req.meta.project_name}",
        "callback_url": callback_url,
        "callback_method": "get",
    })

    return {"payment_url": plink["short_url"], "report_id": report_id}

@app.get("/download/{report_id}")
async def download_report(report_id: str, request: Request):
    """Called by Razorpay redirect after payment. Verify signature and serve file."""
    params = dict(request.query_params)
    
    # Verify Razorpay signature
    plink_id   = params.get("razorpay_payment_link_id", "")
    ref_id     = params.get("razorpay_payment_link_reference_id", "")
    status     = params.get("razorpay_payment_link_status", "")
    signature  = params.get("razorpay_signature", "")

    if plink_id and signature:
        payload = f"{plink_id}|{ref_id}|{status}"
        mac = hmac.new(RAZORPAY_KEY_SECRET.encode(), payload.encode(), hashlib.sha256)
        expected = mac.hexdigest()
        if not hmac.compare_digest(expected, signature):
            raise HTTPException(status_code=400, detail="Invalid payment signature")
    else:
        raise HTTPException(status_code=400, detail="Missing payment parameters")

    # Fetch report from DB
    rec = db()["reports"].find_one({"report_id": report_id})
    if not rec:
        raise HTTPException(status_code=404, detail="Report not found or expired")

    # Mark as paid
    db()["reports"].update_one({"report_id": report_id}, {"$set": {"paid": True}})

    fname = f"{rec['project_name'].replace(' ','_')}_{rec['mode'].replace(' ','_')}_Report.docx"
    mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return StreamingResponse(
        io.BytesIO(rec["report_bytes"]),
        media_type=mime,
        headers={"Content-Disposition": f'attachment; filename="{fname}"'}
    )

@app.post("/webhook/razorpay")
async def razorpay_webhook(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Razorpay-Signature", "")
    mac = hmac.new(WEBHOOK_SIGN.encode(), body, hashlib.sha256)
    if not hmac.compare_digest(mac.hexdigest(), signature):
        raise HTTPException(status_code=400, detail="Invalid signature")
    return {"status": "ok"}

