from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os, uuid, numpy as np, librosa, tensorflow as tf, json
app = FastAPI(title='Vehicle Audio Damage Detector (Keras Demo)')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
UPLOADS='uploads'; os.makedirs(UPLOADS, exist_ok=True)
MODEL_PATH='../models/vehicle_classifier.h5'  # relative to backend folder when run in root
CLASS_NAMES = ['no_issue','engine_knock','brake_squeal','flat_tire','exhaust_leak','gear_noise']
SOLUTION_MAP = {c:{'temporary':'Temporary advice','permanent':'Permanent fix','estimated_cost':1000} for c in CLASS_NAMES}

def load_model(path=MODEL_PATH):
    if not os.path.exists(path): return None
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        print('Model load error:', e)
        return None

model = load_model()

def extract_mel(path, sr=16000, duration=2.5, n_mels=64):
    y, _ = librosa.load(path, sr=sr, mono=True)
    max_len = int(sr*duration)
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)))
    else:
        y = y[:max_len]
    mel = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min())/(mel_db.max()-mel_db.min()+1e-9)
    mel_db = mel_db.astype('float32')
    # model expects shape (1, n_mels, time, 1) or (1, n_mels, time)
    return np.expand_dims(mel_db, axis=0)

@app.get('/health')
async def health(): return {'status':'ok'}

@app.get('/classes')
async def classes(): return {'classes': CLASS_NAMES}

@app.post('/predict')
async def predict(file: UploadFile = File(...), label: str = Form(None)):
    if not file.filename:
        raise HTTPException(status_code=400, detail='Empty filename')
    fname = f"{uuid.uuid4().hex}_{file.filename}"
    path = os.path.join(UPLOADS, fname)
    with open(path, 'wb') as f:
        f.write(await file.read())
    mel = extract_mel(path)  # shape (1, n_mels, time)
    if model is None:
        # return dummy uniform probabilities
        probs = np.ones(len(CLASS_NAMES))/len(CLASS_NAMES)
        idx = 0
        confidence = float(probs[idx])
        cls = CLASS_NAMES[idx]
        return JSONResponse({'class':cls, 'confidence':confidence, 'probabilities':{c:float(p) for c,p in zip(CLASS_NAMES, probs)}, 'solution':SOLUTION_MAP[cls]})
    # adapt mel shape to model input
    inp = mel[..., np.newaxis]  # (1, n_mels, time, 1)
    preds = model.predict(inp)[0]
    idx = int(np.argmax(preds))
    cls = CLASS_NAMES[idx]
    confidence = float(preds[idx])
    probs = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
    return JSONResponse({'class':cls, 'confidence':confidence, 'probabilities':probs, 'solution':SOLUTION_MAP[cls]})

@app.get('/download/sample/{name}')
async def download_sample(name: str):
    p = os.path.join('..','datasets','samples', name)
    if not os.path.exists(p):
        raise HTTPException(status_code=404)
    return FileResponse(p, filename=name)
