import os
import pickle
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# 1. Environment & API Setup
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    raise RuntimeError("TMDB_API_KEY MISSING. Put it in .env as TMDB_API_KEY=xxxx")

# 2. App Initialization
app = FastAPI(title="Movie Recommender API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Path Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

# 4. Global Data Placeholders
df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None
TITLE_TO_IDX: Optional[Dict[str, int]] = None

# 5. Pydantic Models (ORDER MATTERS)
class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None

class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None 

class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = []

class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]

# 6. Utility Helpers
def _norm_title(t: str) -> str:
    return str(t).strip().lower()

def make_img_url(path: Optional[str]) -> Optional[str]:
    if not path: return None
    return f"{TMDB_IMG_500}{path}"

async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    q = dict(params)
    q["api_key"] = TMDB_API_KEY
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{TMDB_BASE}{path}", params=q)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail="TMDB API Error")
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

async def tmdb_cards_from_results(results: List[dict], limit: int = 20) -> List[TMDBMovieCard]:
    out = []
    for m in (results or [])[:limit]:
        out.append(TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or m.get("name") or "",
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        ))
    return out

async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    data = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})
    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []) or [],
    )

async def tmdb_search_first(query: str) -> Optional[dict]:
    data = await tmdb_get("/search/movie", {"query": query, "language": "en-US"})
    results = data.get("results", [])
    return results[0] if results else None

def tfidf_recommend_titles(query_title: str, top_n: int = 10) -> List[Tuple[str, float]]:
    global df, tfidf_matrix, TITLE_TO_IDX
    key = _norm_title(query_title)
    if not TITLE_TO_IDX or key not in TITLE_TO_IDX:
        return []
    
    idx = TITLE_TO_IDX[key]
    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()
    order = np.argsort(-scores)
    
    out = []
    for i in order:
        if int(i) == idx: continue
        title_i = str(df.iloc[int(i)]["title"])
        out.append((title_i, float(scores[int(i)])))
        if len(out) >= top_n: break
    return out

async def attach_tmdb_card_by_title(title: str) -> Optional[TMDBMovieCard]:
    try:
        m = await tmdb_search_first(title)
        if not m: return None
        return TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or title,
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )
    except: return None

# 7. Startup Event: Load Pickles
@app.on_event("startup")
def load_pickles():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX
    try:
        with open(DF_PATH, "rb") as f: df = pickle.load(f)
        with open(INDICES_PATH, "rb") as f: indices_obj = pickle.load(f)
        with open(TFIDF_MATRIX_PATH, "rb") as f: tfidf_matrix = pickle.load(f)
        with open(TFIDF_PATH, "rb") as f: tfidf_obj = pickle.load(f)
        
        # Build normalized index map
        TITLE_TO_IDX = {str(k).strip().lower(): int(v) for k, v in indices_obj.items()}
        print("Pickle files loaded successfully!")
    except Exception as e:
        print(f"Error loading pickles: {e}")
        raise RuntimeError(e)

# 8. Routes
@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
):
    best = await tmdb_search_first(query)
    if not best:
        raise HTTPException(status_code=404, detail="Movie not found")

    details = await tmdb_movie_details(int(best["id"]))

    # TF-IDF logic
    recs = tfidf_recommend_titles(details.title, top_n=tfidf_top_n)
    tfidf_items = []
    for title, score in recs:
        card = await attach_tmdb_card_by_title(title)
        tfidf_items.append(TFIDFRecItem(title=title, score=score, tmdb=card))

    # Genre logic
    genre_recs = []
    if details.genres:
        genre_id = details.genres[0]["id"]
        discover = await tmdb_get("/discover/movie", {"with_genres": genre_id, "sort_by": "popularity.desc"})
        genre_recs = await tmdb_cards_from_results(discover.get("results", []), limit=genre_limit)

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs
    )