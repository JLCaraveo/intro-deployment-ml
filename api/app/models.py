from pydantic import BaseModel

class PredictionRequest(BaseModel):
    opening_gross : float
    screens : float
    production_buget : float
    title_year : int
    aspect_ratio : float
    duration : float
    cast_total_facebook_likes : float
    budget : float
    imdb_score : float


class PredictionResponse(BaseModel):
    worldwide_gross : float