# training_entrypoints.py
from train_controller import train_with_champion_wrapper  # <- a NON-streamlit module
from timing_model import train_timing_opportunity_model   # <- also non-streamlit

def train_sharp_model_for_market(*, sport: str, market: str, bucket_name: str, log_func=print):
    return train_with_champion_wrapper(
        sport=sport,
        market=market,
        bucket_name=bucket_name,
        log_func=log_func,
    )

def train_timing_model_for_market(*, sport: str, bucket_name: str, log_func=print):
    return train_timing_opportunity_model(
        sport=sport,
        bucket_name=bucket_name,
        log_func=log_func,
    )
