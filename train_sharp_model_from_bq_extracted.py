from sharp_line_dashboard import (
    train_with_champion_wrapper,
    train_timing_opportunity_model,
)

def train_sharp_model_for_market(*, sport: str, market: str, bucket_name: str, log_func=print, **kwargs):
    return train_with_champion_wrapper(
        sport=sport,
        market=market,
        bucket_name=bucket_name,
        **kwargs,
    )

def train_timing_model_for_market(*, sport: str, bucket_name: str = None, log_func=print, **kwargs):
    return train_timing_opportunity_model(sport=sport)
