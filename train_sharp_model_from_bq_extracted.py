from sharp_line_dashboard import (
    train_with_champion_wrapper,
    train_timing_opportunity_model,
)


def train_sharp_model_for_market(
    *,
    sport: str,
    market: str,
    bucket_name: str,
    log_func=print,
    **kwargs,
):
    """
    Train the sharp model using ALL available scored history.
    """

    # Do not allow an old caller to accidentally reintroduce
    # a 700/900-day training restriction.
    kwargs.pop("days_back", None)

    return train_with_champion_wrapper(
        sport=sport,
        market=market,
        bucket_name=bucket_name,
        log_func=log_func,
        days_back=None,
        **kwargs,
    )


def train_timing_model_for_market(
    *,
    sport: str,
    bucket_name: str = None,
    log_func=print,
    **kwargs,
):
    """
    Train the timing model using ALL available scored history.
    """

    kwargs.pop("days_back", None)

    return train_timing_opportunity_model(
        sport=sport,
        days_back=None,
        gcs_bucket=bucket_name,
    )
