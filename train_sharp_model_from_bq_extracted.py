from sharp_line_dashboard import (
    train_with_champion_wrapper,
    train_timing_opportunity_model,
)


def train_sharp_model_for_market(
    *,
    sport,
    market,
    bucket_name,
    log_func=print,
    **kwargs,
):
    """
    Production sharp-model entrypoint.

    ALL available scored history is used by default for every sport.
    """
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
    sport,
    bucket_name=None,
    log_func=print,
    **kwargs,
):
    """
    Production timing-model entrypoint.

    ALL available scored history is used by default for every sport.
    """
    kwargs.pop("days_back", None)
    return train_timing_opportunity_model(
        sport=sport,
        days_back=None,
        gcs_bucket=bucket_name,
        **kwargs,
    )
