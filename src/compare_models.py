MODEL_REGISTRY = {
    "baseline": {
        "module": "models.baseline",
        "class_name": "TemporalFusionTransformer",
        "loss_name": "quantile_loss",
    },
    "no_lstm": {
        "module": "models.no_lstm",
        "class_name": "TemporalFusionTransformer",
        "loss_name": "quantile_loss",
    },
    "no_attention": {
        "module": "models.no_attention",
        "class_name": "TemporalFusionTransformer",
        "loss_name": "quantile_loss",
    },
    "mlp_features": {
        "module": "models.mlp_features",
        "class_name": "TemporalFusionTransformer",
        "loss_name": "quantile_loss",
    },
    "transformer_only": {
        "module": "models.transformer_only",
        "class_name": "TemporalFusionTransformer",
        "loss_name": "quantile_loss",
    },
}