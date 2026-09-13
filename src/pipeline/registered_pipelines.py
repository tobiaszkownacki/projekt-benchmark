from pipeline.pipeline_registration import InfrastructureRegistration, register_new_infrastructure

register_new_infrastructure(
    InfrastructureRegistration(
        name="athena",
        executor_path="pipeline.adapters.athena.AthenaExecutor",
        completion_path="pipeline.adapters.athena.AthenaCompletionSource",
    )
)
