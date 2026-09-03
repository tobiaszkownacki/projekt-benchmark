from pipeline.infrastructure_registration import InfrastructureRegistration, register_new_infrastructure

register_new_infrastructure(
    InfrastructureRegistration(
        name="athena",
        executor_path="pipeline.executor_adapters.athena_adapter.AthenaExecutor",
        completion_path="pipeline.executor_adapters.athena_adapter.AthenaCompletionSource",
    )
)
