from pipeline.pipeline_registration import InfrastructureRegistration, register_new_infrastructure

register_new_infrastructure(
    InfrastructureRegistration(
        name="athena",
        executor_path="pipeline.adapters.athena.AthenaExecutor",
        completion_path="pipeline.completion.PolledCompletionSource",
    )
)

# The same pipeline, running the benchmark on the machine that hosts it. It is
# how submit -> poll -> download gets exercised at all: cluster credentials are
# personal, so nothing else can run that path in a test.
register_new_infrastructure(
    InfrastructureRegistration(
        name="local",
        executor_path="pipeline.adapters.local.LocalExecutor",
        completion_path="pipeline.completion.PolledCompletionSource",
    )
)
