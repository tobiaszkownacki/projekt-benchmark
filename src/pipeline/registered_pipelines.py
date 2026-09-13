from pipeline.adapters.slurm_site import SlurmCredentials, SlurmSite
from pipeline.pipeline_registration import InfrastructureRegistration, register_new_infrastructure

ATHENA = SlurmSite(
    name="athena",
    partition="plgrid-gpu-a100",
    credentials=SlurmCredentials(
        host="ATHENA_HOST",
        user="ATHENA_USER",
        password="ATHENA_PASSWORD",
        account="ATHENA_ACCOUNT",
        remote_path="ATHENA_REMOTE_PATH",
    ),
)

register_new_infrastructure(
    InfrastructureRegistration(
        name=ATHENA.name,
        executor_path="pipeline.adapters.slurm.SlurmExecutor",
        executor_config={"site": ATHENA},
        completion_path="pipeline.completion.PolledCompletionSource",
    )
)
