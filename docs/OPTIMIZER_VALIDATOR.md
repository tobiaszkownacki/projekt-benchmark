# Optimizer Validator

The validator checks whether an optimizer submitted by a user complies with
the benchmark interface before the job is added to the queue and sent to the
target compute infrastructure. This allows errors in optimizer code to be
detected locally without running a complete benchmark.

## Input and output

The validator receives the optimizer source code as a single Python file. The
web application passes the submitted code to a Docker container through
standard input. Inside the container, it is temporarily saved as
`/tmp/optimizer.py` on the size-limited in-memory filesystem.

The validator produces:

- exit code `0` when all checks pass;
- exit code `1` when an error is detected;
- a text report containing `PASSED` and `ERROR` messages together with details
  of the performed checks.

The report is stored as `validator_log` and returned to the user. An invalid
submission receives the `rejected` status and is not added to the task queue. A
valid submission receives the `accepted` status, after which its benchmark
tasks are created.

## How validation works

The
[`verify_optimizer.py`](../src/benchmark_core/optimization_engine/optimizers/validation/verify_optimizer.py)
script dynamically imports a class from the submitted file and checks whether:

1. the class provides the `step()` and `get_output_type()` methods;
2. the first constructor argument after `self` is named `initial_params`;
3. `get_output_type()` declares a supported NumPy or CuPy data format;
4. an optimizer instance can be created using an initial parameter vector;
5. a single `step(evaluator)` call works with a small PyTorch test model and
   returns a `bool` value;
6. the optimizer communicates with `ModelEvaluator` and, when gradients are
   calculated, updates its parameters.

The functional test uses a small model with one linear layer, a random batch
of four samples, and `CrossEntropyLoss`. This is not a test of the algorithm's
quality or convergence. It only confirms that the optimizer can be integrated
with the benchmark mechanism.

## Use in the project

Validation on the web application side is handled by
[`src/web/app/services/validator.py`](../src/web/app/services/validator.py).
For submitted source code, this module:

1. makes a preliminary classification of the optimizer as `gradient` or
   `gradient_free`;
2. passes the code to the container through standard input;
3. starts a Docker container;
4. passes the submitted file path to `verify_optimizer.py`;
5. captures the combined standard output and standard error streams;
6. considers validation successful when the process exits with code `0` and
   the report does not contain an `ERROR` message.

Submitted code is untrusted, so the container runs without network access. The root filesystem
and mounted source code are read-only, system privileges are restricted, and
execution has a time limit. The temporary source disappears when the container
is removed after validation.

Built-in optimizers from the repository do not go through this procedure. They
are accepted without another protocol check.

## Role of `Dockerfile.validator`

[`Dockerfile.validator`](../Dockerfile.validator) defines a dedicated validator
image. It installs the project dependencies, copies the `src` directory, sets
`PYTHONPATH`, switches to a non-root user, and configures
`verify_optimizer.py` as the image entry point.

The `validator_image` service in `docker-compose.yml` builds this Dockerfile as
`benchmark-validator:latest`. It is an init-style build dependency and does not
stay running. The `web` service starts a fresh container from the prebuilt
image only when an uploaded optimizer needs validation. Running
`docker compose up --build` builds both the application and validator images.

The web container uses the host Docker daemon through the mounted
`/var/run/docker.sock`. Docker Desktop normally exposes the socket to group
`0`; on native Linux, `DOCKER_GID` may need to be set to the numeric ID of the
host's Docker group. Mounting the Docker socket gives the web service powerful
access to the host daemon, so access to the web container must be protected.

Validation is enabled by default and can be disabled with
`VALIDATOR_ENABLED=0`. The image name can be changed with `VALIDATOR_IMAGE`,
but it must refer to an image containing all benchmark dependencies. If
validation is explicitly disabled or the Docker command is unavailable, the
implementation accepts the submission and records that the code was not
checked.
