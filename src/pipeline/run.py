import logging
import sys

from pipeline.pipeline_builder import PipelineBuilder

logger = logging.getLogger(__name__)

def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if len(argv) !=1 or argv[0] not in ("worker","downloader","poller"):
        raise SystemExit("entrypoint needs to follow this pattern: python -m pipeline.run <worker|downloader|poller>")

    service_to_run = argv[0]
    logger.info(f"starting: {service_to_run}")
    service = PipelineBuilder().build(service_to_run)
    service.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))