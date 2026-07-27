from frontend.pooler.abstract_poller import Poller


class AthenaPoller(Poller):
    def check_status(self) -> bool:
        pass

    def check_finalized_jobs(self) -> None:
        pass

    def check_running_jobs(self) -> None:
        pass