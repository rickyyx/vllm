import abc
import logging
import sys
import time

logger = logging.getLogger(__name__)


class FaultAwareDaemon:
    """A base class for creating fault tolerant for daemon process actors.

    The subclasses of this class should implement the following methods:
    - daemon_setup: A setup method that is called once before the daemon loop.
    - daemon_step: A method that is called in a loop.
    - handle_step_exception: A method that is called when an exception is

    If daemon_step raises an exception, the actor will be recreated if
    recreate_failed_actors is True, and left dead otherwise.

    Args:
        recreate_failed_actors: If True, the actor will be recreated if it
            crashes. If False, the actor will be left dead.
        delay_between_actor_restarts_s: The delay between actor restarts in
            seconds. Only used if recreate_failed_actors is True.
    """

    def __init__(self,
                 recreate_failed_actors: bool = False,
                 delay_between_actor_restarts_s: float = 0.0) -> None:
        self._recreate_failed_actors = recreate_failed_actors
        self._delay_between_actor_restarts_s = delay_between_actor_restarts_s

    def ping(self) -> str:
        """Ping the actor. Can be used as a health check.

        Returns:
            "pong" if actor is up and well.
        """
        return "pong"

    def run(self, *args, **kwargs) -> None:
        """Run the daemon process.

        Args:
            args: Optional additional args to pass to all daemon
                implementation calls.
            kwargs: Optional additional kwargs to pass to all daemon
                implementation calls.
        """
        logger.info("Beginning the daemon process...")

        self.daemon_setup(*args, **kwargs)
        try:
            while True:
                self.daemon_step(*args, **kwargs)
        except Exception as e:
            self.handle_step_exception(e, *args, **kwargs)
            logger.exception(
                "Worker exception caught during daemon step:\n"
                "%s\n", str(e))
            # Actor should be recreated by Ray.
            if self._recreate_failed_actors:
                logger.info("Restarting actor ...")
                # Small delay to allow logs messages to propagate.
                time.sleep(self._delay_between_actor_restarts_s)
                # Kill this worker so Ray Core can restart it.
                sys.exit(1)
            # Actor should be left dead.
            else:
                logger.info(
                    "Raising exception due to failure of the daemon process ..."
                )
                raise e

    @abc.abstractmethod
    def daemon_setup(self, *args, **kwargs) -> None:
        """A setup method that is called once before the daemon loop."""
        raise NotImplementedError

    @abc.abstractmethod
    def daemon_step(self, *args, **kwargs) -> None:
        """A method that is called in a loop."""
        raise NotImplementedError

    @abc.abstractmethod
    def handle_step_exception(self, exception: Exception, *args,
                              **kwargs) -> None:
        """A method that is called when an exception is raised in daemon_step.

        Args:
            exception: The exception that was raised.
        """
        raise NotImplementedError
