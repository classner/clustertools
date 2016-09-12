"""Run a script in hard cage limits until it succeeds."""
from os import path
import subprocess
import logging

import click

from clustertools.logging import LOGFORMAT

logging.basicConfig(level=logging.DEBUG,
                    format=LOGFORMAT)
LOGGER = logging.getLogger(__name__)
CAGE_FP = path.abspath(path.join(path.dirname(__file__), 'cage'))


@click.command()
@click.argument('memory', type=click.INT)
@click.argument('command', nargs=-1)
def cli(memory, command):
    """
    Encage a process with the given memory limit (in Gb).
    """
    command = [str(cmd) for cmd in command]
    LOGGER.info("Running encaged: `%s`.", str(command))
    assert memory is not None
    cage_options = ['--confess']
    LOGGER.info("Enforcing memory limit of %dGb.", memory)
    cage_options.extend(['-memlimit-rss', str(memory * 1024 * 1024)])
    succeeded = False
    while not succeeded:
        LOGGER.info("Starting execution attempt.")
        try:
            subprocess.check_call([CAGE_FP] + cage_options + command)
            succeeded = True
            LOGGER.info("Attempt complete.")
        except Exception as ex:
            LOGGER.info("Attempt failed: %s.", str(ex))
    LOGGER.info("Cage released.")

if __name__ == '__main__':
    cli()
