"""Convenient condor submit wrapper."""
from os import path
import time
import subprocess
import logging

import click

from clustertools.logging import LOGFORMAT


LOGGER = logging.getLogger(__name__)


@click.command()
@click.argument('command', nargs=-1)
# Job specification.
@click.option("--request_cpus", type=click.INT, default=1,
              help="The number of CPUs requested.")
@click.option("--request_memory", type=click.INT,
              help="Requested memory in Gb.")
@click.option("--request_gpus", type=click.INT, default=0,
              help="The number of GPUs requested.")
@click.option("--prio", type=click.INT, default=0,
              help="The priority to add to default.")
# Job restrictions.
@click.option("--allow_gpu_nodes_for_cpuonly", type=click.BOOL, default=False, is_flag=True,
              help=("By default, if only CPUs are requested, no GPU nodes are "
                    "considered. If set, GPU nodes are also taken into "
                    "consideration."))
@click.option("--avoid_nodes", type=click.STRING, default=None,
              help="Comma separated list of nodes to avoid, e.g., 'g008,g009'.")
@click.option("--force_node", type=click.STRING, default=None,
              help="Force submission on a specific node, e.g., 'g008'.")
@click.option("--gpu_memory_gt", type=click.INT, default=None,
              help="GPU memory must be greater than this threshold (Mb).")
@click.option("--gpu_memory_lt", type=click.INT, default=None,
              help="GPU memory must be less than this threshold (Mb).")
# Logging.
@click.option("--stdout_fp", type=click.Path(dir_okay=False, writable=True), default=None,
              help="Filepath to redirect the stdout to. Defaults to `$exec_$date_out.txt`.")
@click.option("--stderr_fp", type=click.Path(dir_okay=False, writable=True), default=None,
              help="Filepath to redirect the stderr to. Default to `stdout_fp`.")
# Notification.
@click.option("--notify_success", type=click.BOOL, default=False, is_flag=True,
              help="Send an email to user on success.")
@click.option("--notify_failure", type=click.BOOL, default=False, is_flag=True,
              help="Send an email to user on failure.")
@click.option("--notify_email", type=click.STRING, default=None,
              help=("Specify a custom email address for notification. Defaults "
                    "to MPI email address."))
def cli(command,
        request_cpus=1, request_memory=4, request_gpus=0, prio=0,
        allow_gpu_nodes_for_cpuonly=False, avoid_nodes=None, force_node=None,
        gpu_memory_gt=None, gpu_memory_lt=None,
        stdout_fp=None, stderr_fp=None,
        notify_success=False, notify_failure=False, notify_email=None):
    LOGGER.info("Preparing to submit cluster job...")
    # Checks.
    assert request_cpus > 0 and request_memory > 0 and request_gpus > 0 and prio > 0
    if avoid_nodes is not None and force_node is not None:
        LOGGER.warn("--avoid_nodes and --force_node set! This calls for trouble!")
    if gpu_memory_lt is not None and gpu_memory_gt is not None:
        LOGGER.warn("--gpu_memory_lt and --gpu_memory_gt both set!")
    if (gpu_memory_lt is not None or gpu_memory_gt is not None) and request_gpus == 0:
        raise Exception("Requested GPU restrictions without a GPU!")
    # Get executable command.
    full_command = subprocess.check_output(['which', command[0]]).strip()
    LOGGER.info("Executing `%s` with parameters: %s.",
                full_command, str(command[1:]))
    LOGGER.info("Creating job specification...")
    # Executable.
    condor_commandp.extend(["-append", "executable="+full_command])
    condor_commandp.extend(["-append", "arguments=\"{}\"".format([prm.replace('"', '\\"')
                            for prm in command[1:]].join(" "))])
    condor_commandp = ['condor_submit']
    requirements = []
    condor_commandp.extend(["-append", "request_cpus={}".format(request_cpus)])
    condor_commandp.extend(["-append", "request_memory={}".format(request_memory)])
    if gpus > 0:
        condor_commandp.extend(["-append", "request_gpus={}".format(request_gpus)])
    condor_commandp.extend(["-append", "priority={}".format(prio)])
    # Build the requirements.
    if not allow_gpu_nodes_for_cpuonly and gpus == 0:
        requirements.append("TARGET.TotalGPUs=?=0")
    if avoid_nodes is not None:
        requirements.append(["Machine!=\"{}.internal.cluster.is.localnet\"".format(mname)
                             for mname in avoid_nodes.split(",")].join("&&"))
    if force_node is not None:
        requirements.append("Machine==\"{}.internal.cluster.is.localnet\"".format(force_node))
    if gpu_memory_gt is not None:
        requirements.append("TARGET.CUDAGlobalMemoryMb>{}".format(gpu_memory_gt))
    if gpu_memory_lt is not None:
        requirements.append("TARGET.CUDAGlobalMemoryMb<{}".format(gpu_memory_lt))
    condor_commandp.extend(["-append", "requirements={}".format(requirements.join("&&"))])
    # Logging options.
    if stdout_fp is None:
        stdout_fp = path.abspath(path.basename(full_command) + '_' +
                                 time.strftime("%Y-%m-%d_%H-%M-%S") + '_out.txt')
    if stderr_fp is None:
        stderr_fp = stdout_fp
    condor_commandp.extend(["-append", "output="+stdout_fp])
    condor_commandp.extend(["-append", "error="+stderr_fp])
    # Notification options.
    if notify_failure:
        if notify_success:
            notify_string = "Always"
        else:
            notify_string = "Error"
    elif notify_success:
        notify_string = "Complete"
    else:
        notify_string = "Never"
    condor_commandp.extend(["-append", "notification="+notify_string])
    condor_commandp.extend(["-append", "notify_user="+notify_email])
    LOGGER.debug("Created submission command: `%s`.", str(condor_commandp))
    LOGGER.info("Submitting...")
    subprocess.check_call(condor_commandp)
    LOGGER.info("Done.")


if __name__ == '__main__':
    logging.basicConfig(
        format=LOGFORMAT,
        level=logging.DEBUG)
    cli()
