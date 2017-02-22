#!/usr/bin/env python2
"""Convenient condor submit wrapper."""
import os
from os import path
import re
import time
import subprocess
import logging
import tempfile
import getpass

import click

from clustertools.log import LOGFORMAT

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format=LOGFORMAT,
    level=logging.INFO)


@click.command()
@click.argument('command', nargs=-1)
# Job specification.
@click.option("--request_cpus", type=click.INT, default=1,
              help="The number of CPUs requested.")
@click.option("--request_memory", type=click.INT, default=8,
              help="Requested memory in Gb.")
@click.option("--request_gpus", type=click.INT, default=0,
              help="The number of GPUs requested.")
@click.option("--prio", type=click.INT, default=0,
              help="The priority to add to default.")
@click.option("--shell", type=click.STRING, default=None,
              help="The execution shell. If unspecified, use the plain HTCondor env.")
@click.option("--n_copies", type=click.INT, default=1,
              help="The number of job copies to start.")
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
@click.option("--run_encaged", type=click.BOOL, default=False, is_flag=True,
              help=("Run the command within memory limits and repeat it until "
                    "it succeeds."))
@click.option("--parallel_rest_name", type=click.STRING, default=None,
              help="Specify a name for the parallel restriction. Default: username.")
@click.option("--max_parallel", type=click.INT, default=None,
              help=("Run at maximum the given number of jobs within the given "
                    "parallel restriction."))
# Logging.
@click.option("--stdout_fp", type=click.Path(writable=True), default=None,
              help=("Filepath to redirect the stdout to. Defaults to "
                    "`$exec_$date_out.log`. If it is a directory, create the "
                    "default out file in the given directory."))
@click.option("--stderr_fp", type=click.Path(dir_okay=False, writable=True), default=None,
              help="Filepath to redirect the stderr to. Defaults to `stdout_fp`.")
# Notification.
@click.option("--notify_success", type=click.BOOL, default=False, is_flag=True,
              help="Send an email to user on success.")
@click.option("--notify_failure", type=click.BOOL, default=False, is_flag=True,
              help="Send an email to user on failure.")
@click.option("--notify_always", type=click.BOOL, default=False, is_flag=True,
              help="Shorthand for `--notify_success --notify_failure`.")
@click.option("--notify_email", type=click.STRING, default=None,
              help=("Specify a custom email address for notification. Defaults "
                    "to MPI email address."))
def cli(command,  # pylint: disable=too-many-statements, too-many-branches, too-many-arguments
        request_cpus=1, request_memory=4, request_gpus=0, prio=0, shell=None, n_copies=1,
        allow_gpu_nodes_for_cpuonly=False, avoid_nodes=None, force_node=None,
        gpu_memory_gt=None, gpu_memory_lt=None, run_encaged=False,
        parallel_rest_name=None, max_parallel=None,
        stdout_fp=None, stderr_fp=None,
        notify_success=False, notify_failure=False, notify_always=False, notify_email=None):
    """Submit a cluster job."""
    LOGGER.info("Preparing to submit cluster job...")
    # Checks.
    assert request_cpus > 0 and request_memory > 0 and request_gpus >= 0 and prio >= 0
    if avoid_nodes is not None and force_node is not None:
        LOGGER.warn("--avoid_nodes and --force_node set! This calls for trouble!")
    if gpu_memory_lt is not None and gpu_memory_gt is not None:
        LOGGER.warn("--gpu_memory_lt and --gpu_memory_gt both set!")
    if (gpu_memory_lt is not None or gpu_memory_gt is not None) and request_gpus == 0:
        raise Exception("Requested GPU restrictions without a GPU!")
    if parallel_rest_name is None:
        parallel_rest_name = getpass.getuser()
    # Unify unicode handling.
    command = [str(cmd) for cmd in command]
    # Get executable command.
    full_command = subprocess.check_output(['which', command[0]]).strip()
    LOGGER.info("Executing `%s` with parameters: %s.",
                full_command, str(command[1:]))
    LOGGER.info("Creating job specification...")
    if shell is not None:
        shell_command = subprocess.check_output(['which', shell]).strip()
        full_inner_command = full_command
        full_command = shell_command
        command = [shell_command, "-c",
                   " ".join([full_inner_command] +
                            [cmd.strip() if not " " in cmd.strip() else
                             '"' + cmd.replace('"', '\\"') + '"'
                             for cmd in command[1:]])]
    condor_sub = []
    # Executable.
    # (see http://research.cs.wisc.edu/htcondor/manual/current/condor_submit.html)
    # Single ticks within the arguments must be escaped, and arguments with
    # spaces must be put in single ticks.
    command = [cmd.replace("'", "''") for cmd in command]
    command = [cmd.replace('"', '""') for cmd in command]
    command = [cmd.strip() if not " " in cmd.strip() else "'" + cmd + "'"
               for cmd in command]
    if run_encaged:
        encage_command = subprocess.check_output(['which', 'encaged']).strip()
        LOGGER.debug("Using encage command `%s`.", encage_command)
        condor_sub.append("executable="+encage_command)
        condor_sub.append("arguments=\"{}\"".format(" ".join(
            [str(request_memory - 1), '--'] + command)))
    else:
        condor_sub.append("executable="+full_command)
        condor_sub.append("arguments=\"{}\"".format(" ".join(command[1:])))
    condor_sub.append("request_cpus={}".format(request_cpus))
    condor_sub.append("request_memory={}".format(request_memory * 1024))
    if request_gpus > 0:
        condor_sub.append("request_gpus={}".format(request_gpus))
    condor_sub.append("priority={}".format(-999+prio))
    if max_parallel is not None:
        parallel_tokens_per_job = 10000 // max_parallel
        LOGGER.info("Using parallel restriction `user.%s:%d`.",
                    parallel_rest_name, parallel_tokens_per_job)
        condor_sub.append("concurrency_limits=user.%s:%d" % (
            parallel_rest_name, parallel_tokens_per_job))
    # Build the requirements.
    requirements = []
    if not allow_gpu_nodes_for_cpuonly and request_gpus == 0:
        requirements.append("TARGET.TotalGPUs=?=0")
    if avoid_nodes is not None:
        requirements.append("&&".join(
            ["Machine!=\"{}.internal.cluster.is.localnet\"".format(mname)
             for mname in avoid_nodes.split(",")]))
    if force_node is not None:
        requirements.append("Machine==\"{}.internal.cluster.is.localnet\"".format(force_node))
    if gpu_memory_gt is not None:
        requirements.append("TARGET.CUDAGlobalMemoryMb>{}".format(gpu_memory_gt))
    if gpu_memory_lt is not None:
        requirements.append("TARGET.CUDAGlobalMemoryMb<{}".format(gpu_memory_lt))
    condor_sub.append("requirements={}".format("&&".join(requirements)))
    # Logging options.
    if stdout_fp is None or path.isdir(stdout_fp):
        if stdout_fp is not None and path.isdir(stdout_fp):
            logdir = stdout_fp[:]
        else:
            logdir = os.getcwd()
        if shell is not None:
            stdout_fp = path.abspath(
                path.join(
                    logdir,
                    path.basename(full_inner_command) + '_' +
                    time.strftime("%Y-%m-%d_%H-%M-%S") + '_$(Process)_out.log'))
        else:
            stdout_fp = path.abspath(
                path.join(
                    logdir,
                    path.basename(full_command) + '_' +
                    time.strftime("%Y-%m-%d_%H-%M-%S") + '_$(Process)_out.log'))
    if stderr_fp is None:
        stderr_fp = stdout_fp
    condor_sub.append("output="+stdout_fp)
    condor_sub.append("error="+stderr_fp)
    # Notification options.
    if notify_failure or notify_always:
        if notify_success or notify_always:
            notify_string = "Always"
        else:
            notify_string = "Error"
    elif notify_success:
        notify_string = "Complete"
    else:
        notify_string = "Never"
    condor_sub.append("notification="+notify_string)
    if notify_email is not None:
        condor_sub.append("notify_user="+notify_email)
    condor_sub.append("queue %d" % (n_copies))
    with tempfile.NamedTemporaryFile(mode='w') as subfile:
        LOGGER.info("Using temporary submission file `%s`.", subfile.name)
        for line in condor_sub:
            subfile.file.write(line + "\n")
        subfile.file.flush()
        LOGGER.debug("Created submission file as: `%s`.", str("\n".join(condor_sub)))
        subfile.file.close()
        LOGGER.debug("File contents:")
        LOGGER.debug("--------------------------------------------------------")
        with open(subfile.name, 'r') as debugfile:
            for line in debugfile:
                LOGGER.debug(line.strip())
        LOGGER.debug("--------------------------------------------------------")
        LOGGER.info("Submitting...")
        try:
            output = subprocess.check_output(['condor_submit', subfile.name])
            regexm = re.search("submitted to cluster (\d*)", output)
            job_id = int(regexm.group(1))
            LOGGER.info("Job submitted with id %d.", job_id)
            if prio != 0:
                LOGGER.info("Setting priority...")
                subprocess.check_call(['condor_prio',
                                       '+%d' % (prio),
                                       str(job_id)])
                LOGGER.info("Priority set.")
        except Exception as ex:  # pylint: disable=broad-except
            LOGGER.critical("Submission failed: %s!", str(ex))
    LOGGER.info("Done.")


if __name__ == '__main__':
    cli()  # pylint: disable=no-value-for-parameter
