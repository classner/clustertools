"""Logging tools."""
import os
import sys
import socket
import logging


LOGGER = logging.getLogger(__name__)

# Well-parsable and informative log output.
LOGFORMAT = '%(asctime)s %(name)s %(levelname)s %(message)s'


def log_sys_info(logfunc=logging.info):
    logfunc("----------------------------------------------------------------")
    logfunc("Node information:")
    logfunc("Fully qualified domain name: %s", socket.getfqdn())
    logfunc("Environment:")
    for varname, varval in os.environ.items():
        logfunc("    %s=%s", varname, varval)
    logfunc("End-Environment.")
    logfunc("Python information:")
    logfunc("Executable: %s", sys.executable)
    logfunc("Version: %s", sys.version)
    logfunc("End-Python.")
    logfunc("End-Node information.")
    logfunc("----------------------------------------------------------------")
