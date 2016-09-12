"""Configuration helpers."""
import re
import logging

LOGGER = logging.getLogger(__name__)


def available_cpu_count():
    """Number of available CPUs for this session."""
    try:
        m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$',  # pylint: disable=invalid-name
                      open('/proc/self/status').read())
        if m:
            res = bin(int(m.group(1).replace(',', ''), 16)).count('1')
            if res > 0:
                return res
    except IOError:
        LOGGER.warn("Could not determine number of allowed CPUs! Falling back to 1!")
        return 1
