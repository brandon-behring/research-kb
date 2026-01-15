"""Load testing for research-kb daemon.

This package provides Locust-based load testing for the research-kb daemon
Unix socket interface.

Usage:
    # Interactive mode
    locust -f tests/load/locustfile.py

    # Headless mode for CI
    locust -f tests/load/locustfile.py --headless -u 50 -r 10 --run-time 2m

Requirements:
    pip install locust
"""
