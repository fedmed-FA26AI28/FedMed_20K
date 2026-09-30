"""Monitor and log edge device system resources (CPU, GPU, RAM, Temperature) during FL training."""

from monitoring.resource import get_resource_usage, get_device_type

__all__ = ["get_resource_usage", "get_device_type"]
