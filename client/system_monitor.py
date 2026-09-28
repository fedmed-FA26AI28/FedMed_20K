"""Theo dõi và ghi log tài nguyên hệ thống của Jetson (CPU, GPU, RAM, Nhiệt độ) trong quá trình train FL."""

from monitoring.resource import get_resource_usage, get_device_type

__all__ = ["get_resource_usage", "get_device_type"]
