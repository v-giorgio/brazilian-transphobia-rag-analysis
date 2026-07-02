"""Project configuration loaded from environment variables."""

from os import getenv


YOUTUBE_API_KEY = getenv("YOUTUBE_API_KEY", "")
