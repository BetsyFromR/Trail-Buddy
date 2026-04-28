from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class UserProfile(TypedDict, total=False):
    half_marathon_pr: str
    longest_trail: str
    experience_level: str
    target_race: str
    target_distance_km: float
    target_elevation_m: float
    location: str
    language: str


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    profile: UserProfile
    retrieved: list[str]
