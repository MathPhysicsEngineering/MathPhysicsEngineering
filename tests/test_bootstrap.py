import asyncio

from premarket_agent.bootstrap import create_system
from premarket_agent.config.settings import get_settings


def test_create_system() -> None:
    controller, context = create_system(get_settings())
    assert controller is not None
    assert context.market_data is not None
    assert context.research_service is not None
    assert context.learning_service is not None
    assert context.portfolio_service is not None


