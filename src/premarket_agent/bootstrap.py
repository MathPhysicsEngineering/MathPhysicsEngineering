"""Application bootstrap logic."""

from __future__ import annotations

from typing import Optional, Tuple

from .agents import (
    AgentController,
    ExecutionAgent,
    MarketIntelAgent,
    MonitoringAgent,
    ResearchAgent,
    ScoringAgent,
)
from .config.settings import Settings, get_settings
from .models import FeatureVectorBuilder
from .services.analysis import RiskManager, ScoringService
from .services.broker import BrokerService, InteractiveBrokersClient
from .services.learning import LearningService
from .services.market_data import MarketDataService
from .services.news import FundamentalsService, NewsService
from .services.portfolio import PortfolioService
from .services.research import ResearchService
from .services.simulation import SimulationService
from .utils import AsyncRateLimiter
from .utils.logging import configure_logging


class ApplicationContext:
    """Aggregated application state for cleanup."""

    def __init__(
        self,
        controller: AgentController,
        market_data: MarketDataService,
        news_service: NewsService,
        fundamentals_service: FundamentalsService,
        broker_service: BrokerService,
        research_service: ResearchService,
        scoring_service: ScoringService,
        risk_manager: RiskManager,
        learning_service: LearningService,
        feature_builder: FeatureVectorBuilder,
        portfolio_service: PortfolioService,
    ) -> None:
        self.controller = controller
        self.market_data = market_data
        self.news_service = news_service
        self.fundamentals_service = fundamentals_service
        self.broker_service = broker_service
        self.research_service = research_service
        self.scoring_service = scoring_service
        self.risk_manager = risk_manager
        self.learning_service = learning_service
        self.feature_builder = feature_builder
        self.portfolio_service = portfolio_service

    async def shutdown(self) -> None:
        await self.market_data.close()
        await self.news_service.close()
        await self.fundamentals_service.close()
        await self.broker_service.shutdown()


def create_system(settings: Optional[Settings] = None) -> Tuple[AgentController, ApplicationContext]:
    """Instantiate the agent system."""
    settings = settings or get_settings()
    configure_logging()

    ib_client = InteractiveBrokersClient(settings)
    yahoo_limiter = AsyncRateLimiter(settings.yahoo_requests_per_second)
    market_data = MarketDataService(
        settings=settings,
        ib_client=ib_client,
        rate_limiter=yahoo_limiter,
    )
    news_service = NewsService(rate_limiter=yahoo_limiter)
    fundamentals_service = FundamentalsService(rate_limiter=yahoo_limiter)
    research_service = ResearchService(
        settings=settings,
        news_service=news_service,
        fundamentals_service=fundamentals_service,
    )
    feature_builder = FeatureVectorBuilder()
    learning_service = LearningService(feature_builder=feature_builder)
    risk_manager = RiskManager(settings=settings)
    scoring_service = ScoringService(
        settings=settings,
        risk_manager=risk_manager,
        learning=learning_service,
    )
    broker_service = BrokerService(settings=settings, ib_client=ib_client)
    portfolio_service = PortfolioService(broker=broker_service, market_data=market_data)

    controller = AgentController(
        settings=settings,
        agents=[
            MarketIntelAgent(market_data=market_data),
            ResearchAgent(
                research_service=research_service,
                max_remote_symbols=settings.max_remote_research_symbols,
            ),
            ScoringAgent(scoring_service=scoring_service, risk_manager=risk_manager),
            ExecutionAgent(broker_service=broker_service),
            MonitoringAgent(
                broker_service=broker_service,
                market_data=market_data,
                risk_manager=risk_manager,
            ),
        ],
    )

    context = ApplicationContext(
        controller=controller,
        market_data=market_data,
        news_service=news_service,
        fundamentals_service=fundamentals_service,
        broker_service=broker_service,
        research_service=research_service,
        scoring_service=scoring_service,
        risk_manager=risk_manager,
        learning_service=learning_service,
        feature_builder=feature_builder,
        portfolio_service=portfolio_service,
    )
    return controller, context


def create_simulation(settings: Optional[Settings] = None) -> Tuple[SimulationService, ApplicationContext]:
    """Utility factory for the simulation workflow."""
    _, context = create_system(settings)
    simulation = SimulationService(
        settings=settings or get_settings(),
        market_data=context.market_data,
        research=context.research_service,
        scoring=context.scoring_service,
        broker=context.broker_service,
        learning=context.learning_service,
        portfolio=context.portfolio_service,
    )
    return simulation, context


