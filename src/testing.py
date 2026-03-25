"""Testing and evaluation utilities for floor plan generation models.

Provides test function for evaluating model performance using multiple validation
metrics including parsability, geometry validity, coverage, and room requirements.
"""

from transformers import PreTrainedModel

from datasets import DatasetDict

from src.log_writer import LogWriter
from src.generation import Generator
from src.training_config import TrainingConfig
from src.evaluation_metrics import (
    ParsabilityRate,
    CoverageTest,
    GeomValidityRate,
    RoomsOverlappingTest,
    RequiredRoomsTest
)


def test(
        model: PreTrainedModel,
        tokenizer, dataset: DatasetDict,
        config: TrainingConfig,
        log_writer: LogWriter
    ):
    """Evaluate model performance on validation dataset using multiple metrics.
    
    Generates sequences for the test set and measures parsability, geometry validity,
    coverage, room overlaps, and required room presence. Logs results to TensorBoard
    using LogWriter.
    
    :param model: Pre-trained model for generation
    :param tokenizer: Tokenizer for encoding and decoding
    :param dataset: Dataset split containing test data
    :param config: Training configuration
    :param log_writer: Logger for writing results to TensorBoard
    """
    pars_rate = ParsabilityRate()
    validity_rate = GeomValidityRate()
    cov_rate = CoverageTest()
    room_overlap_rate = RoomsOverlappingTest()
    required_rooms = RequiredRoomsTest()

    generator = Generator(model, tokenizer, dataset)

    for batch in generator.generate_in_batches():
        floor_plans = pars_rate.parse(batch)
        floor_plans = validity_rate.filter_out_invalid(floor_plans)
        cov_rate.measure(floor_plans)
        room_overlap_rate.measure(floor_plans)
        required_rooms.measure(floor_plans)
    
    hparams = config.train_config.copy()
    hparams = hparams | config.model_config.copy()

    hparams.pop("eval_steps", None)
    hparams.pop("log_comment", None)
    hparams.pop("name", None)
    hparams.pop("lr_scheduler", None)

    metrics = {}

    pars_rate.add_to_metrics(metrics)
    validity_rate.add_to_metrics(metrics)
    cov_rate.add_to_metrics(metrics)
    room_overlap_rate.add_to_metrics(metrics)
    required_rooms.add_to_metrics(metrics)

    log_writer.add_hparams(hparams, metrics)
